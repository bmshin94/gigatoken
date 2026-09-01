//! Pretokenization: split documents into pretokens following a tokenizer's
//! pretokenization regex.
//!
//! The implementations live in `fast` (one submodule per scheme:
//! `fast::r50k` for GPT-2, `fast::cl100k` for GPT-4, ...), selected via
//! [`PretokenizerType`].
//!
//! The main entry points are:
//! - `pretokenize_as_iter`: iterate pretokens of a `&[u8]` (r50k scheme)
//! - `PretokenizerType::pretokenize`: iterate pretokens of any scheme
//! - `pretokenize_par_bytes`: parallel pretokenization with document splitting and counting

pub(crate) use crate::pretokenize::pretoken::Pretoken;
use crate::input::par_document_chunks;
use rayon::prelude::*;
use std::collections::HashMap;

pub mod fast;
mod options;
mod pretoken;
mod unicode;

pub use fast::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer,
};
pub use options::{FastPretokenizerDispatch, PretokenizerType};

/// Default document separator used in common training corpora.
pub const DEFAULT_SEPARATOR: &[u8] = b"<|endoftext|>";

/// Iterate the pretokens of `bytes` using the production (r50k) pretokenizer.
#[inline]
pub fn pretokenize_as_iter(bytes: &[u8]) -> FastR50kPretokenizer<'_> {
    FastR50kPretokenizer::new(bytes)
}

// Batched pretoken pulling (the encode loop's input interface)

/// Chunk size of [`PretokenSpans::fill_spans_keyed`] — the live entries of
/// one [`SpanBatch`] fill.
pub const PRETOKEN_CHUNK: usize = 256;

/// Both 64-bit halves of the per-length pack mask in scalar ALU ops, for
/// the latency-chained per-span paths (a dependent table load measured
/// worse there); `fast::mask::PACK_MASK_TABLE` is built from this `const fn`.
#[inline(always)]
pub(crate) const fn pack_mask_halves(n: usize) -> (u64, u64) {
    debug_assert!(n >= 1 && n <= 15);
    let s = (n * 8) as u32;
    let lo = if n < 8 { u64::MAX >> (64u32.wrapping_sub(s) & 63) } else { u64::MAX };
    let hi = if n > 8 { u64::MAX >> (128u32.wrapping_sub(s) & 63) } else { 0 };
    (lo, hi)
}

// Key packing and token-lane stores read/write native-endian words and
// assume little-endian layouts.
#[cfg(target_endian = "big")]
compile_error!("gigatoken's key packing and token-lane stores assume little-endian byte order");

/// Pack a pretoken of ≤ 15 bytes into a `u128` cache key: bytes in the low
/// 15 lanes, length in the top byte (so keys never collide across lengths
/// and a real key is never 0). `None` for longer pretokens (slice-keyed
/// fallback map). One unaligned 16-byte load + mask when the load cannot
/// cross a page boundary, else a plain copy; both give the identical key.
#[inline(always)]
pub(crate) fn pack_pretoken_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n > 15 {
        return None;
    }
    if n == 0 {
        // Empty pretokens (public API only) take key 0, the long-map route.
        return Some(0);
    }
    let p = bytes.as_ptr();
    let low = if (p as usize) & 4095 <= 4096 - 16 {
        // SAFETY: the offset within the (≥ 4096-byte) page is ≤ 4096 - 16,
        // so a 16-byte read stays inside the page holding `p`, which is
        // mapped because `p` points to at least one valid byte.
        let v = unsafe { (p as *const u128).read_unaligned() };
        let (mask_lo, mask_hi) = pack_mask_halves(n);
        ((v as u64 & mask_lo) as u128) | ((((v >> 64) as u64 & mask_hi) as u128) << 64)
    } else {
        // Rare: `p` is within 16 bytes of a page boundary. Gather with a
        // plain copy (≤ 15 bytes) — correctness over speed on this cold
        // path. Lanes past `n` stay zero, so no mask is needed.
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(bytes);
        u128::from_le_bytes(lanes)
    };
    Some(low | ((n as u128) << 120))
}

/// Hash of a packed pretoken key. Every consumer (fill loops, cache
/// rehash, vocab seeding) must hash a key the same way: the arms below
/// differ and may never mix in one process. aarch64 picks its arm at
/// compile time; x86_64 per process via [`crc_hash_selected`] (the hot
/// fill loops embed one arm per monomorphization, see [`fill_span_hash`]).
/// Every arm maps key 0 to hash 0.
#[inline(always)]
pub(crate) fn pretoken_key_hash(key: u128) -> u64 {
    // aarch64-unknown-linux-gnu needs `-C target-feature=+crc` for this arm.
    #[cfg(all(target_arch = "aarch64", target_feature = "crc"))]
    {
        // Hardware CRC32: linear over GF(2), so the low bits (the table
        // index) see every key bit.
        use core::arch::aarch64::__crc32d;
        // SAFETY: gated on the `crc` target feature at compile time.
        unsafe { __crc32d(__crc32d(0, key as u64), (key >> 64) as u64) as u64 }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crc_hash_selected() {
            // SAFETY: `crc_hash_selected` verified SSE4.2 support (or the
            // build enables it statically, folding this branch away).
            unsafe { pretoken_key_hash_crc32c(key) }
        } else {
            pretoken_key_hash_fold(key)
        }
    }
    #[cfg(not(any(all(target_arch = "aarch64", target_feature = "crc"), target_arch = "x86_64")))]
    {
        pretoken_key_hash_fold(key)
    }
}

/// The multiply-fold arm of [`pretoken_key_hash`], used wherever no
/// hardware CRC arm applies. Maps key 0 to hash 0.
#[allow(dead_code)] // no cfg arm references it under aarch64 + crc
#[inline(always)]
fn pretoken_key_hash_fold(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    let mut h = (lo ^ hi.rotate_right(25)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 32;
    h
}

/// The hardware CRC32C (SSE4.2) arm of [`pretoken_key_hash`]. `sse4.2` is
/// not baseline x86-64, so it is selected per process by
/// [`crc_hash_selected`] rather than at compile time.
///
/// # Safety
///
/// The CPU must support SSE4.2: reached only after [`crc_hash_selected`]
/// returned true, directly or via a `fill_span_hash::<true>` instantiation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline]
unsafe fn pretoken_key_hash_crc32c(key: u128) -> u64 {
    use core::arch::x86_64::_mm_crc32_u64;
    // SAFETY: SSE4.2 is enabled on this function; the caller (per the
    // contract above) only reaches it on a CPU that has it.
    unsafe { _mm_crc32_u64(_mm_crc32_u64(0, key as u64), (key >> 64) as u64) }
}

/// Does this process hash pretoken keys with CRC32C (x86_64)? A pure
/// function of the CPU, so every hash site branching on it agrees for the
/// process lifetime. A cached atomic load after the first call.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn crc_hash_selected() -> bool {
    std::arch::is_x86_feature_detected!("sse4.2")
}

/// Per-span hash for the monomorphized fill-loop bodies:
/// [`pretoken_key_hash`] with the x86_64 per-process selection hoisted out
/// of the per-span path. Reachability rule: `X86_CRC = true` is
/// instantiated only inside the sse4.2-gated fill wrappers, entered only
/// when [`crc_hash_selected`]; `false` only on the other dispatch arm. Off
/// x86_64 this IS [`pretoken_key_hash`].
#[inline(always)]
pub(crate) fn fill_span_hash<const X86_CRC: bool>(key: u128) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if X86_CRC {
            debug_assert!(crc_hash_selected());
            // SAFETY: the `true` instantiation is only reachable from the
            // sse4.2-gated fill wrappers (reachability rule above).
            unsafe { pretoken_key_hash_crc32c(key) }
        } else {
            pretoken_key_hash_fold(key)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        pretoken_key_hash(key)
    }
}

/// One batch slot: a pretoken span with its packed cache key and key hash,
/// as one 32-byte record (AoS: one store stream per fill, one cache line
/// per probe). `meta` is keyed on `key`:
/// - `key != 0` (short pretoken, ≤ 15 bytes): the full 64-bit key hash;
///   the span length rides in the key's top byte.
/// - `key == 0` (long pretoken, or an empty span through the public
///   adapter): the span length in bytes (the long route never probes the
///   short table; prefetching it as a hash is harmless).
///
/// Fields are `pub(crate)`: only the in-crate fill loops may write entries
/// (see the [`PretokenSpans`] safety contract).
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub(crate) struct BatchEntry {
    pub(crate) key: u128,
    pub(crate) ptr: *const u8,
    pub(crate) meta: u64,
}

const _: () = assert!(std::mem::size_of::<BatchEntry>() == 32);

impl BatchEntry {
    /// Span length, independent of the short/long route.
    #[inline(always)]
    pub(crate) fn span_len(&self) -> usize {
        if self.key != 0 { (self.key >> 120) as usize } else { self.meta as usize }
    }
}

/// Readable slack entries past a full chunk, so the emit loop's
/// prefetch-ahead `entries[i + D].meta` load needs no index clamp (never
/// written by a fill; a stale prefetch is harmless).
pub(crate) const SPAN_BATCH_SLACK: usize = 16;

/// One chunk of pretoken spans with their packed cache keys and hashes
/// ([`BatchEntry`]), filled by [`PretokenSpans::fill_spans_keyed`]. Fills
/// only ever write the first [`PRETOKEN_CHUNK`] entries.
pub struct SpanBatch<'a> {
    /// `pub(crate)`: writable only by the in-crate fill loops, which uphold
    /// the [`PretokenSpans`] safety contract on every entry they write.
    pub(crate) entries: [BatchEntry; PRETOKEN_CHUNK + SPAN_BATCH_SLACK],
    /// The entries' `ptr`s borrow the spans' backing storage.
    _spans: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> SpanBatch<'a> {
    pub fn new() -> Self {
        SpanBatch {
            entries: [BatchEntry { key: 0, ptr: std::ptr::null(), meta: 0 };
                PRETOKEN_CHUNK + SPAN_BATCH_SLACK],
            _spans: std::marker::PhantomData,
        }
    }

    /// Reconstruct entry `i`'s span.
    ///
    /// # Safety
    /// Entry `i` must have been written by the most recent fill (`i` below
    /// its returned count), so `ptr` still points at a live span of `'a`.
    #[inline(always)]
    pub unsafe fn span(&self, i: usize) -> &'a [u8] {
        let e = &self.entries[i];
        // SAFETY: per the contract, `ptr` points at `span_len()` live bytes.
        unsafe { std::slice::from_raw_parts(e.ptr, e.span_len()) }
    }
}

impl Default for SpanBatch<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// A source of pretoken spans, pulled a chunk at a time with their cache
/// keys derived on the way out. `Tokenizer::memoized_encode` uses this
/// instead of `Iterator` so pulling happens in a dedicated out-of-line
/// loop where the pretokenizer state stays register-allocated, and the
/// key math fills the serial span walker's idle issue slots.
///
/// # Safety
///
/// The consumer trusts every fill unconditionally: after
/// [`Self::fill_spans_keyed`] returns `n`, the emit loop calls
/// [`SpanBatch::span`] (a raw `from_raw_parts`) on any entry `i < n`.
/// Implementations must therefore uphold, for every call returning `n`:
///
/// - `batch.entries[0..n]` were all written by THIS call (no stale entries
///   counted), and
/// - each written entry holds a valid `(ptr, len)` into caller-live input
///   bytes of lifetime `'a`: `ptr` non-null and readable for
///   [`BatchEntry::span_len`] bytes, with `key`/`meta` derived from exactly
///   those bytes via `pack_pretoken_key`/`pretoken_key_hash` semantics.
///
/// The entry fields are `pub(crate)`, so implementations outside this
/// crate cannot write entries at all and can only soundly return 0.
pub unsafe trait PretokenSpans<'a> {
    /// Fill `batch` from the front with the next pretoken spans, calling
    /// `prefetch(hash)` for each. Returns how many were written; a short
    /// count (including 0) means the input is exhausted. (Storing the hash
    /// rather than recomputing it at the consumer measured 4% faster.)
    fn fill_spans_keyed(&mut self, batch: &mut SpanBatch<'a>, prefetch: &impl Fn(u64)) -> usize;
}

/// Shared body of the iterator-backed [`PretokenSpans`] implementations
/// (spans with no single backing buffer; sources that walk one slice use
/// [`fill_spans_keyed_with_buf`]). No production caller walks this path,
/// so it takes no CRC-monomorphized wrapper.
#[inline(always)]
pub(crate) fn fill_spans_keyed_with<'a>(
    mut next: impl FnMut() -> Option<&'a [u8]>,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    let mut n = 0;
    while n < PRETOKEN_CHUNK {
        let Some(span) = next() else { break };
        let (key, h) = match pack_pretoken_key(span) {
            Some(key) => (key, pretoken_key_hash(key)),
            None => (0, 0),
        };
        prefetch(h);
        // Long (and empty) spans record their length; short spans their
        // hash (see `BatchEntry::meta`).
        let meta = if key != 0 { h } else { span.len() as u64 };
        batch.entries[n] = BatchEntry { key, ptr: span.as_ptr(), meta };
        n += 1;
    }
    n
}

/// [`fill_spans_keyed_with`] for span sources that walk a single backing
/// slice, with `next` yielding `(start, end)` (`start < end`) into `bytes`.
/// Knowing the buffer turns [`pack_pretoken_key`]'s two data-dependent
/// branches into a select and one hoisted buffer-end bound.
#[inline(always)]
pub(crate) fn fill_spans_keyed_with_buf<'a>(
    bytes: &'a [u8],
    next: impl FnMut() -> Option<(usize, usize)>,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    // Hash-arm dispatch, once per fill (see `fill_span_hash`).
    #[cfg(target_arch = "x86_64")]
    if crc_hash_selected() {
        // SAFETY: `crc_hash_selected` verified SSE4.2 support.
        return unsafe { fill_spans_keyed_with_buf_crc(bytes, next, batch, prefetch) };
    }
    fill_spans_keyed_with_buf_impl::<false>(bytes, next, batch, prefetch)
}

/// The SSE4.2 (CRC-hash) monomorphization of [`fill_spans_keyed_with_buf`].
///
/// # Safety
///
/// The CPU must support SSE4.2 ([`crc_hash_selected`] must have returned
/// true).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn fill_spans_keyed_with_buf_crc<'a>(
    bytes: &'a [u8],
    next: impl FnMut() -> Option<(usize, usize)>,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    fill_spans_keyed_with_buf_impl::<true>(bytes, next, batch, prefetch)
}

/// [`fill_spans_keyed_with_buf`]'s loop body, monomorphized on the hash
/// arm (`X86_CRC` — see [`fill_span_hash`]'s reachability contract).
#[inline(always)]
fn fill_spans_keyed_with_buf_impl<'a, const X86_CRC: bool>(
    bytes: &'a [u8],
    mut next: impl FnMut() -> Option<(usize, usize)>,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    // A 16-byte load at `start` is in bounds iff `start < tail_lim`.
    let tail_lim = bytes.len().saturating_sub(15);
    let mut n = 0;
    while n < PRETOKEN_CHUNK {
        let Some((start, end)) = next() else { break };
        debug_assert!(start < end && end <= bytes.len());
        // SAFETY: `next` returns in-bounds span boundaries.
        let span = unsafe { bytes.get_unchecked(start..end) };
        let len = end - start;
        let long = len > 15;
        // `|` not `||`: one combined test, taken for all but the tail spans.
        let key = if long | (start < tail_lim) {
            // SAFETY: `start < tail_lim` puts `start + 16` within `bytes`;
            // a long span (len ≥ 16) contains its own first 16 bytes.
            let v = unsafe { (bytes.as_ptr().add(start) as *const u128).read_unaligned() };
            let m = len.min(15);
            let (mask_lo, mask_hi) = pack_mask_halves(m);
            let lo = (v as u64) & mask_lo;
            let hi = ((v >> 64) as u64 & mask_hi) | ((m as u64) << 56);
            let packed = (lo as u128) | ((hi as u128) << 64);
            if long { 0 } else { packed }
        } else {
            // Short span starting in the buffer's last 15 bytes: gather
            // with a plain copy, once per input. Lanes past `len` stay
            // zero, so no mask is needed.
            let mut lanes = [0u8; 16];
            lanes[..len].copy_from_slice(span);
            u128::from_le_bytes(lanes) | ((len as u128) << 120)
        };
        let h = fill_span_hash::<X86_CRC>(key);
        prefetch(h);
        // Long spans record their length instead of the (unused) hash —
        // see `BatchEntry::meta`.
        let meta = if long { len as u64 } else { h };
        batch.entries[n] = BatchEntry { key, ptr: span.as_ptr(), meta };
        n += 1;
    }
    n
}

/// Adapter giving any pretoken iterator the [`PretokenSpans`] interface.
/// The `Fast*` pretokenizers implement the trait directly over their
/// walker state instead (routing them through `Iterator::next` measured
/// ~23% of warm encode time in call overhead).
pub struct SpanIter<I>(pub I);

// SAFETY: delegates to `fill_spans_keyed_with`, which writes exactly the
// first `n` entries from the iterator's live `'a` spans.
unsafe impl<'a, I: Iterator<Item = Pretoken<'a>>> PretokenSpans<'a> for SpanIter<I> {
    #[inline(never)]
    fn fill_spans_keyed(&mut self, batch: &mut SpanBatch<'a>, prefetch: &impl Fn(u64)) -> usize {
        fill_spans_keyed_with(|| self.0.next().map(|p| p.0), batch, prefetch)
    }
}

// Pretoken-safe document splitting

/// Split `bytes` into ranges of roughly `target` bytes whose boundaries are
/// pretoken boundaries under every supported pretokenization scheme, so
/// encoding the ranges independently and concatenating the token streams is
/// identical to encoding `bytes` in one pass.
///
/// A boundary sits on a space preceded by an ASCII alphanumeric and
/// followed by an ASCII letter. No scheme's pretoken can cross such a
/// point: whitespace attaches only as a single leading space of a
/// following word, trailing attachments (`[\r\n]*`) contain no space, and
/// the all-whitespace rules never see a run crossing it.
///
/// `added_tokens` (byte sequence, `rstrip` flag) are matched atomically
/// before pretokenization, so a boundary is rejected when an occurrence
/// straddles it (only tokens containing a space can) or when an `rstrip`
/// occurrence ends exactly at it (the boundary space would otherwise
/// encode as plain text instead of being absorbed). `lstrip` needs no
/// counterpart: a boundary space before a match is trimmed identically at
/// the start of the next chunk.
pub fn safe_split_ranges(
    bytes: &[u8],
    target: usize,
    added_tokens: &[(&[u8], bool)],
) -> Vec<std::ops::Range<usize>> {
    let blockers: Vec<memchr::memmem::Finder> = added_tokens
        .iter()
        .filter(|(t, _)| t.contains(&b' '))
        .map(|(t, _)| memchr::memmem::Finder::new(t))
        .collect();
    let max_blocker = blockers.iter().map(|f| f.needle().len()).max().unwrap_or(0);
    // rstrip tokens can only end at a boundary when their last byte is the
    // alphanumeric byte before the boundary space.
    let end_blockers: Vec<&[u8]> = added_tokens
        .iter()
        .filter(|(t, rstrip)| *rstrip && t.last().is_some_and(u8::is_ascii_alphanumeric))
        .map(|&(t, _)| t)
        .collect();
    // Whether an added-token occurrence spans the cut between `p - 1` and
    // `p`. Such an occurrence must start within `max_blocker - 1` bytes
    // before `p`, so searching a window of that radius is exhaustive.
    let cuts_added_token = |p: usize| -> bool {
        let lo = p.saturating_sub(max_blocker.saturating_sub(1));
        let hi = (p + max_blocker.saturating_sub(1)).min(bytes.len());
        blockers.iter().any(|f| {
            f.find_iter(&bytes[lo..hi])
                .any(|s| lo + s < p && lo + s + f.needle().len() > p)
        })
    };
    // Whether an rstrip added-token occurrence ends exactly at `p`.
    let ends_rstrip_token = |p: usize| end_blockers.iter().any(|t| bytes[..p].ends_with(t));
    let len = bytes.len();
    let target = target.max(1);
    let mut out = Vec::new();
    let mut start = 0;
    'chunks: while start < len {
        let mut probe = start + target;
        while probe + 1 < len {
            if bytes[probe] == b' '
                && bytes[probe - 1].is_ascii_alphanumeric()
                && bytes[probe + 1].is_ascii_alphabetic()
                && !(max_blocker > 0 && cuts_added_token(probe))
                && !(!end_blockers.is_empty() && ends_rstrip_token(probe))
            {
                out.push(start..probe);
                start = probe;
                continue 'chunks;
            }
            probe += 1;
        }
        out.push(start..len);
        break;
    }
    if out.is_empty() {
        out.push(0..0);
    }
    out
}

// Parallel pretokenization with document splitting

/// Pretokenize `bytes` in parallel, splitting documents on `separator`.
/// Returns a map of pretoken → count.
pub fn pretokenize_par_bytes<'a>(
    bytes: &'a [u8],
    separator: &'a [u8],
) -> HashMap<Pretoken<'a>, usize, rustc_hash::FxBuildHasher> {
    let chunks = par_document_chunks(bytes, separator, rayon::current_num_threads());
    chunks
        .into_par_iter()
        .map(|docs| {
            let mut counts = HashMap::default();
            for doc in docs {
                for p in FastR50kPretokenizer::new(doc) {
                    *counts.entry(p).or_default() += 1;
                }
            }
            counts
        })
        .reduce(HashMap::default, |mut acc, m| {
            if acc.is_empty() {
                return m;
            }
            for (k, v) in m {
                *acc.entry(k).or_default() += v;
            }
            acc
        })
}

#[cfg(test)]
mod test {
    use super::*;
    use fast::test_support::{assert_matches_regex, load_owt_prefix};

    const GPT2_REGEX: &str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    /// `safe_split_ranges` must produce boundaries that no pretoken crosses,
    /// for every supported scheme: pretokenizing the ranges independently and
    /// concatenating must equal pretokenizing the whole input in one pass.
    #[test]
    fn test_safe_split_ranges_pretoken_equivalent() {
        let input = load_owt_prefix(2_000_000);

        let ranges = safe_split_ranges(&input, 10_000, &[]);
        assert!(ranges.len() > 100, "expected many splits, got {}", ranges.len());
        // Ranges must cover the input contiguously.
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, input.len());
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }

        fn collect<'a, I: Iterator<Item = Pretoken<'a>>>(it: I) -> Vec<&'a [u8]> {
            it.map(|p| p.0).collect()
        }

        macro_rules! check_scheme {
            ($name:literal, $ctor:path) => {
                let whole = collect($ctor(&input));
                let split: Vec<&[u8]> = ranges
                    .iter()
                    .flat_map(|r| collect($ctor(&input[r.clone()])))
                    .collect();
                assert_eq!(whole, split, "scheme {} differs across safe splits", $name);
            };
        }
        check_scheme!("r50k", FastR50kPretokenizer::new);
        check_scheme!("cl100k", FastCl100kPretokenizer::new);
        check_scheme!("qwen2", FastQwen2Pretokenizer::new);
        check_scheme!("qwen3_5", FastQwen35Pretokenizer::new);
        check_scheme!("olmo3", FastOlmo3Pretokenizer::new);
        check_scheme!("deepseek_v3", FastDeepSeekV3Pretokenizer::new);
    }

    /// Deterministic LCG word soup so word lengths vary and split probes
    /// hit `special` (followed by `sep`) at every possible phase.
    fn lcg_words(special: &[u8], sep: &[u8], period: usize) -> Vec<u8> {
        let mut rng = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as usize
        };
        let words: [&[u8]; 5] = [b"alpha ", b"be ", b"gamma7 ", b"x ", b"delta "];
        let mut input = Vec::new();
        for _ in 0..4000 {
            input.extend_from_slice(words[next() % words.len()]);
            if next() % period == 0 {
                input.extend_from_slice(special);
                input.extend_from_slice(sep);
            }
        }
        input
    }

    /// Boundaries must never cut an occurrence of a space-containing added
    /// token, while splitting still proceeds elsewhere in the document.
    #[test]
    fn test_safe_split_ranges_avoids_added_tokens() {
        let special: &[u8] = b"<|multi word special|>";
        let input = lcg_words(special, b"", 9);

        let ranges = safe_split_ranges(&input, 300, &[(special, false)]);
        assert!(ranges.len() > 50, "expected many splits, got {}", ranges.len());
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, input.len());
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }

        let occurrences: Vec<usize> =
            memchr::memmem::find_iter(&input, special).collect();
        assert!(!occurrences.is_empty());
        let cuts_occurrence = |p: usize| {
            occurrences.iter().any(|&s| s < p && s + special.len() > p)
        };
        for r in &ranges[1..] {
            assert!(!cuts_occurrence(r.start), "boundary {} cuts an occurrence", r.start);
        }

        // The input must actually tempt the splitter: without the
        // added-token check, some boundary lands inside an occurrence.
        let unaware = safe_split_ranges(&input, 300, &[]);
        assert!(
            unaware[1..].iter().any(|r| cuts_occurrence(r.start)),
            "test input never places a naive boundary inside the special token"
        );
    }

    /// A boundary must never sit right after an rstrip added-token
    /// occurrence: the boundary space would open the next chunk as plain
    /// text instead of being absorbed by the token's rstrip.
    #[test]
    fn test_safe_split_ranges_avoids_rstrip_token_ends() {
        let special: &[u8] = b"TOK1";
        let input = lcg_words(special, b" ", 6);
        let ends_at = |p: usize| p >= special.len() && &input[p - special.len()..p] == special;

        let ranges = safe_split_ranges(&input, 200, &[(special, true)]);
        assert!(ranges.len() > 50, "expected many splits, got {}", ranges.len());
        for r in &ranges[1..] {
            assert!(!ends_at(r.start), "boundary {} follows an rstrip token", r.start);
        }

        // The input must actually tempt the splitter: without the rstrip
        // flag, some boundary lands right after an occurrence.
        let unaware = safe_split_ranges(&input, 200, &[(special, false)]);
        assert!(
            unaware[1..].iter().any(|r| ends_at(r.start)),
            "test input never places a naive boundary after the rstrip token"
        );
    }

    /// The production (fast r50k) pretokenizer against the GPT-2 reference
    /// regex on ~5 MB of OWT, token by token.
    #[test]
    fn test_pretokenizer_matches_regex_owt() {
        let input = load_owt_prefix(5_000_000);
        let text = std::str::from_utf8(&input).unwrap();
        assert_matches_regex(GPT2_REGEX, text, pretokenize_as_iter(&input));
    }
}

#[cfg(test)]
mod span_source_tests {
    use super::*;

    fn check_source<'a>(
        mut src: impl PretokenSpans<'a>,
        reference: impl Iterator<Item = Pretoken<'a>>,
        scheme: &str,
    ) {
        let expected: Vec<&[u8]> = reference.map(|p| p.0).collect();
        let mut got: Vec<&[u8]> = Vec::new();
        let mut batch = SpanBatch::new();
        let prefetched = std::cell::Cell::new(0usize);
        loop {
            let n = src.fill_spans_keyed(&mut batch, &|_h| {
                prefetched.set(prefetched.get() + 1)
            });
            for i in 0..n {
                // SAFETY: i < n, the count just returned by the fill.
                let span = unsafe { batch.span(i) };
                let (want_key, want_hash) = match pack_pretoken_key(span) {
                    Some(key) => (key, pretoken_key_hash(key)),
                    None => (0, 0),
                };
                let e = &batch.entries[i];
                assert_eq!(e.key, want_key, "{scheme}: bad key for {span:?}");
                let want_meta = if want_key != 0 {
                    want_hash
                } else {
                    span.len() as u64
                };
                assert_eq!(e.meta, want_meta, "{scheme}: bad meta for {span:?}");
                got.push(span);
            }
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
        assert_eq!(prefetched.get(), got.len(), "{scheme}: one prefetch per span");
        assert_eq!(
            got.len(),
            expected.len(),
            "{scheme}: span count mismatch"
        );
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                g, e,
                "{scheme}: span {i} diverged: {:?} vs {:?}",
                String::from_utf8_lossy(g),
                String::from_utf8_lossy(e)
            );
        }
    }

    /// [`check_source`] for every mask-scanner scheme on `b`.
    fn check_all_mask_schemes(b: &[u8]) {
        check_source(FastR50kPretokenizer::new(b), FastR50kPretokenizer::new(b), "r50k");
        check_source(
            FastCl100kPretokenizer::new(b),
            FastCl100kPretokenizer::new(b),
            "cl100k",
        );
        check_source(FastQwen2Pretokenizer::new(b), FastQwen2Pretokenizer::new(b), "qwen2");
        check_source(
            FastQwen35Pretokenizer::new(b),
            FastQwen35Pretokenizer::new(b),
            "qwen3_5",
        );
        check_source(FastOlmo3Pretokenizer::new(b), FastOlmo3Pretokenizer::new(b), "olmo3");
        check_source(
            FastDeepSeekV3Pretokenizer::new(b),
            FastDeepSeekV3Pretokenizer::new(b),
            "deepseek_v3",
        );
    }

    /// Every scheme's chunked `fill_spans_keyed` must reproduce its
    /// iterator's spans exactly, with keys/hashes derived per the shared
    /// helpers, one prefetch per span — including chunk-boundary and
    /// end-of-input handling (buffer sizes straddle multiples of
    /// PRETOKEN_CHUNK spans).
    #[test]
    fn fill_spans_keyed_matches_iterator_all_schemes() {
        let pieces: &[&str] = &[
            "word", " word", "12", " 345678", "'ll", "'s", " ", "  ", "\n", "\r\n", "\t",
            "!", " ?!", "(", "caf\u{e9}", "\u{65e5}\u{672c}", "\u{1F389}", "\u{00A0}",
            "\u{2003}", "a", " I", "don't", "one two three four five", "\u{4e2d}\u{6587}",
            "\u{30ab}\u{30bf}", "123456", " longpretokenword", "supercalifragilistic",
        ];
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..60 {
            // Vary length so end-of-input lands at many chunk offsets.
            let target = 40 + round * 97;
            let mut buf = Vec::new();
            while buf.len() < target {
                buf.extend_from_slice(pieces[(rng() % pieces.len() as u64) as usize].as_bytes());
            }
            let b: &[u8] = &buf;
            check_all_mask_schemes(b);
            check_source(
                SpanIter(FastR50kPretokenizer::new(b)),
                FastR50kPretokenizer::new(b),
                "span_iter",
            );
            for pt in [
                PretokenizerType::GPT2,
                PretokenizerType::GPT4,
                PretokenizerType::Qwen2,
                PretokenizerType::Qwen35,
                PretokenizerType::Olmo3,
                PretokenizerType::DeepSeekV3,
                PretokenizerType::Kimi,
            ] {
                check_source(pt.pretokenize(b), pt.pretokenize(b), "dispatch");
            }
        }
    }

    /// Long runs hit the two-phase walker's rare paths: a pretoken longer
    /// than its u16 offset window (direct single-span emit, both the
    /// clean-batch and bad-zone variants), a scalar overrun whose end
    /// leaves the window mid-fill (dropped and re-derived next fill), and
    /// maximum-density boundary batches straddling fill boundaries.
    #[test]
    fn fill_spans_keyed_long_runs_all_schemes() {
        let mut cases: Vec<Vec<u8>> = Vec::new();
        // > 64 KB letter run between ordinary words.
        let mut v = b"intro words ".to_vec();
        v.extend(std::iter::repeat_n(b'a', 70_000));
        v.extend_from_slice(b" tail words");
        cases.push(v);
        // > 64 KB space run: boundaries only at the run's edges.
        let mut v = b"x".to_vec();
        v.extend(std::iter::repeat_n(b' ', 70_000));
        v.extend_from_slice(b"y z");
        cases.push(v);
        // > 64 KB of 3-byte unicode whitespace: batch-edge straddles make
        // bad zones, and the scalar walk overruns the whole run.
        let mut v = b"hello world ".repeat(4);
        for _ in 0..25_000 {
            v.extend_from_slice("\u{2003}".as_bytes());
        }
        v.extend_from_slice(b"end");
        cases.push(v);
        // Monster token flush at end of input.
        cases.push(std::iter::repeat_n(b'a', 70_000).collect());
        // Single-byte token alternation (64 boundaries per batch), then a
        // digit run some schemes split every 3 chars.
        let mut v = b"a1".repeat(2_000);
        v.extend(std::iter::repeat_n(b'7', 40_000));
        v.extend_from_slice(b" end");
        cases.push(v);
        for case in &cases {
            check_all_mask_schemes(case);
        }
    }
}
