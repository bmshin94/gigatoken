pub(crate) mod pretoken_cache;
pub mod sentencepiece;
pub mod tiktoken;

/// Ask the kernel for 2 MiB pages over `[ptr, ptr + bytes)` before first
/// touch (fewer faults, and Zen drops software prefetches that miss the
/// TLB). A hint only; no-op off Linux. Lives here so the bin target's
/// module tree sees it too.
pub(crate) fn madvise_hugepage(ptr: *mut u8, bytes: usize) {
    #[cfg(target_os = "linux")]
    {
        // Align inward: malloc pointers sit 16 B past the page boundary and
        // madvise returns EINVAL on an unaligned start.
        const PAGE: usize = 4096;
        let start = (ptr as usize + PAGE - 1) & !(PAGE - 1);
        let end = (ptr as usize).saturating_add(bytes);
        if end > start {
            // SAFETY: the range lies within one live allocation, and the
            // hint does not read or write the memory.
            unsafe {
                libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (ptr, bytes);
}

use crate::token::TokenId;
use eyre::{Result, anyhow};
use std::collections::HashMap;

#[derive(Clone)]
pub struct ByteRemapping {
    /// Maps each byte value to the token ID of its single-byte vocab entry.
    /// The IDs need not be < 256 (e.g. DeepSeek puts its byte tokens at
    /// 3..=258, after the special tokens).
    mapping: Vec<TokenId>,
}

/// Whether `b` can appear anywhere in a valid UTF-8 byte stream. `0xC0`/`0xC1`
/// are overlong two-byte leads and `0xF5..=0xFF` encode code points beyond
/// U+10FFFF, so none of them ever occur in valid UTF-8.
fn is_valid_utf8_byte(b: u8) -> bool {
    !matches!(b, 0xC0 | 0xC1 | 0xF5..=0xFF)
}

impl ByteRemapping {
    /// Build the byte → token-ID table by scanning `vocab` for single-byte
    /// entries (lowest ID wins). Returns `None` when the mapping is the
    /// identity (token ID == byte value), and an error if some byte value
    /// that can appear in valid UTF-8 has no single-byte token.
    ///
    /// A vocab may legitimately omit single-byte tokens for the bytes that
    /// never occur in valid UTF-8 (`0xC0`, `0xC1`, and `0xF5..=0xFF` — overlong
    /// and out-of-range lead bytes). Byte-level vocabularies trained only on
    /// UTF-8 text — e.g. ModernBERT / GPT-NeoX — drop them. Since such a byte
    /// can never reach the merge loop from valid input, its absence is not an
    /// error; we fill it with a placeholder ID so the table can never yield an
    /// out-of-range `TokenId` even if fed malformed bytes.
    pub fn from_byte_vocab(vocab: &[impl AsRef<[u8]>]) -> Result<Option<Self>> {
        const UNSET: u32 = u32::MAX;
        let mut mapping = vec![TokenId(UNSET); 256];
        for (id, entry) in vocab.iter().enumerate() {
            if let &[b] = entry.as_ref()
                && mapping[b as usize].0 == UNSET
            {
                mapping[b as usize] = TokenId(id as u32);
            }
        }
        if let Some(missing) = mapping
            .iter()
            .enumerate()
            .position(|(b, t)| t.0 == UNSET && is_valid_utf8_byte(b as u8))
        {
            return Err(anyhow!(
                "Byte remapping failed: no single-byte vocab entry for byte {missing:#04x}"
            ));
        }
        // Fill the tolerated (never-in-UTF-8) gaps with a safe placeholder so
        // an unexpected malformed byte indexes a valid token instead of OOB.
        for t in mapping.iter_mut() {
            if t.0 == UNSET {
                *t = TokenId(0);
            }
        }
        Ok(mapping
            .iter()
            .enumerate()
            .any(|(b, t)| t.0 != b as u32)
            .then_some(ByteRemapping { mapping }))
    }
}

/// Merge-pair lookup replacing the hashbrown `merges` map on the miss
/// path: a dense grid for pairs with both sides `< 2^DENSE_LOG2` (byte
/// tokens plus the earliest ~1.8k merges, which dominate lookups) over a
/// flat open-addressed table of all merges at ≤ 1/2 load whose `u64` slots
/// pack key and value, so a probe is one multiply, one load, one compare.
/// Values are merged token IDs (`u32::MAX` = no merge). [`Self::build`]
/// returns `None` for vocabularies that violate its packing invariants.
pub(crate) struct PairRankTable {
    /// `dense[(a << DENSE_LOG2) | b]`, `u32::MAX` when the pair does not merge.
    dense: Box<[u32]>,
    /// `((a << 21 | b) << 21) | merged_id`; `u64::MAX` = empty (real slots
    /// fit 63 bits, and the sentinel's key field exceeds every real key).
    slots: Box<[u64]>,
    /// `slots.len() - 1` (length is a power of two).
    mask: usize,
    /// `64 - log2(slots.len())`: the hash keeps the top bits.
    shift: u32,
}

/// Every token ID must fit a 21-bit key lane (covers any vocab < 2M IDs).
const PAIR_ID_BITS: u32 = 21;
/// Dense grid covers pairs with both sides < 2^11 (16 MiB, L3-resident).
const DENSE_LOG2: u32 = 11;

impl PairRankTable {
    /// Build the table, or `None` when this vocabulary cannot use it (IDs
    /// too large for the packed key, or pathological clustering).
    pub(crate) fn build<S: std::hash::BuildHasher>(
        merges: &HashMap<(TokenId, TokenId), TokenId, S>,
        vocab_len: usize,
    ) -> Option<Self> {
        // Every ID that can appear as a merge-loop symbol must be < 2^21;
        // the per-merge check is defensive (`merges` is caller-supplied).
        if vocab_len > 1 << PAIR_ID_BITS {
            return None;
        }
        let id_limit = 1u32 << PAIR_ID_BITS;
        if merges
            .iter()
            .any(|(&(a, b), &m)| a.0 >= id_limit || b.0 >= id_limit || m.0 >= id_limit)
        {
            return None;
        }

        let mut dense = vec![u32::MAX; 1usize << (2 * DENSE_LOG2)].into_boxed_slice();
        // The flat level holds all merges (dense subset included) at ≤ 1/2 load.
        let n_slots = (merges.len().max(1) * 2).next_power_of_two().max(64);
        let shift = 64 - n_slots.trailing_zeros();
        let mask = n_slots - 1;
        let mut slots = vec![u64::MAX; n_slots].into_boxed_slice();
        for (&(a, b), &m) in merges {
            if (a.0 | b.0) >> DENSE_LOG2 == 0 {
                dense[((a.0 as usize) << DENSE_LOG2) | b.0 as usize] = m.0;
            }
            let key = ((a.0 as u64) << PAIR_ID_BITS) | b.0 as u64;
            let mut idx = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize;
            let mut displacement = 0usize;
            while slots[idx] != u64::MAX {
                idx = (idx + 1) & mask;
                displacement += 1;
                // Give up on ugly clustering rather than degrade every miss lookup.
                if displacement > 64 {
                    return None;
                }
            }
            slots[idx] = (key << PAIR_ID_BITS) | m.0 as u64;
        }

        Some(PairRankTable { dense, slots, mask, shift })
    }

    /// Merged token ID of the pair `(a, b)`, or `u32::MAX` when it does not merge.
    #[inline(always)]
    pub(crate) fn rank(&self, a: TokenId, b: TokenId) -> u32 {
        if (a.0 | b.0) >> DENSE_LOG2 == 0 {
            let idx = ((a.0 as usize) << DENSE_LOG2) | b.0 as usize;
            // SAFETY: both IDs < 2^DENSE_LOG2, so idx < dense.len().
            return unsafe { *self.dense.get_unchecked(idx) };
        }
        let key = ((a.0 as u64) << PAIR_ID_BITS) | b.0 as u64;
        let mut idx = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
        loop {
            // SAFETY: idx starts < slots.len() (shift keeps log2(len) bits)
            // and stays masked.
            let slot = unsafe { *self.slots.get_unchecked(idx) };
            if slot >> PAIR_ID_BITS == key {
                return (slot & ((1 << PAIR_ID_BITS) - 1)) as u32;
            }
            if slot == u64::MAX {
                return u32::MAX;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// `prefetcht0` the first line [`Self::rank`] would load for `(a, b)`.
    /// The short merge's refresh lookups sit on a serial chain; both refresh
    /// pairs are known before the list surgery, so their loads can overlap
    /// it. x86_64 only (aarch64 uses the NEON core).
    #[inline(always)]
    pub(crate) fn prefetch_rank(&self, a: TokenId, b: TokenId) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `_mm_prefetch` does not dereference; the index math is
        // `rank`'s, so the address stays inside the live allocation.
        unsafe {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let addr = if (a.0 | b.0) >> DENSE_LOG2 == 0 {
                let idx = ((a.0 as usize) << DENSE_LOG2) | b.0 as usize;
                self.dense.as_ptr().add(idx) as *const i8
            } else {
                let key = ((a.0 as u64) << PAIR_ID_BITS) | b.0 as u64;
                let idx = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
                self.slots.as_ptr().add(idx) as *const i8
            };
            _mm_prefetch(addr, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (a, b);
    }
}

/// Reusable scratch buffers for [`bpe_merge_symbols_with_scratch`], so the
/// miss path performs no per-pretoken allocations.
#[derive(Default)]
pub struct MergeScratch {
    next: Vec<u32>,
    prev: Vec<u32>,
    heap: Vec<std::cmp::Reverse<u64>>,
}

/// Pack a heap entry as `(rank << 32) | position`: rank first, position
/// as tie-break, in half the element size of a tuple.
#[inline(always)]
fn pack_merge_entry(rank: u32, pos: u32) -> u64 {
    ((rank as u64) << 32) | pos as u64
}

/// Apply BPE merges to a symbol sequence; priority is the merged token's
/// ID (lower = first), as for tiktoken-style vocabularies.
pub fn bpe_merge_symbols<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    symbols: &mut Vec<TokenId>,
) {
    bpe_merge_symbols_with_scratch(merges, symbols, &mut MergeScratch::default());
}

/// [`bpe_merge_symbols`] with caller-provided scratch buffers.
pub fn bpe_merge_symbols_with_scratch<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    symbols: &mut Vec<TokenId>,
    scratch: &mut MergeScratch,
) {
    bpe_merge_symbols_by_rank(
        &|a, b| merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
        symbols,
        scratch,
    );
}

/// Id-as-rank merge, dispatching on size: `get_rank` returns the merged
/// token's ID (== merge priority) or `u32::MAX` for no merge. Out of line:
/// it only runs on cache misses.
#[inline(never)]
pub(crate) fn bpe_merge_symbols_by_rank(
    get_rank: &impl Fn(TokenId, TokenId) -> u32,
    symbols: &mut Vec<TokenId>,
    scratch: &mut MergeScratch,
) {
    let n = symbols.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_MERGE_MAX {
        bpe_merge_symbols_small(get_rank, symbols);
        return;
    }
    bpe_merge_symbols_heap(
        &|a, b| {
            let rank = get_rank(a, b);
            (TokenId(rank), rank)
        },
        symbols,
        scratch,
    );
}

/// Min-heap + doubly-linked-list merge for n > SMALL_MERGE_MAX, O(n log n).
/// `get` returns `(merged, rank)`, rank `u32::MAX` = no merge; lowest rank
/// first, then lowest position.
fn bpe_merge_symbols_heap(
    get: &impl Fn(TokenId, TokenId) -> (TokenId, u32),
    symbols: &mut Vec<TokenId>,
    scratch: &mut MergeScratch,
) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = symbols.len();
    // Doubly-linked list via u32 index arrays.
    const NONE: u32 = u32::MAX;
    let next = &mut scratch.next;
    let prev = &mut scratch.prev;
    next.clear();
    next.extend(1..n as u32);
    next.push(NONE);
    prev.clear();
    prev.push(NONE);
    prev.extend(0..n as u32 - 1);

    // Min-heap of packed (rank, position), heapified in O(n).
    let mut seeds = std::mem::take(&mut scratch.heap);
    seeds.clear();
    for i in 0..n - 1 {
        let (_, rank) = get(symbols[i], symbols[i + 1]);
        if rank != u32::MAX {
            seeds.push(Reverse(pack_merge_entry(rank, i as u32)));
        }
    }
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::from(seeds);

    while let Some(Reverse(entry)) = heap.pop() {
        let pos = (entry & u32::MAX as u64) as usize;
        let expected_rank = (entry >> 32) as u32;
        let right = next[pos];
        if right == NONE {
            continue;
        }
        let right = right as usize;
        // Stale entries (invalidated by an earlier merge) no longer match.
        let (merged, rank) = get(symbols[pos], symbols[right]);
        if rank == expected_rank {
            symbols[pos] = merged;
            let right_right = next[right];
            next[pos] = right_right;
            if right_right != NONE {
                prev[right_right as usize] = pos as u32;
            }
            // `next[right] = NONE` invalidates any stale heap entries at
            // `right`; nothing reads `prev[right]` once it is unlinked.
            next[right] = NONE;

            let left = prev[pos];
            if left != NONE {
                let (_, rank) = get(symbols[left as usize], symbols[pos]);
                if rank != u32::MAX {
                    heap.push(Reverse(pack_merge_entry(rank, left)));
                }
            }
            if next[pos] != NONE {
                let (_, rank) = get(symbols[pos], symbols[next[pos] as usize]);
                if rank != u32::MAX {
                    heap.push(Reverse(pack_merge_entry(rank, pos as u32)));
                }
            }
        }
    }
    // The pop loop drained the heap; keep its capacity for the next call.
    scratch.heap = heap.into_vec();

    // Compact survivors in place: their indices are strictly increasing,
    // so writes never overtake reads.
    let mut write = 0;
    let mut i = 0;
    loop {
        symbols[write] = symbols[i];
        write += 1;
        if next[i] == NONE {
            break;
        }
        i = next[i] as usize;
    }
    symbols.truncate(write);
}

/// Sequences up to this length use the linear-scan merge instead of the heap.
const SMALL_MERGE_MAX: usize = 32;

/// BPE merge for n <= SMALL_MERGE_MAX in the style of tiktoken's
/// `byte_pair_merge`: per-position ranks, linear min scan, merge, refresh
/// the two neighbor ranks. Same priority order (lowest ID, then lowest
/// position) as the heap loop. Vec-based core for the 16..=32-symbol
/// long-miss path; see `bpe_merge_symbols_short_scalar` for why the cores
/// stay separate.
fn bpe_merge_symbols_small(
    get_rank: &impl Fn(TokenId, TokenId) -> u32,
    symbols: &mut Vec<TokenId>,
) {
    let n = symbols.len();
    debug_assert!((2..=SMALL_MERGE_MAX).contains(&n));
    // Stack-resident doubly-linked list. Sentinels: next[last] == n,
    // prev[0] == u8::MAX; both fail `< n` checks.
    let mut next = [0u8; SMALL_MERGE_MAX];
    let mut prev = [0u8; SMALL_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // ranks[i] = priority of merging the pair starting at active position i;
    // MAX when there is no merge or the position was merged away.
    let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
    for i in 0..n - 1 {
        ranks[i] = get_rank(symbols[i], symbols[i + 1]);
    }
    loop {
        let mut best = u32::MAX;
        let mut best_i = 0;
        for (i, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_i = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        let i = best_i;
        symbols[i] = TokenId(best);
        // Unlink the right element of the merged pair.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        next[i] = new_right as u8;
        ranks[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            ranks[i] = get_rank(symbols[i], symbols[new_right]);
        } else {
            ranks[i] = u32::MAX;
        }
        let left = prev[i] as usize;
        if left < n {
            ranks[left] = get_rank(symbols[left], symbols[i]);
        }
    }
    // Compact survivors in place: list indices are strictly increasing, so
    // writes never overtake reads.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    symbols.truncate(write);
}

/// Symbol capacity of the stack-array short merges: short pretokens are
/// ≤ 15 bytes, so at most 15 initial symbols.
pub(crate) const SHORT_MERGE_MAX: usize = 16;

/// [`bpe_merge_symbols_small`] over a caller-owned stack array — the
/// short-pretoken (<= 15 symbols) miss path's merge. Returns the merged
/// length. The three small cores (this, `_short_neon`, and the Vec-based
/// `_small`) are deliberately separate: unifying them measured as a
/// regression. `prefetch_rank` is a no-op closure on the HashMap fallback.
pub(crate) fn bpe_merge_symbols_short_scalar(
    get_rank: impl Fn(TokenId, TokenId) -> u32,
    prefetch_rank: impl Fn(TokenId, TokenId),
    symbols: &mut [TokenId; SHORT_MERGE_MAX],
    n: usize,
) -> usize {
    debug_assert!((2..=SHORT_MERGE_MAX - 1).contains(&n));
    // Stack-resident doubly-linked list; see `bpe_merge_symbols_small`.
    let mut next = [0u8; SHORT_MERGE_MAX];
    let mut prev = [0u8; SHORT_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // ranks[i] = priority of merging the pair starting at active position i;
    // MAX when there is no merge or the position was merged away.
    let mut ranks = [u32::MAX; SHORT_MERGE_MAX];
    for i in 0..n - 1 {
        ranks[i] = get_rank(symbols[i], symbols[i + 1]);
    }
    loop {
        let mut best = u32::MAX;
        let mut best_i = 0;
        for (i, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_i = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        let i = best_i;
        // Both refresh pairs are known now, so request their rank lines
        // before the list surgery. Reading `left` early is safe: new_right
        // > i, so the `prev[new_right]` store cannot alias `prev[i]`.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        let left = prev[i] as usize;
        if new_right < n {
            prefetch_rank(TokenId(best), symbols[new_right]);
        }
        if left < n {
            prefetch_rank(symbols[left], TokenId(best));
        }
        symbols[i] = TokenId(best);
        // Unlink the right element of the merged pair.
        next[i] = new_right as u8;
        ranks[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            ranks[i] = get_rank(symbols[i], symbols[new_right]);
        } else {
            ranks[i] = u32::MAX;
        }
        if left < n {
            ranks[left] = get_rank(symbols[left], symbols[i]);
        }
    }
    // Compact survivors in place: list indices are strictly increasing, so
    // writes never overtake reads.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    write
}

/// [`bpe_merge_symbols_short_scalar`] with a branchless NEON min-rank scan.
/// Rank and position share one lane, `pr[i] = (rank << 8) | i`, so the
/// vector minimum picks the lowest rank, then the lowest position — the
/// scalar scan's order. Ranks are < 2^21 (a [`PairRankTable`] invariant,
/// hence the table parameter), so packed lanes are < 2^29 and "no merge"
/// (`u32::MAX << 8`) sorts above every real lane. Inactive lanes stay at
/// `u32::MAX`; live lane indices are `< n`, so `n <= 8` reads two vectors.
#[cfg(target_arch = "aarch64")]
pub(crate) fn bpe_merge_symbols_short_neon(
    table: &PairRankTable,
    symbols: &mut [TokenId; SHORT_MERGE_MAX],
    n: usize,
) -> usize {
    use core::arch::aarch64::{vld1q_u32, vminq_u32, vminvq_u32};
    debug_assert!((2..=SHORT_MERGE_MAX - 1).contains(&n));
    /// Every packed value at or above this has rank u32::MAX (no merge).
    const NO_MERGE_FLOOR: u32 = u32::MAX << 8;
    let pack = |rank: u32, i: usize| (rank << 8) | i as u32;
    // Stack-resident doubly-linked list; see `bpe_merge_symbols_small`.
    let mut next = [0u8; SHORT_MERGE_MAX];
    let mut prev = [0u8; SHORT_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // pr[i] = packed (rank, position) of the pair starting at active
    // position i; the only array the scan reads.
    let mut pr = [u32::MAX; SHORT_MERGE_MAX];
    for i in 0..n - 1 {
        pr[i] = pack(table.rank(symbols[i], symbols[i + 1]), i);
    }
    let narrow = n <= 8;
    loop {
        // SAFETY: pr is 16 contiguous u32s; vld1q_u32 has no alignment
        // requirement beyond u32's.
        let best = unsafe {
            let p = pr.as_ptr();
            let m01 = vminq_u32(vld1q_u32(p), vld1q_u32(p.add(4)));
            let m = if narrow {
                m01
            } else {
                let m23 = vminq_u32(vld1q_u32(p.add(8)), vld1q_u32(p.add(12)));
                vminq_u32(m01, m23)
            };
            vminvq_u32(m)
        };
        if best >= NO_MERGE_FLOOR {
            break;
        }
        let i = (best & 0xFF) as usize;
        symbols[i] = TokenId(best >> 8);
        // Unlink the right element of the merged pair.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        next[i] = new_right as u8;
        pr[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            pr[i] = pack(table.rank(symbols[i], symbols[new_right]), i);
        } else {
            pr[i] = u32::MAX;
        }
        let left = prev[i] as usize;
        if left < n {
            pr[left] = pack(table.rank(symbols[left], symbols[i]), left);
        }
    }
    // Compact survivors in place.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    write
}

/// Vocabulary entries as `(id, bytes)` pairs in ID order, skipping IDs with
/// no assigned content. Shared by both tokenizer types' `vocab_entries`.
pub(crate) fn vocab_entries(
    vocab: &[std::sync::Arc<[u8]>],
) -> impl Iterator<Item = (u32, &[u8])> {
    vocab
        .iter()
        .enumerate()
        .filter(|(_, bytes)| !bytes.is_empty())
        .map(|(id, bytes)| (id as u32, bytes.as_ref()))
}

/// Pack a ranked-merge pair key into one `u64`: a single-multiply hash
/// instead of a two-round tuple hash, probed ~2x per symbol.
#[inline(always)]
pub fn ranked_merge_key(a: TokenId, b: TokenId) -> u64 {
    ((a.0 as u64) << 32) | b.0 as u64
}

/// Explicit merge-priority table: `ranked_merge_key(a, b)` → `(merged, rank)`.
pub type RankedMerges = HashMap<u64, (TokenId, u32), rustc_hash::FxBuildHasher>;

/// Merge rules of a ranked table as `(left, right)` byte pairs in rank order.
pub(crate) fn ranked_merge_entries<'a>(
    rm: &RankedMerges,
    vocab: &'a [std::sync::Arc<[u8]>],
) -> Vec<(&'a [u8], &'a [u8])> {
    let mut ranked: Vec<(u64, u32)> = rm.iter().map(|(&key, &(_, rank))| (key, rank)).collect();
    ranked.sort_unstable_by_key(|&(_, rank)| rank);
    ranked
        .into_iter()
        .map(|(key, _)| {
            (
                vocab[(key >> 32) as usize].as_ref(),
                vocab[key as u32 as usize].as_ref(),
            )
        })
        .collect()
}

/// Ranked-merge variant of [`bpe_merge_symbols_small`]: allocation-free BPE
/// for short symbol sequences with priority from the table's explicit rank.
/// `(merged, rank)` for a pair, or rank `u32::MAX` for no merge.
#[inline(always)]
fn ranked_get<S: std::hash::BuildHasher>(
    merges: &HashMap<u64, (TokenId, u32), S>,
    a: TokenId,
    b: TokenId,
) -> (TokenId, u32) {
    merges.get(&ranked_merge_key(a, b)).map_or((TokenId(0), u32::MAX), |&m| m)
}

/// Merges `symbols` in place and returns the surviving count.
pub(crate) fn bpe_merge_symbols_ranked_slice<S: std::hash::BuildHasher>(
    merges: &HashMap<u64, (TokenId, u32), S>,
    symbols: &mut [TokenId],
) -> usize {
    let get = |a, b| ranked_get(merges, a, b);
    let n = symbols.len();
    debug_assert!((2..=SMALL_MERGE_MAX).contains(&n));
    // Stack-resident doubly-linked list; see `bpe_merge_symbols_small`.
    let mut next = [0u8; SMALL_MERGE_MAX];
    let mut prev = [0u8; SMALL_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // For the pair starting at active position i: its merge priority and
    // merged token. Rank u32::MAX = no merge (or merged away).
    let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
    let mut merged = [TokenId(0); SMALL_MERGE_MAX];
    for i in 0..n - 1 {
        (merged[i], ranks[i]) = get(symbols[i], symbols[i + 1]);
    }
    loop {
        let mut best = u32::MAX;
        let mut best_i = 0;
        for (i, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_i = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        let i = best_i;
        symbols[i] = merged[i];
        // Unlink the right element of the merged pair.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        next[i] = new_right as u8;
        ranks[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            (merged[i], ranks[i]) = get(symbols[i], symbols[new_right]);
        } else {
            ranks[i] = u32::MAX;
        }
        let left = prev[i] as usize;
        if left < n {
            (merged[left], ranks[left]) = get(symbols[left], symbols[i]);
        }
    }
    // Compact survivors in place.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    write
}

/// Apply BPE merges with explicit ranks (`(a, b) → (merged, rank)`, lower
/// rank first), as SentencePiece-style tokenizers need.
pub fn bpe_merge_symbols_ranked<S: std::hash::BuildHasher>(
    merges: &HashMap<u64, (TokenId, u32), S>,
    symbols: &mut Vec<TokenId>,
) {
    let n = symbols.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_MERGE_MAX {
        let new_len = bpe_merge_symbols_ranked_slice(merges, symbols);
        symbols.truncate(new_len);
        return;
    }
    bpe_merge_symbols_heap(&|a, b| ranked_get(merges, a, b), symbols, &mut MergeScratch::default());
}

pub use sentencepiece::SentencePieceBPE;
pub use tiktoken::Tokenizer;

#[cfg(test)]
pub(crate) mod test_util {
    /// xorshift64: deterministic, dependency-free RNG for test inputs.
    pub(crate) struct XorShift64(pub u64);

    impl XorShift64 {
        pub(crate) fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        pub(crate) fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::XorShift64;
    use super::*;

    fn random_merges(
        rng: &mut XorShift64,
        n_merges: usize,
        id_range: u32,
    ) -> HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher> {
        let mut merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        for i in 0..n_merges {
            let a = TokenId(rng.below(id_range as u64) as u32);
            let b = TokenId(rng.below(id_range as u64) as u32);
            merges.entry((a, b)).or_insert(TokenId(256 + i as u32));
        }
        merges
    }

    /// The flat table must agree with the hashbrown map on every merge pair
    /// and return the no-merge sentinel everywhere else.
    #[test]
    fn pair_rank_table_matches_map() {
        let mut rng = XorShift64(0x1234_5678_9ABC_DEF0);
        let id_range = 4096u32;
        let merges = random_merges(&mut rng, 3000, id_range);
        let table = PairRankTable::build(&merges, id_range as usize).expect("build");
        for (&(a, b), &m) in &merges {
            assert_eq!(table.rank(a, b), m.0, "pair ({}, {})", a.0, b.0);
        }
        for _ in 0..100_000 {
            let a = TokenId(rng.below(id_range as u64) as u32);
            let b = TokenId(rng.below(id_range as u64) as u32);
            let expected = merges.get(&(a, b)).map_or(u32::MAX, |m| m.0);
            assert_eq!(table.rank(a, b), expected, "pair ({}, {})", a.0, b.0);
        }
        // Oversized IDs must refuse the table, not corrupt it.
        assert!(PairRankTable::build(&merges, (1 << PAIR_ID_BITS) + 1).is_none());
        let mut big = merges.clone();
        big.insert((TokenId(1 << PAIR_ID_BITS), TokenId(0)), TokenId(300));
        assert!(PairRankTable::build(&big, id_range as usize).is_none());
    }

    /// The stack-array short merges (scalar and NEON) must produce exactly
    /// the sequence of the Vec-based merge loop — same merges, same order,
    /// same tie-breaks — across random symbol sequences and merge tables.
    #[test]
    fn short_merges_match_vec_merge_loop() {
        let mut rng = XorShift64(0xDEAD_BEEF_0BAD_F00D);
        for trial in 0..2000 {
            let id_range = 300 + (trial % 7) as u32 * 500;
            let merges = random_merges(&mut rng, 200 + trial % 800, id_range);
            let table = PairRankTable::build(&merges, id_range as usize + 1024).expect("build");
            let n = 2 + rng.below(14) as usize; // 2..=15
            let init: Vec<TokenId> = (0..n)
                .map(|_| TokenId(rng.below(id_range as u64) as u32))
                .collect();

            let mut reference = init.clone();
            bpe_merge_symbols_with_scratch(&merges, &mut reference, &mut MergeScratch::default());

            let mut scalar = [TokenId(0); SHORT_MERGE_MAX];
            scalar[..n].copy_from_slice(&init);
            let len = bpe_merge_symbols_short_scalar(
                |a, b| table.rank(a, b),
                |a, b| table.prefetch_rank(a, b),
                &mut scalar,
                n,
            );
            assert_eq!(&scalar[..len], &reference[..], "scalar diverged: {init:?}");

            #[cfg(target_arch = "aarch64")]
            {
                let mut neon = [TokenId(0); SHORT_MERGE_MAX];
                neon[..n].copy_from_slice(&init);
                let len = bpe_merge_symbols_short_neon(&table, &mut neon, n);
                assert_eq!(&neon[..len], &reference[..], "neon diverged: {init:?}");
            }
        }
    }

    /// The ranked cores (small slice + heap) must agree with the id-as-rank
    /// cores on a table whose rank is the merged ID, for n in both regimes.
    #[test]
    fn ranked_and_id_as_rank_cores_agree() {
        let mut rng = XorShift64(0x5EED_1234_ABCD_0001);
        let mut scratch = MergeScratch::default();
        for trial in 0..1000 {
            let id_range = 300 + (trial % 5) as u32 * 400;
            let merges = random_merges(&mut rng, 200 + trial % 600, id_range);
            let ranked: RankedMerges = merges
                .iter()
                .map(|(&(a, b), &m)| (ranked_merge_key(a, b), (m, m.0)))
                .collect();
            let n = 2 + rng.below(63) as usize; // 2..=64
            let init: Vec<TokenId> = (0..n)
                .map(|_| TokenId(rng.below(id_range as u64) as u32))
                .collect();
            let mut reference = init.clone();
            bpe_merge_symbols_with_scratch(&merges, &mut reference, &mut scratch);
            let mut got = init.clone();
            bpe_merge_symbols_ranked(&ranked, &mut got);
            assert_eq!(got, reference, "n={n}: {init:?}");
        }
    }
}
