use crate::bpe::pretoken_cache::ShortPretokenCache;
#[cfg(target_arch = "aarch64")]
use crate::bpe::bpe_merge_symbols_short_neon;
use crate::bpe::RankedMerges;
use crate::bpe::{
    ByteRemapping, MergeScratch, PairRankTable, SHORT_MERGE_MAX, bpe_merge_symbols,
    bpe_merge_symbols_by_rank, bpe_merge_symbols_ranked, bpe_merge_symbols_ranked_slice,
    bpe_merge_symbols_short_scalar, bpe_merge_symbols_with_scratch,
};
use crate::pretokenize::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer, PRETOKEN_CHUNK,
    Pretoken, PretokenSpans, PretokenizerType, SpanBatch, pack_pretoken_key, pretoken_key_hash,
};
use crate::token::TokenId;
use eyre::Result;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

pub type MergeMap = HashMap<(TokenId, TokenId), TokenId, FxBuildHasher>;
pub type VocabInv = HashMap<Arc<[u8]>, TokenId, FxBuildHasher>;

/// Immutable model tables, shared across forks behind `Arc` (the rare
/// `add_special_token` mutation goes through `Arc::make_mut`).
#[derive(Clone)]
pub(crate) struct Model {
    pub(crate) merges: Arc<MergeMap>,
    /// Flat pair-rank table for the miss path; `None` when IDs do not fit
    /// its packed keys.
    pair_ranks: Option<Arc<PairRankTable>>,
    /// Explicit merge priorities for rank-mapped vocabularies (RoBERTa/OPT/
    /// DeBERTa); when set, `merges` is empty and `pair_ranks` is `None`.
    ranked_merges: Option<Arc<RankedMerges>>,
    pub(crate) vocab: Arc<Vec<Arc<[u8]>>>,
    pub(crate) vocab_inv: Arc<VocabInv>,
    pub(crate) byte_remapping: Option<ByteRemapping>,
}

/// Byte-level BPE tokenizer (tiktoken / GPT-2 style): initial symbols are
/// bytes; merge priority is the merged token's vocab ID unless
/// `ranked_merges` is set.
pub struct Tokenizer {
    pub(crate) model: Model,
    /// Append-only arena for cached encodings of 5+ tokens (shorter ones
    /// live inline in the cache entry).
    token_arena: Vec<TokenId>,
    /// Cache for pretokens of ≤ 15 bytes (see `pretoken_cache.rs`).
    pretoken_cache: ShortPretokenCache,
    /// Cache for longer pretokens.
    pretoken_cache_long: HashMap<Box<[u8]>, (u32, u32), FxBuildHasher>,
    merge_scratch: MergeScratch,
    symbol_scratch: Vec<TokenId>,
    pub(crate) pretokenizer_type: PretokenizerType,
    /// Added tokens, matched atomically before pretokenization like HF's
    /// AddedVocabulary.
    added_tokens: Vec<AddedTokenDef>,
    /// Leftmost-longest automaton over `added_tokens` (pattern index ==
    /// vec index).
    added_matcher: Option<aho_corasick::AhoCorasick>,
    /// NFC-normalize segments before pretokenization (HF `NFC` normalizer).
    normalize_nfc: bool,
    /// HF `ByteLevel(add_prefix_space=true)`: a segment not starting with a
    /// space gets one.
    add_prefix_space: bool,
    /// HF BPE `ignore_merges`: a pretoken that is a whole vocab entry
    /// encodes as that single ID.
    ignore_merges: bool,
    /// Cache memory budget; `None` = unbounded.
    cache_budget: Option<CacheBudget>,
}

/// Budget split of [`Tokenizer::set_max_cache_bytes`] plus per-generation
/// counters.
#[derive(Clone)]
struct CacheBudget {
    total_bytes: usize,
    /// Short-table slot ceiling (power of two): at it, reaching the 3/4
    /// growth threshold wipes instead of doubling.
    short_slots: usize,
    /// Token-arena sub-budget in entries.
    arena_entries: usize,
    /// Long-map byte budget (key bytes + `LONG_ENTRY_BYTES` each).
    long_bytes: usize,
    long_bytes_used: usize,
    /// Largest single encoding appended to the arena, allowed as slack past
    /// `arena_entries` so one recurring giant pretoken cannot force a wipe
    /// per occurrence. Kept across wipes.
    max_encoding: usize,
    /// Wipes since the budget was set (or the fork was created).
    generations: u64,
}

impl CacheBudget {
    /// Estimated non-key bytes per long-map entry.
    const LONG_ENTRY_BYTES: usize = 48;

    /// Split `total_bytes` between the three caches given the seed
    /// footprint. Allocates nothing.
    fn derive(total_bytes: usize, n_seed: usize, seed_arena_len: usize) -> Self {
        // Ceiling: largest power of two with slots * 32 <= 70% of the
        // budget (floor 2^16); the seed can push it higher.
        let target = ((total_bytes / 32).saturating_mul(7) / 10).max(1 << 16);
        let mut short_slots = ShortPretokenCache::required_capacity(n_seed, 1 << target.ilog2());
        // Keep the seed at <= 5/8 of the ceiling so each generation admits
        // at least capacity/8 entries before the wipe threshold.
        while n_seed * 8 > short_slots * 5 {
            short_slots *= 2;
        }
        let rem = total_bytes.saturating_sub(short_slots * 32);
        // Even arena/long split, floored so degenerate budgets stay
        // functional.
        CacheBudget {
            total_bytes,
            short_slots,
            arena_entries: ((rem / 2) / 4).max(seed_arena_len * 2 + 4096),
            long_bytes: (rem / 2).max(64 << 10),
            long_bytes_used: 0,
            max_encoding: 0,
            generations: 0,
        }
    }
}

/// NFC-normalize a segment if needed, using `buf` as scratch. ASCII,
/// already-normalized, and invalid-UTF-8 segments pass through.
fn nfc_segment<'a>(seg: &'a [u8], buf: &'a mut String) -> &'a [u8] {
    if seg.is_ascii() {
        return seg;
    }
    let Ok(s) = std::str::from_utf8(seg) else {
        return seg;
    };
    let nfc = icu::normalizer::ComposingNormalizer::new_nfc();
    if nfc.is_normalized(s) {
        return seg;
    }
    buf.clear();
    nfc.normalize_to(s, buf)
        .expect("writing to a String cannot fail");
    buf.as_bytes()
}

/// Cache-value packing (shared by the short-pretoken table and decode in
/// the encode loop). `val` low byte: token count in bits 0-6 plus a
/// "spilled" flag in bit 7. Inline values (1-4 tokens; only the first ID
/// must fit 24 bits — true of every real vocab) carry tokens 1-2 in `val`
/// bits 8-31 and 32-63 and tokens 3-4 in `ext`'s two u32 lanes; spilled
/// values carry the token-arena offset in `val`'s high 32 bits and leave
/// `ext` unused.
const VAL_SPILL: u64 = 0x80;

#[inline(always)]
fn pack_val_inline(symbols: &[TokenId]) -> Option<(u64, u64)> {
    match *symbols {
        [a] if a.0 < (1 << 24) => Some((1 | ((a.0 as u64) << 8), 0)),
        [a, b] if a.0 < (1 << 24) => {
            Some((2 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32), 0))
        }
        [a, b, c] if a.0 < (1 << 24) => Some((
            3 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32),
            c.0 as u64,
        )),
        [a, b, c, d] if a.0 < (1 << 24) => Some((
            4 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32),
            c.0 as u64 | ((d.0 as u64) << 32),
        )),
        _ => None,
    }
}

/// View a `TokenId` slice as its underlying `u32`s (repr(transparent)),
/// so bulk emits are `extend_from_slice` memcpys.
#[inline(always)]
fn token_ids_as_u32s(toks: &[TokenId]) -> &[u32] {
    // SAFETY: TokenId is #[repr(transparent)] over u32.
    unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u32, toks.len()) }
}

/// Unpack an inline value's four token lanes (lanes past the count are
/// another key's leftovers; callers truncate by the count).
#[inline(always)]
fn unpack_val_lanes(val: u64, ext: u64) -> [u32; 4] {
    [
        (val >> 8) as u32 & 0xFF_FFFF,
        (val >> 32) as u32,
        ext as u32,
        (ext >> 32) as u32,
    ]
}

/// One piece of the added-token walk ([`Tokenizer::for_each_piece`]): a
/// text segment with its source byte offset, or an added token's ID.
enum Piece<'a> {
    Segment(&'a [u8], usize),
    Added(TokenId),
}

/// One added token as configured by the loader: byte content, emitted ID,
/// and HF `AddedToken` whitespace-stripping flags (`lstrip` absorbs
/// whitespace before a match, `rstrip` after it).
#[derive(Clone, Debug)]
pub struct AddedTokenDef {
    pub content: Arc<[u8]>,
    pub id: TokenId,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// Byte offset after the leading Unicode whitespace of `bytes` (the set of
/// `str::trim_start`, which is what HF's `\s*` sees). Invalid UTF-8 stops
/// the scan.
fn trim_ws_start(bytes: &[u8]) -> usize {
    let mut pos = 0;
    while pos < bytes.len() {
        let width = match bytes[pos] {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => break,
        };
        let Some(chunk) = bytes.get(pos..pos + width) else {
            break;
        };
        match std::str::from_utf8(chunk) {
            Ok(s) if s.chars().next().is_some_and(char::is_whitespace) => pos += width,
            _ => break,
        }
    }
    pos
}

/// Length of `bytes` after trimming trailing Unicode whitespace (the set of
/// `str::trim_end`). Invalid UTF-8 stops the scan.
fn trim_ws_end(bytes: &[u8]) -> usize {
    let mut end = bytes.len();
    while end > 0 {
        // Back up over at most 3 continuation bytes to the character start.
        let mut start = end - 1;
        while start > 0 && (bytes[start] & 0xC0) == 0x80 && end - start < 4 {
            start -= 1;
        }
        match std::str::from_utf8(&bytes[start..end]) {
            Ok(s) if s.chars().next().is_some_and(char::is_whitespace) => end = start,
            _ => break,
        }
    }
    end
}

/// Overwrite the short-cache entry of every 1..=15-byte added-token content
/// that resolves in `vocab_inv` with that single ID. The single body every
/// reseed applies after the vocab seed, so parent and forks agree.
fn apply_added_token_overwrites(
    added_tokens: &[AddedTokenDef],
    vocab_inv: &VocabInv,
    pretoken_cache: &mut ShortPretokenCache,
    token_arena: &mut Vec<TokenId>,
) {
    for tok in added_tokens {
        let content = &tok.content;
        if !(1..=15).contains(&content.len()) {
            continue;
        }
        let Some(&id) = vocab_inv.get(content) else {
            continue;
        };
        let key = pack_pretoken_key(content).expect("length checked <= 15");
        let h = pretoken_key_hash(key);
        let (val, ext) = Tokenizer::pack_val(&[id], token_arena);
        pretoken_cache.replace(key, h, val, ext);
    }
}

/// The vocab entries the short table seeds (1..=15 bytes).
fn short_vocab(vocab: &[Arc<[u8]>]) -> impl Iterator<Item = &[u8]> {
    vocab
        .iter()
        .map(|b| b.as_ref())
        .filter(|b| (1..=15).contains(&b.len()))
}

/// Initial symbols: each byte's single-byte token ID.
#[inline]
fn remap_bytes(br: Option<&ByteRemapping>, bytes: &[u8], out: &mut [TokenId]) {
    match br {
        Some(br) => {
            for (dst, &b) in out.iter_mut().zip(bytes) {
                *dst = br.mapping[b as usize];
            }
        }
        None => {
            for (dst, &b) in out.iter_mut().zip(bytes) {
                *dst = TokenId(b as u32);
            }
        }
    }
}

impl Model {
    /// Encoding of one short pretoken (1..=15 bytes) into `buf`, returning
    /// its token count. Shared by the vocab seed and the miss path, so a
    /// seeded value is exactly what a cold miss would compute.
    #[inline]
    fn seed_encode(
        &self,
        ignore_merges: bool,
        bytes: &[u8],
        buf: &mut [TokenId; SHORT_MERGE_MAX],
    ) -> usize {
        if ignore_merges && let Some(&id) = self.vocab_inv.get(bytes) {
            buf[0] = id;
            return 1;
        }
        let n = bytes.len();
        debug_assert!((1..SHORT_MERGE_MAX).contains(&n));
        remap_bytes(self.byte_remapping.as_ref(), bytes, &mut buf[..n]);
        if n < 2 {
            return n;
        }
        match (self.ranked_merges.as_deref(), self.pair_ranks.as_deref()) {
            (Some(rm), _) => bpe_merge_symbols_ranked_slice(rm, &mut buf[..n]),
            #[cfg(target_arch = "aarch64")]
            (None, Some(table)) => bpe_merge_symbols_short_neon(table, buf, n),
            // x86 stays scalar: AVX2/AVX-512 min-scans measured ~1% slower
            // on Zen 5, see profiling/x86_port_plan.md §6.
            #[cfg(not(target_arch = "aarch64"))]
            (None, Some(table)) => bpe_merge_symbols_short_scalar(
                |a, b| table.rank(a, b),
                |a, b| table.prefetch_rank(a, b),
                buf,
                n,
            ),
            (None, None) => bpe_merge_symbols_short_scalar(
                |a, b| self.merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                |_, _| {},
                buf,
                n,
            ),
        }
    }
}

/// Establish seed-level state in `cache`: every short vocab entry's seed
/// encoding, then the added-token overwrites. Without `ignore_merges` the
/// seed is the MERGE RESULT, not the entry's own ID (merge-unreachable
/// vocab entries exist, e.g. qwen3_5, and HF returns the decomposition);
/// with it, the own ID. Duplicate byte strings seed the same value.
fn seed_into(
    model: &Model,
    ignore_merges: bool,
    added_tokens: &[AddedTokenDef],
    cache: &mut ShortPretokenCache,
    arena: &mut Vec<TokenId>,
) {
    let mut buf = [TokenId(0); SHORT_MERGE_MAX];
    for bytes in short_vocab(&model.vocab) {
        let key = pack_pretoken_key(bytes).expect("length checked <= 15");
        let h = pretoken_key_hash(key);
        let n = model.seed_encode(ignore_merges, bytes, &mut buf);
        let (val, ext) = Tokenizer::pack_val(&buf[..n], arena);
        cache.replace(key, h, val, ext);
    }
    apply_added_token_overwrites(added_tokens, &model.vocab_inv, cache, arena);
}

/// A fresh short table at seed level, sized for the seed and at least
/// `min_slots` (so seeding never grows it).
fn seeded_pretoken_cache(
    model: &Model,
    ignore_merges: bool,
    added_tokens: &[AddedTokenDef],
    arena: &mut Vec<TokenId>,
    min_slots: usize,
) -> ShortPretokenCache {
    let mut cache =
        ShortPretokenCache::with_at_least(short_vocab(&model.vocab).count(), min_slots);
    seed_into(model, ignore_merges, added_tokens, &mut cache, arena);
    cache
}

impl Tokenizer {
    /// Default cache budget: 512 MiB per encode worker.
    /// `set_max_cache_bytes(None)` restores unbounded growth.
    pub const DEFAULT_MAX_CACHE_BYTES: usize = 512 << 20;

    pub fn new(merges: MergeMap, vocab: Vec<Vec<u8>>, byte_remapping: Option<ByteRemapping>) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect();
        Self::from_tables(merges, None, vocab, byte_remapping)
    }

    /// Construct from an explicit-rank merge table (`ranked_merge_key(a, b)`
    /// → `(merged, rank)`), for vocabularies whose IDs do not follow merge
    /// order.
    pub fn new_ranked(
        ranked_merges: RankedMerges,
        vocab: Vec<Vec<u8>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect();
        Self::from_tables(HashMap::default(), Some(ranked_merges), vocab, byte_remapping)
    }

    /// Shared construction tail. Pipeline settings start as placeholders
    /// (GPT-2 pretokenization, no added tokens) every loader must overwrite.
    fn from_tables(
        merges: MergeMap,
        ranked_merges: Option<RankedMerges>,
        vocab: Vec<Arc<[u8]>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab_inv: VocabInv = vocab
            .iter()
            .cloned()
            .zip((0u32..).map(TokenId::from))
            .collect();
        let pair_ranks = if ranked_merges.is_none() {
            PairRankTable::build(&merges, vocab.len()).map(Arc::new)
        } else {
            None
        };
        let model = Model {
            merges: Arc::new(merges),
            pair_ranks,
            ranked_merges: ranked_merges.map(Arc::new),
            vocab: Arc::new(vocab),
            vocab_inv: Arc::new(vocab_inv),
            byte_remapping,
        };
        let mut token_arena = Vec::new();
        let pretoken_cache = seeded_pretoken_cache(&model, false, &[], &mut token_arena, 0);
        // The default budget's split derives from the just-seeded state, so
        // this is the only table build.
        let cache_budget = Some(CacheBudget::derive(
            Self::DEFAULT_MAX_CACHE_BYTES,
            short_vocab(&model.vocab).count(),
            token_arena.len(),
        ));
        Tokenizer {
            model,
            token_arena,
            pretoken_cache,
            pretoken_cache_long: HashMap::default(),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_matcher: None,
            normalize_nfc: false,
            add_prefix_space: false,
            ignore_merges: false,
            cache_budget,
        }
    }

    /// Pack a cache value: inline when possible, else spilled to the arena.
    #[inline(always)]
    fn pack_val(symbols: &[TokenId], token_arena: &mut Vec<TokenId>) -> (u64, u64) {
        pack_val_inline(symbols).unwrap_or_else(|| {
            let offset = token_arena.len() as u64;
            token_arena.extend_from_slice(symbols);
            (VAL_SPILL | symbols.len() as u64 | (offset << 32), 0)
        })
    }

    /// Reconstruct the merge rules from a vocabulary listed in merge order
    /// (tiktoken files): each entry is one merge of two earlier entries.
    pub fn from_ranks(vocab: Vec<Vec<u8>>) -> Result<Self> {
        let mut merges = MergeMap::default();
        let vocab: Vec<Arc<[u8]>> = vocab.into_iter().map(Into::into).collect();
        let vocab_inv: VocabInv = vocab
            .iter()
            .cloned()
            .zip((0u32..).map(TokenId::from))
            .collect();
        for (id, bytes) in vocab.iter().enumerate() {
            if bytes.len() < 2 {
                continue;
            }
            let mut symbols: Vec<TokenId> = bytes
                .iter()
                .map(|b| vocab_inv[std::slice::from_ref(b)])
                .collect();
            bpe_merge_symbols(&merges, &mut symbols);
            assert_eq!(symbols.len(), 2, "vocab entry {id} is not one merge of earlier entries");
            merges.insert((symbols[0], symbols[1]), TokenId::from(id));
        }
        let byte_remapping = ByteRemapping::from_byte_vocab(&vocab)?;
        Ok(Self::from_tables(merges, None, vocab, byte_remapping))
    }

    /// A tokenizer sharing this one's model with a freshly seeded cache, for
    /// per-thread encoding. Loader-phase mutators (`set_*`,
    /// `add_special_token*`) must run before forking: existing forks keep
    /// the old state.
    pub fn fork(&self) -> Self {
        self.fork_sized(0)
    }

    /// [`Self::fork`] with the caches pre-sized for a worker expected to
    /// encode roughly `expected_bytes` (capacity hints; the caches still
    /// grow past them).
    pub(crate) fn fork_sized(&self, expected_bytes: usize) -> Self {
        // Heaps' law on OWT-like text: distinct short pretokens ≈ 3.45·n^0.62.
        // Size for that at 3/4 load with 1.4x headroom, clamped to 2^16..2^22 slots.
        let distinct = 3.45 * (expected_bytes as f64).powf(0.62);
        let mut cache_slots = ((distinct * (4.0 / 3.0) * 1.4) as usize)
            .clamp(1 << 16, 1 << 22)
            .next_power_of_two();
        let mut arena_cap = (expected_bytes / 256).min(1 << 24);
        let mut long_cap = (expected_bytes / 8192).min(1 << 20);
        if let Some(b) = &self.cache_budget {
            // Each worker gets the FULL budget; the estimates are clamped under it.
            cache_slots = cache_slots.min(b.short_slots);
            arena_cap = arena_cap.min(b.arena_entries);
            long_cap = long_cap.min(b.long_bytes / CacheBudget::LONG_ENTRY_BYTES);
        }
        let mut token_arena = Vec::with_capacity(arena_cap);
        let pretoken_cache = seeded_pretoken_cache(
            &self.model,
            self.ignore_merges,
            &self.added_tokens,
            &mut token_arena,
            cache_slots,
        );
        Tokenizer {
            model: self.model.clone(),
            token_arena,
            pretoken_cache,
            pretoken_cache_long: HashMap::with_capacity_and_hasher(long_cap, FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: self.pretokenizer_type,
            added_tokens: self.added_tokens.clone(),
            added_matcher: self.added_matcher.clone(),
            normalize_nfc: self.normalize_nfc,
            add_prefix_space: self.add_prefix_space,
            ignore_merges: self.ignore_merges,
            cache_budget: self.cache_budget.as_ref().map(|b| CacheBudget {
                long_bytes_used: 0,
                max_encoding: 0,
                generations: 0,
                ..b.clone()
            }),
        }
    }

    pub fn set_pretokenizer_type(&mut self, pretokenizer_type: PretokenizerType) {
        self.pretokenizer_type = pretokenizer_type;
    }

    pub fn pretokenizer_type(&self) -> PretokenizerType {
        self.pretokenizer_type
    }

    /// Enable NFC normalization of non-added-token segments (HF
    /// `normalizer: {"type": "NFC"}`).
    pub fn set_normalize_nfc(&mut self, normalize_nfc: bool) {
        self.normalize_nfc = normalize_nfc;
    }

    /// Enable HF `ByteLevel(add_prefix_space=true)` semantics.
    pub fn set_add_prefix_space(&mut self, add_prefix_space: bool) {
        self.add_prefix_space = add_prefix_space;
    }

    /// Enable HF BPE `ignore_merges`: a pretoken that is a whole vocab entry
    /// encodes as that single ID. Re-seeds the short cache for the new flag.
    pub fn set_ignore_merges(&mut self, ignore_merges: bool) {
        if self.ignore_merges != ignore_merges {
            self.ignore_merges = ignore_merges;
            self.reseed_or_rederive();
        }
    }

    /// Bound the total memory of the encode caches (short table, long map,
    /// token arena), or lift the bound with `None`; see `CacheBudget` for
    /// the split. Crossing a bound on the miss path wipes all three back to
    /// seed level; cache contents never affect output.
    pub fn set_max_cache_bytes(&mut self, budget: Option<usize>) {
        let Some(total_bytes) = budget else {
            self.cache_budget = None;
            return;
        };
        let n_seed = short_vocab(&self.model.vocab).count() + self.added_tokens.len();
        self.pretoken_cache
            .reset_to_capacity(ShortPretokenCache::required_capacity(n_seed, 0));
        self.token_arena = Vec::new();
        self.pretoken_cache_long = HashMap::default();
        self.reseed_cache();
        self.cache_budget =
            Some(CacheBudget::derive(total_bytes, n_seed, self.token_arena.len()));
    }

    /// The configured cache budget in bytes; `None` when unbounded.
    pub fn max_cache_bytes(&self) -> Option<usize> {
        self.cache_budget.as_ref().map(|b| b.total_bytes)
    }

    /// Current number of cached pretoken entries (short table + long map).
    /// Drops back toward vocab-seed level when a budgeted cache wipes.
    pub fn cache_entries(&self) -> usize {
        self.pretoken_cache.len() + self.pretoken_cache_long.len()
    }

    /// Re-derive the budget split after a loader-phase mutation changes the
    /// seed footprint (a stale split could leave the grown seed with no
    /// post-wipe headroom).
    /// Loader-phase mutator epilogue: re-derive the budget split (which
    /// reseeds) when bounded, else just reseed.
    fn reseed_or_rederive(&mut self) {
        match self.cache_budget.as_ref().map(|b| b.total_bytes) {
            Some(total) => self.set_max_cache_bytes(Some(total)),
            None => self.reseed_cache(),
        }
    }

    /// Seed-level state in the existing table and arena, value-identical to
    /// a fresh fork's.
    fn reseed_cache(&mut self) {
        seed_into(
            &self.model,
            self.ignore_merges,
            &self.added_tokens,
            &mut self.pretoken_cache,
            &mut self.token_arena,
        );
    }

    /// Budget check at the top of the miss path (before the miss's own
    /// insert, so a wipe can never invalidate an arena offset it just
    /// packed). Returns whether a wipe happened — the caller's
    /// probe-reported insert slot is then stale.
    #[inline]
    fn wipe_if_over_budget(&mut self) -> bool {
        let Some(b) = &self.cache_budget else {
            return false;
        };
        let short_at_growth = self.pretoken_cache.capacity() >= b.short_slots
            && (self.pretoken_cache.len() + 1) * 4 > self.pretoken_cache.capacity() * 3;
        if !short_at_growth
            && self.token_arena.len() <= b.arena_entries + b.max_encoding
            && b.long_bytes_used <= b.long_bytes
        {
            return false;
        }
        self.wipe_generation();
        true
    }

    /// The generation wipe: zero the short table in place, drop arena
    /// tokens and long entries, re-seed. Every discarded entry re-misses.
    #[cold]
    #[inline(never)]
    fn wipe_generation(&mut self) {
        self.pretoken_cache.clear();
        self.token_arena.clear();
        self.pretoken_cache_long.clear();
        self.reseed_cache();
        let b = self
            .cache_budget
            .as_mut()
            .expect("wipe_generation only runs with a budget set");
        b.long_bytes_used = 0;
        b.generations += 1;
        // Give back capacity an oversized encoding grew mid-generation.
        self.token_arena.shrink_to(b.arena_entries + 4096);
    }

    /// Set the added tokens matched atomically by
    /// [`Self::encode_with_added_tokens_flat`]. Empty contents are ignored.
    pub fn set_added_tokens(&mut self, added_tokens: Vec<AddedTokenDef>) {
        let mut added_tokens: Vec<AddedTokenDef> = added_tokens
            .into_iter()
            .filter(|t| !t.content.is_empty())
            .collect();
        added_tokens.sort_by_key(|t| std::cmp::Reverse(t.content.len()));
        self.added_matcher = (!added_tokens.is_empty()).then(|| {
            aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(added_tokens.iter().map(|t| t.content.as_ref()))
                .expect("added-token automaton construction cannot fail")
        });
        self.added_tokens = added_tokens;
        // Restores the outgoing overwrites' seed values and applies the new ones.
        self.reseed_or_rederive();
    }

    /// Register one additional added token, extending the decode vocab when
    /// its id lies outside the base ranks.
    pub fn add_special_token(&mut self, content: Vec<u8>, id: TokenId) {
        self.add_special_tokens([(content, id)]);
    }

    /// Register a batch of special added tokens: all vocab entries are
    /// written first, then `set_added_tokens` rebuilds the matcher and the
    /// cache overwrites once.
    pub fn add_special_tokens(&mut self, tokens: impl IntoIterator<Item = (Vec<u8>, TokenId)>) {
        let mut added = self.added_tokens.clone();
        for (content, id) in tokens {
            let idx = id.0 as usize;
            // `make_mut` copies only when a fork holds the tables too (never
            // during loading).
            let vocab = Arc::make_mut(&mut self.model.vocab);
            if idx >= vocab.len() {
                vocab.resize(idx + 1, Arc::from(Vec::new().as_slice()));
            }
            if vocab[idx].is_empty() {
                vocab[idx] = content.clone().into();
                // A duplicate byte string switches `vocab_inv` to the new ID;
                // `set_added_tokens` below re-derives the cache overwrites.
                Arc::make_mut(&mut self.model.vocab_inv).insert(vocab[idx].clone(), id);
            }
            added.push(AddedTokenDef {
                content: content.into(),
                id,
                lstrip: false,
                rstrip: false,
            });
        }
        self.set_added_tokens(added);
    }

    /// Size of the vocabulary: one greater than the largest token ID,
    /// including added tokens (IDs with no assigned content count too).
    pub fn vocab_size(&self) -> usize {
        self.model.vocab.len()
    }

    /// Vocabulary entries as `(id, bytes)` pairs in ID order, including
    /// added tokens and skipping IDs with no assigned content.
    pub fn vocab_entries(&self) -> impl Iterator<Item = (u32, &[u8])> {
        super::vocab_entries(&self.model.vocab)
    }

    /// Merge rules as `(left, right)` byte pairs in merge-priority order
    /// (priority equals the merged token's ID for tiktoken vocabularies;
    /// rank-mapped vocabularies keep their explicit rank order).
    pub fn merge_entries(&self) -> Vec<(&[u8], &[u8])> {
        let model = &self.model;
        if let Some(rm) = model.ranked_merges.as_deref() {
            return super::ranked_merge_entries(rm, &model.vocab);
        }
        let mut ranked: Vec<(u32, u32, u32)> =
            model.merges.iter().map(|(&(a, b), &m)| (a.0, b.0, m.0)).collect();
        ranked.sort_unstable_by_key(|&(.., priority)| priority);
        ranked
            .into_iter()
            .map(|(a, b, _)| {
                (
                    model.vocab[a as usize].as_ref(),
                    model.vocab[b as usize].as_ref(),
                )
            })
            .collect()
    }

    /// Added-token contents paired with their `rstrip` flag, for
    /// `pretokenize::safe_split_ranges`: an rstrip occurrence must not end
    /// exactly at a chunk boundary, or the whitespace it would absorb lands
    /// at the start of the next chunk and encodes as plain text.
    pub fn added_token_split_blockers(&self) -> Vec<(&[u8], bool)> {
        self.added_tokens
            .iter()
            .map(|t| (t.content.as_ref(), t.rstrip))
            .collect()
    }

    /// Leftmost added-token occurrence at or after `from`, longest on ties.
    /// Returns `(start, end, index into added_tokens)`.
    fn find_added_token(&self, bytes: &[u8], from: usize) -> Option<(usize, usize, usize)> {
        let m = self.added_matcher.as_ref()?.find(&bytes[from..])?;
        Some((from + m.start(), from + m.end(), m.pattern().as_usize()))
    }

    /// Shared piece walk of the added-token pipeline: split out added-token
    /// occurrences and hand each piece — the (possibly NFC-normalized)
    /// segment between occurrences, or the added token's ID — to `f` in
    /// input order.
    fn for_each_piece(&mut self, bytes: &[u8], mut f: impl FnMut(&mut Self, Piece<'_>)) {
        let normalize_nfc = self.normalize_nfc;
        let mut nfc_buf = String::new();
        let mut prefix_buf = Vec::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let (mut seg_end, added) = match self.find_added_token(bytes, pos) {
                Some((start, end, idx)) => {
                    let t = &self.added_tokens[idx];
                    (start, Some((end, t.id, t.lstrip, t.rstrip)))
                }
                None => (bytes.len(), None),
            };
            // An lstrip added token absorbs the whitespace before it (HF's
            // `\s*` on the left of the match); drop it from the segment.
            if let Some((_, _, true, _)) = added {
                seg_end = pos + trim_ws_end(&bytes[pos..seg_end]);
            }
            let mut segment = if normalize_nfc {
                nfc_segment(&bytes[pos..seg_end], &mut nfc_buf)
            } else {
                &bytes[pos..seg_end]
            };
            if self.add_prefix_space && !segment.is_empty() && segment[0] != b' ' {
                prefix_buf.clear();
                prefix_buf.push(b' ');
                prefix_buf.extend_from_slice(segment);
                segment = &prefix_buf;
            }
            f(self, Piece::Segment(segment, pos));
            match added {
                Some((end, id, _, rstrip)) => {
                    f(self, Piece::Added(id));
                    // An rstrip added token absorbs the whitespace after it.
                    pos = if rstrip { end + trim_ws_start(&bytes[end..]) } else { end };
                }
                None => break,
            }
        }
    }

    /// Encode raw text like the full HuggingFace `tokenizers` pipeline:
    /// added-token occurrences emit their single ID, the segments between
    /// them are pretokenized with this tokenizer's scheme and BPE-encoded.
    /// Tokens are appended to `out` as raw u32 ids (the batch engine's
    /// output shape).
    pub fn encode_with_added_tokens_flat(&mut self, bytes: &[u8], out: &mut Vec<u32>) {
        let pt = self.pretokenizer_type;
        self.for_each_piece(bytes, |this, piece| match piece {
            Piece::Segment(segment, _) => this.memoized_encode_flat(pt.pretokenize(segment), out),
            Piece::Added(id) => out.push(id.0),
        });
    }

    /// Encode each pretoken through the cache, calling `f` with one token
    /// slice per pretoken. A thin wrapper over [`Self::memoized_encode_flat`]:
    /// each chunk's tokens land in a reused buffer with per-pretoken end
    /// offsets recorded on the side.
    pub fn memoized_encode<'i>(
        &mut self,
        mut pretokens: impl PretokenSpans<'i>,
        mut f: impl FnMut(&[TokenId]),
    ) {
        let mut batch = SpanBatch::new();
        let mut out: Vec<u32> = Vec::new();
        let mut ends = [0usize; PRETOKEN_CHUNK];
        loop {
            let cache = &self.pretoken_cache;
            let n = pretokens.fill_spans_keyed(&mut batch, &|h| cache.prefetch_l2(h));
            if n == 0 {
                break;
            }
            out.clear();
            self.probe_emit_chunk(&batch, n, &mut out, |i, w| ends[i] = w);
            let mut start = 0;
            for &end in &ends[..n] {
                // SAFETY: TokenId is repr(transparent) over u32, and the
                // recorded ends partition `out` (0 <= start <= end <= len).
                f(unsafe {
                    std::slice::from_raw_parts(
                        out.as_ptr().add(start) as *const TokenId,
                        end - start,
                    )
                });
                start = end;
            }
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
    }

    /// Flat variant of [`Self::memoized_encode`]: the identical token stream
    /// appended to `out` as raw u32 ids. Runs in chunks of `PRETOKEN_CHUNK`
    /// pretokens through two phases — pull spans from the walker with keys
    /// derived and probe lines prefetched into L2 on the way out, then
    /// probe and emit — so each probe line has a chunk of latency to arrive.
    pub fn memoized_encode_flat<'i>(
        &mut self,
        mut pretokens: impl PretokenSpans<'i>,
        out: &mut Vec<u32>,
    ) {
        let mut batch = SpanBatch::new();
        loop {
            let cache = &self.pretoken_cache;
            let n = pretokens.fill_spans_keyed(&mut batch, &|h| cache.prefetch_l2(h));
            if n == 0 {
                break;
            }
            self.probe_emit_chunk(&batch, n, out, |_, _| {});
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
    }

    /// Probe-and-emit for one chunk: every iteration stores the probed
    /// value's four token lanes unconditionally at the write cursor and
    /// advances by the token count only when the fast predicate (pair hit ∧
    /// inline value ∧ short key) holds; stores past the cursor are dead.
    /// Everything else takes the `#[cold]` slow path. `record(i, cursor)`
    /// runs once per pretoken.
    ///
    /// Slack invariant: `out.capacity() >= cursor + 4 * (iterations left)`,
    /// established by the reserve below and re-established by the slow
    /// path after any reallocation, so the two 8-byte stores are always in
    /// bounds.
    #[inline(always)]
    fn probe_emit_chunk(
        &mut self,
        batch: &SpanBatch<'_>,
        n: usize,
        out: &mut Vec<u32>,
        mut record: impl FnMut(usize, usize),
    ) {
        // One check up front so the batch indexing below is provably in bounds.
        assert!(n <= PRETOKEN_CHUNK);
        if n == 0 {
            return;
        }
        out.reserve(4 * n);
        let mut w = out.len();
        // Loop-invariant raw cursors, refreshed only after the slow path
        // (the one thing that can move `out` or the table).
        let mut dst = out.as_mut_ptr();
        let mut table = self.pretoken_cache.probe_view();
        // Probe-stage prefetch distance: promotes the line L2 -> L1 (the
        // fill phase staged it into L2).
        const D: usize = 16;
        const _: () = assert!(D <= crate::pretokenize::SPAN_BATCH_SLACK);
        for i in 0..D.min(n) {
            table.prefetch(batch.entries[i].meta);
        }
        for i in 0..n {
            // Unclamped: the batch carries D slack entries past a full
            // chunk; stale/zero `meta` prefetches a harmless in-bounds line.
            table.prefetch(batch.entries[i + D].meta);
            let (key, h) = (batch.entries[i].key, batch.entries[i].meta);
            let (val, ext, found) = table.probe_pair(key, h);
            // `key != 0` folds the long-pretoken route in AND guards the
            // empty-slot sentinel (probe_pair matches key 0 against empty
            // slots); on !found the lanes below are another entry's, dead
            // because the cursor does not advance.
            let fast = found & (val & VAL_SPILL == 0) & (key != 0);
            // Lanes 1-2 in one u64 store, lanes 3-4 are `ext` verbatim.
            let ab = ((val >> 8) & 0x00FF_FFFF) | (val & 0xFFFF_FFFF_0000_0000);
            // SAFETY: the slack invariant leaves >= 4 u32s past `w`.
            unsafe {
                let p = dst.add(w);
                (p as *mut u64).write_unaligned(ab);
                (p.add(2) as *mut u64).write_unaligned(ext);
            }
            w += if fast { (val & 0x7F) as usize } else { 0 };
            if !fast {
                // For key == 0 `h` is really the span length; the slow path
                // never reads `h` on the long route, so it passes through
                // unfiltered (a select here got hoisted into the hot loop).
                // SAFETY: entry `i` was written by this chunk's fill, so
                // `ptr` points at a live span of the input's lifetime.
                let bytes = unsafe { batch.span(i) };
                w = self.probe_emit_slow(bytes, key, h, out, w);
                dst = out.as_mut_ptr();
                table = self.pretoken_cache.probe_view();
            }
            record(i, w);
        }
        // SAFETY: w <= capacity by the slack invariant, and every element
        // below `w` was written (fast advances never skip lanes; the slow
        // path appends through Vec).
        unsafe { out.set_len(w) };
    }

    /// Everything [`Self::probe_emit_chunk`]'s fast predicate rejects.
    /// Appends this pretoken's tokens at cursor `w` and returns the new
    /// cursor, re-establishing the slack invariant. `h` is only read when
    /// `key != 0`.
    #[cold]
    #[inline(never)]
    fn probe_emit_slow(
        &mut self,
        bytes: &[u8],
        key: u128,
        h: u64,
        out: &mut Vec<u32>,
        w: usize,
    ) -> usize {
        // SAFETY: elements below `w` are initialized and w <= capacity
        // (emit-loop invariant); Vec append methods need len in sync.
        unsafe { out.set_len(w) };
        if key != 0 {
            // A miss hands back the insert slot its walk found, so the
            // miss path's insert skips re-walking the (just-touched)
            // chain.
            match self.pretoken_cache.get_or_slot(key, h) {
                Ok((val, ext)) => {
                    let len = (val & 0x7F) as usize;
                    if val & VAL_SPILL == 0 {
                        out.extend_from_slice(&unpack_val_lanes(val, ext)[..len]);
                    } else {
                        let start = (val >> 32) as usize;
                        // SAFETY: recorded right after appending `len`
                        // tokens at `start`; the arena only shrinks in a
                        // generation wipe, which also clears every cache
                        // entry referencing it.
                        let toks =
                            unsafe { self.token_arena.get_unchecked(start..start + len) };
                        out.extend_from_slice(token_ids_as_u32s(toks));
                    }
                }
                Err(slot) => self.encode_pretoken_miss(bytes, key, h, slot, out),
            }
        } else {
            // Long pretokens (> 15 bytes, rare) always spill to the arena;
            // their token counts can exceed the packed-value range, so
            // they bypass it entirely.
            match self.pretoken_cache_long.get(bytes) {
                Some(&(offset, len)) => {
                    let start = offset as usize;
                    // SAFETY: as above.
                    let toks = unsafe {
                        self.token_arena.get_unchecked(start..start + len as usize)
                    };
                    out.extend_from_slice(token_ids_as_u32s(toks));
                }
                None => self.encode_pretoken_miss(bytes, 0, 0, 0, out),
            }
        }
        out.reserve(4 * PRETOKEN_CHUNK);
        out.len()
    }

    /// Cache-miss path of the probe/emit loop: BPE-encode `bytes`, record
    /// it in the table `key` routes to (the short table, or the long map
    /// when `key == 0`), and append its tokens to `out`. `slot` is the
    /// short-cache insert position reported by the failed `get_or_slot`
    /// probe (meaningful only when `key != 0`).
    #[inline(never)]
    fn encode_pretoken_miss(
        &mut self,
        bytes: &[u8],
        key: u128,
        h: u64,
        slot: usize,
        out: &mut Vec<u32>,
    ) {
        // Budget check FIRST: a wipe must precede this miss's `pack_val`
        // (whose arena offsets it would otherwise truncate away) and
        // invalidates the probe-reported `slot`. The reseeded table is a
        // subset of the one this key just missed in, so it still misses.
        let mut slot = slot;
        if self.wipe_if_over_budget() && key != 0 {
            slot = self
                .pretoken_cache
                .get_or_slot(key, h)
                .expect_err("reseed cannot add a key that just missed");
        }
        if key != 0 {
            // Short pretoken: stack buffer, same encoding as the vocab seed.
            let mut buf = [TokenId(0); SHORT_MERGE_MAX];
            let n = self.model.seed_encode(self.ignore_merges, bytes, &mut buf);
            let symbols = &buf[..n];
            let (val, ext) = Self::pack_val(symbols, &mut self.token_arena);
            self.pretoken_cache.insert_at(slot, key, h, val, ext);
            out.extend_from_slice(token_ids_as_u32s(symbols));
        } else {
            // Long pretoken (> 15 bytes, rare). No whole-pretoken
            // `vocab_inv` shortcut unless `ignore_merges`: vocab entries can
            // be merge-unreachable and HF returns the decomposition.
            let model = &self.model;
            let symbols = &mut self.symbol_scratch;
            symbols.clear();
            if self.ignore_merges
                && let Some(&id) = model.vocab_inv.get(bytes)
            {
                symbols.push(id);
            } else {
                symbols.resize(bytes.len(), TokenId(0));
                remap_bytes(model.byte_remapping.as_ref(), bytes, symbols);
                match (model.ranked_merges.as_deref(), model.pair_ranks.as_deref()) {
                    (Some(rm), _) => bpe_merge_symbols_ranked(rm, symbols),
                    (None, Some(table)) => bpe_merge_symbols_by_rank(
                        &|a, b| table.rank(a, b),
                        symbols,
                        &mut self.merge_scratch,
                    ),
                    (None, None) => bpe_merge_symbols_with_scratch(
                        &model.merges,
                        symbols,
                        &mut self.merge_scratch,
                    ),
                }
            }
            let len = symbols.len() as u32;
            let offset = self.token_arena.len() as u32;
            self.token_arena.extend_from_slice(symbols);
            // Accounted here, enforced by the next miss's budget check.
            if let Some(b) = &mut self.cache_budget {
                b.long_bytes_used += bytes.len() + CacheBudget::LONG_ENTRY_BYTES;
                b.max_encoding = b.max_encoding.max(len as usize);
            }
            self.pretoken_cache_long.insert(bytes.into(), (offset, len));
            out.extend_from_slice(token_ids_as_u32s(symbols));
        }
    }

    pub fn decode(&self, v: &[TokenId]) -> impl Iterator<Item = u8> {
        v.iter()
            .flat_map(|&token| self.model.vocab[token.0 as usize].as_ref())
            .copied()
    }

    /// Detailed cache stats for memory accounting:
    /// (short_len, short_cap, long_len, long_cap, long_key_bytes, arena_len, arena_cap).
    pub fn cache_mem_stats(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let long_key_bytes: usize = self.pretoken_cache_long.keys().map(|k| k.len()).sum();
        (
            self.pretoken_cache.len(),
            self.pretoken_cache.capacity(),
            self.pretoken_cache_long.len(),
            self.pretoken_cache_long.capacity(),
            long_key_bytes,
            self.token_arena.len(),
            self.token_arena.capacity(),
        )
    }
}

impl Debug for Tokenizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.model.vocab.len())
            .field("merges_count", &self.model.merges.len())
            .field("pair_ranks", &self.model.pair_ranks.is_some())
            .field("byte_remapping", &self.model.byte_remapping.is_some())
            .finish()
    }
}

/// Helpers shared by the test modules below.
#[cfg(test)]
mod test_util {
    use super::*;

    pub(super) fn gpt2_path() -> std::path::PathBuf {
        crate::test_hub::gpt2_tokenizer_json()
    }

    /// Uncached reference encode of one pretoken: byte remap + plain merge
    /// loop over the merges HashMap (no pair-rank table, no cache, no
    /// short-merge kernels).
    pub(super) fn plain_encode_pretoken(tok: &Tokenizer, pretoken: &[u8], out: &mut Vec<u32>) {
        let mut symbols = vec![TokenId(0); pretoken.len()];
        remap_bytes(tok.model.byte_remapping.as_ref(), pretoken, &mut symbols);
        bpe_merge_symbols(&tok.model.merges, &mut symbols);
        out.extend(symbols.iter().map(|t| t.0));
    }

    /// Token-stream equality with first-divergence reporting.
    pub(super) fn assert_ids_eq(actual: &[u32], expected: &[u32], label: &str) {
        if actual != expected {
            let i = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| actual.len().min(expected.len()));
            panic!(
                "{label}: diverged at token {i} (len {} vs expected {}):\n  actual[{i}..]   = {:?}\n  expected[{i}..] = {:?}",
                actual.len(),
                expected.len(),
                &actual[i..(i + 8).min(actual.len())],
                &expected[i..(i + 8).min(expected.len())],
            );
        }
    }

    /// Pretoken lengths through the two-phase walker path
    /// (`fill_spans_keyed`), for comparison against the Iterator path.
    pub(super) fn two_phase_lens<'a>(mut p: impl PretokenSpans<'a>) -> Vec<usize> {
        let mut batch = SpanBatch::new();
        let mut lens = Vec::new();
        loop {
            let n = p.fill_spans_keyed(&mut batch, &|_| {});
            for i in 0..n {
                lens.push(batch.entries[i].span_len());
            }
            if n < PRETOKEN_CHUNK {
                return lens;
            }
        }
    }

    pub(super) use crate::bpe::test_util::XorShift64;
}

#[cfg(test)]
mod tests {
    use super::test_util::assert_ids_eq;
    use super::*;

    /// `add_special_token` whose content duplicates an existing vocab byte
    /// string must resolve to the added ID in `vocab_inv`, the parent's
    /// seeded cache, and forked workers.
    #[test]
    fn add_special_token_duplicate_content_agrees_across_forks() {
        let encode = |t: &mut Tokenizer, input: &[u8]| -> Vec<TokenId> {
            let mut out = Vec::new();
            t.memoized_encode(crate::pretokenize::pretokenize_as_iter(input), |tokens| {
                out.extend_from_slice(tokens)
            });
            out
        };

        // Case 1: added ID above the duplicate's ID.
        let mut merges = MergeMap::default();
        merges.insert((TokenId(104), TokenId(105)), TokenId(256)); // 'h' 'i' -> "hi"
        let mut vocab: Vec<Vec<u8>> = (0..=255u32).map(|b| vec![b as u8]).collect();
        vocab.push(b"hi".to_vec()); // id 256 = "hi"
        let mut tok = Tokenizer::new(merges, vocab, None);
        tok.add_special_token(b"hi".to_vec(), TokenId(1000));
        assert_eq!(tok.model.vocab_inv.get(b"hi".as_slice()), Some(&TokenId(1000)));
        let mut fork = tok.fork();
        assert_eq!(
            encode(&mut tok, b"hi"),
            vec![TokenId(1000)],
            "parent cache must resolve the duplicate to the added ID (vocab_inv's answer)"
        );
        assert_eq!(
            encode(&mut fork, b"hi"),
            vec![TokenId(1000)],
            "forked worker must agree with the parent"
        );

        // Case 2 (mirror): added ID fills an empty placeholder BELOW the
        // duplicate's ID; the fork's reseed alone would pick the higher
        // ID (the merge result), diverging from vocab_inv and the parent.
        let mut merges = MergeMap::default();
        merges.insert((TokenId(104), TokenId(105)), TokenId(257)); // 'h' 'i' -> "hi"
        let mut vocab: Vec<Vec<u8>> = (0..=255u32).map(|b| vec![b as u8]).collect();
        vocab.push(Vec::new()); // id 256: empty placeholder
        vocab.push(b"hi".to_vec()); // id 257 = "hi"
        let mut tok = Tokenizer::new(merges, vocab, None);
        tok.add_special_token(b"hi".to_vec(), TokenId(256));
        assert_eq!(tok.model.vocab_inv.get(b"hi".as_slice()), Some(&TokenId(256)));
        let mut fork = tok.fork();
        assert_eq!(encode(&mut tok, b"hi"), vec![TokenId(256)]);
        assert_eq!(encode(&mut fork, b"hi"), vec![TokenId(256)]);
    }

    /// GPT-2 takes the PairRankTable fast path (table == merges map), and
    /// the vocab seed serves every short vocab word as its own ID (every
    /// base entry is merge-reachable; <|endoftext|> gets the added-token
    /// overwrite).
    #[test]
    fn gpt2_pair_rank_table_and_vocab_seed() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        let tokenizer = load_hf_bpe(super::test_util::gpt2_path()).expect("load GPT-2 tokenizer");

        let table = tokenizer
            .model
            .pair_ranks
            .as_deref()
            .expect("GPT-2 must take the pair-rank fast path");
        for (&(a, b), &m) in tokenizer.model.merges.iter() {
            assert_eq!(table.rank(a, b), m.0, "pair ({}, {})", a.0, b.0);
        }
        // Dense negatives (byte × byte) and flat negatives must agree with
        // the map too.
        for a in (0..50257u32).step_by(97) {
            for b in (0..50257u32).step_by(89) {
                let expected = tokenizer
                    .model
                    .merges
                    .get(&(TokenId(a), TokenId(b)))
                    .map_or(u32::MAX, |m| m.0);
                assert_eq!(table.rank(TokenId(a), TokenId(b)), expected, "pair ({a}, {b})");
            }
        }

        let mut seeded = 0usize;
        for (id, bytes) in tokenizer.vocab_entries() {
            if !(1..=15).contains(&bytes.len()) {
                continue;
            }
            let key = pack_pretoken_key(bytes).unwrap();
            let (val, ext) = tokenizer
                .pretoken_cache
                .get_or_slot(key, pretoken_key_hash(key))
                .expect("short vocab entry must be seeded");
            // Inline 1-token value: the entry's own ID.
            assert_eq!(val, 1 | ((id as u64) << 8), "vocab entry {id}");
            assert_eq!(ext, 0, "vocab entry {id}");
            seeded += 1;
        }
        assert_eq!(tokenizer.pretoken_cache.len(), seeded);
        // A fork starts from the same seed, sharing the same table.
        let fork = tokenizer.fork();
        assert_eq!(fork.pretoken_cache.len(), seeded);
        assert!(fork.model.pair_ranks.is_some());
    }

    #[test]
    fn short_pretoken_cache_serves_repeated_pretokens() {
        use crate::pretokenize::{SpanIter, pack_pretoken_key, pretoken_key_hash};

        let merges = MergeMap::default();
        let vocab = (0..=u8::MAX).map(|byte| vec![byte]).collect();
        let mut tokenizer = Tokenizer::new(merges, vocab, None);
        let bytes = b"hello";

        let mut first = Vec::new();
        tokenizer.memoized_encode(SpanIter([Pretoken(bytes)].into_iter()), |tokens| {
            first.extend(tokens.iter().map(|token| token.0));
        });
        let expected: Vec<u32> = bytes.iter().map(|&byte| byte as u32).collect();
        assert_eq!(first, expected);

        // The 5-token encoding is too long to inline, so it spilled to the
        // arena, but the cache entry serves it either way.
        let key = pack_pretoken_key(bytes).unwrap();
        let h = pretoken_key_hash(key);
        assert!(tokenizer.pretoken_cache.get_or_slot(key, h).is_ok());

        let mut repeated = Vec::new();
        tokenizer.memoized_encode(SpanIter([Pretoken(bytes)].into_iter()), |tokens| {
            repeated.extend(tokens.iter().map(|token| token.0));
        });
        assert_eq!(repeated, first);
        // The 256 single-byte vocab entries are pre-seeded; "hello" is the
        // one entry the encodes added.
        assert_eq!(tokenizer.pretoken_cache.len(), 257);

        // The zero key marks empty slots in the short table, so an empty
        // pretoken (possible through the public API) must take the long-map
        // path.
        tokenizer.memoized_encode(SpanIter([Pretoken(b"")].into_iter()), |tokens| {
            assert!(tokens.is_empty());
        });
        assert!(tokenizer.pretoken_cache_long.contains_key(&b""[..]));
    }

    /// With no added tokens configured, the piece walk reduces to one
    /// whole-input segment: `encode_with_added_tokens_flat` must equal a
    /// direct `memoized_encode_flat` of the same scheme's pretokens.
    #[test]
    fn encode_with_added_tokens_matches_memoized_encode_all_schemes() {
        let schemes = [
            PretokenizerType::GPT2,
            PretokenizerType::GPT4,
            PretokenizerType::Qwen2,
            PretokenizerType::Qwen35,
            PretokenizerType::Olmo3,
            PretokenizerType::DeepSeekV3,
            PretokenizerType::O200k,
            PretokenizerType::Nemotron,
            PretokenizerType::Kimi,
        ];
        let input = "Hello, 世界! café 12345\r\ncan't  stop".as_bytes();

        for scheme in schemes {
            let make_tokenizer = || {
                let merges = MergeMap::default();
                let vocab = (0..=u8::MAX).map(|byte| vec![byte]).collect();
                Tokenizer::new(merges, vocab, None)
            };

            let mut reference = make_tokenizer();
            let mut expected = Vec::new();
            reference.memoized_encode_flat(scheme.pretokenize(input), &mut expected);

            let mut concrete = make_tokenizer();
            concrete.set_pretokenizer_type(scheme);
            let mut actual = Vec::new();
            concrete.encode_with_added_tokens_flat(input, &mut actual);
            assert_eq!(actual, expected, "dispatch differs for {scheme:?}");
        }
    }

    /// `from_ranks` on GPT-2's base vocab (in ID order) must rebuild exactly
    /// the merges tokenizer.json lists, and encode like the HF-loaded model.
    #[test]
    fn from_ranks_reconstructs_gpt2_merges() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        let mut reference = load_hf_bpe(test_util::gpt2_path()).expect("load GPT-2 tokenizer");
        // 256 bytes + 50000 merges; <|endoftext|> (50256) is an added token.
        let vocab: Vec<Vec<u8>> = reference.model.vocab[..50256].iter().map(|b| b.to_vec()).collect();
        let mut tok = Tokenizer::from_ranks(vocab).unwrap();
        assert_eq!(*tok.model.merges, *reference.model.merges);
        assert!(tok.model.byte_remapping.is_some());
        let text = b"This is a test string. Please tokenize it!";
        let (mut expected, mut actual) = (Vec::new(), Vec::new());
        reference.encode_with_added_tokens_flat(text, &mut expected);
        tok.encode_with_added_tokens_flat(text, &mut actual);
        assert_eq!(actual, expected);
        let ids: Vec<TokenId> = actual.iter().map(|&t| TokenId(t)).collect();
        assert_eq!(tok.decode(&ids).collect::<Vec<u8>>(), text);
    }

    /// Byte-level tokenizer plus `extra_vocab` entries, `pairs` merge
    /// rules (`(left, right, merged-id)`, rank = position) and `added`
    /// special tokens, with GPT-2 pretokenization. `ranked` builds the
    /// same model through the explicit-rank merge table, covering the
    /// ranked miss path's wipe handling.
    fn synth_tokenizer(
        extra_vocab: Vec<Vec<u8>>,
        pairs: &[(TokenId, TokenId, u32)],
        added: &[(&[u8], u32)],
        ranked: bool,
    ) -> Tokenizer {
        let mut vocab: Vec<Vec<u8>> = (0..=255u32).map(|b| vec![b as u8]).collect();
        vocab.extend(extra_vocab);
        let mut tok = if ranked {
            let mut rm = RankedMerges::default();
            for (rank, &(a, b, id)) in pairs.iter().enumerate() {
                rm.insert(crate::bpe::ranked_merge_key(a, b), (TokenId(id), rank as u32));
            }
            Tokenizer::new_ranked(rm, vocab, None)
        } else {
            let mut merges = MergeMap::default();
            for &(a, b, id) in pairs {
                merges.insert((a, b), TokenId(id));
            }
            Tokenizer::new(merges, vocab, None)
        };
        tok.set_pretokenizer_type(PretokenizerType::GPT2);
        tok.add_special_tokens(added.iter().map(|&(c, id)| (c.to_vec(), TokenId(id))));
        tok
    }

    /// Budget fixture: word merges, a pure special token, and an added token
    /// duplicating a vocab byte string ("the" -> 301 overrides the merge
    /// result 257), so a wipe must restore the added-token OVERWRITE.
    fn budget_test_tokenizer(ranked: bool) -> Tokenizer {
        let t = |b: u8| TokenId(b as u32);
        let pairs = [
            (t(b't'), t(b'h'), 256),
            (TokenId(256), t(b'e'), 257),
            (t(b'a'), t(b'n'), 258),
            (TokenId(258), t(b'd'), 259),
            (t(b'i'), t(b'n'), 260),
            (TokenId(260), t(b'g'), 261),
            (t(b'e'), t(b'r'), 262),
        ];
        let extra = [
            b"th".as_slice(), b"the", b"an", b"and", b"in", b"ing", b"er",
        ]
        .map(<[u8]>::to_vec)
        .to_vec();
        let added: &[(&[u8], u32)] = &[(b"<|doc|>", 300), (b"the", 301)];
        synth_tokenizer(extra, &pairs, added, ranked)
    }

    const LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    const MIXED: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    /// `n` space-prefixed random words, `len` letters each, drawn
    /// uniformly from `alphabet`.
    fn random_words(
        rng: &mut test_util::XorShift64,
        n: usize,
        len: usize,
        alphabet: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..n {
            out.push(b' ');
            for _ in 0..len {
                out.push(alphabet[(rng.next_u64() % alphabet.len() as u64) as usize]);
            }
        }
        out
    }

    /// Generation wipes since the budget was set (0 when unbounded).
    fn wipe_gens(tok: &Tokenizer) -> u64 {
        tok.cache_budget.as_ref().map_or(0, |b| b.generations)
    }

    /// Budget-test corpus: Zipf-ish common words, rare random words (arena
    /// spills), > 15-byte words (long map), digits, unicode, punctuation,
    /// and added tokens, interleaved so every path runs across each wipe.
    fn budget_test_corpus(n_words: usize, seed: u64) -> Vec<u8> {
        let mut rng = test_util::XorShift64(seed);
        let common: [&str; 20] = [
            "the", "and", "ing", "her", "that", "with", "for", "was", "his", "not", "this",
            "but", "from", "they", "she", "which", "were", "been", "have", "their",
        ];
        let unicode: [&str; 7] = [
            "世界", "café", "naïve", "🦀🔥", "héllo", "Übung", "переменная",
        ];
        let mut out: Vec<u8> = Vec::new();
        for i in 0..n_words {
            match rng.next_u64() % 100 {
                0..=39 => {
                    // Zipf-ish head: min of two draws skews low indices.
                    let idx = (rng.next_u64() % 20).min(rng.next_u64() % 20) as usize;
                    out.extend_from_slice(common[idx].as_bytes());
                }
                40..=79 => {
                    // Rare short words (<= 15 bytes with the leading
                    // space): mostly-distinct cache pressure; 4+ letters
                    // encode to 5+ tokens with the space, spilling to
                    // the arena.
                    let len = 3 + (rng.next_u64() % 8) as usize;
                    for _ in 0..len {
                        out.push(b'a' + (rng.next_u64() % 26) as u8);
                    }
                }
                80..=86 => {
                    // > 15 bytes: the long-map path.
                    let len = 16 + (rng.next_u64() % 40) as usize;
                    for _ in 0..len {
                        out.push(b'a' + (rng.next_u64() % 26) as u8);
                    }
                }
                87..=90 => {
                    let len = 1 + (rng.next_u64() % 12) as usize;
                    for _ in 0..len {
                        out.push(b'0' + (rng.next_u64() % 10) as u8);
                    }
                }
                91..=94 => {
                    out.extend_from_slice(unicode[(rng.next_u64() % 7) as usize].as_bytes());
                }
                95..=96 => out.extend_from_slice(b"!?.,;:()[]{}--"),
                97..=98 => out.extend_from_slice(b"<|doc|>"),
                _ => out.extend_from_slice(b"the weather"),
            }
            out.push(if i % 17 == 0 { b'\n' } else { b' ' });
        }
        out
    }

    /// PARITY + BOUNDS: a budgeted tokenizer that wipes several times must
    /// produce an unbounded one's exact token stream (id-as-rank and
    /// explicit-rank miss paths) with every cache within its share, and a
    /// wipe must restore the added-token overwrite ("the" -> 301) as seen
    /// through the raw memoized path.
    #[test]
    fn budgeted_wipe_parity_and_bounds() {
        for ranked in [false, true] {
            let corpus = budget_test_corpus(300_000, 0x1234_5678_9ABC_DEF0);

            let mut unbounded = budget_test_tokenizer(ranked);
            unbounded.set_max_cache_bytes(None);
            let mut expected: Vec<u32> = Vec::new();
            unbounded.encode_with_added_tokens_flat(&corpus, &mut expected);
            assert_eq!(wipe_gens(&unbounded), 0);

            let mut budgeted = budget_test_tokenizer(ranked);
            budgeted.set_max_cache_bytes(Some(4 << 20));
            let (short_slots, arena_entries, long_bytes) = {
                let b = budgeted.cache_budget.as_ref().unwrap();
                (b.short_slots, b.arena_entries, b.long_bytes)
            };
            assert!(
                budgeted.pretoken_cache.capacity() <= short_slots,
                "short table starts above its ceiling"
            );

            // A budgeted fork inherits the full budget with fresh counters.
            let fork = budgeted.fork();
            assert!(fork.pretoken_cache.capacity() <= short_slots);
            let fb = fork.cache_budget.as_ref().unwrap();
            assert_eq!(fb.total_bytes, 4 << 20);
            assert_eq!((fb.generations, fb.long_bytes_used), (0, 0));

            let mut actual: Vec<u32> = Vec::new();
            budgeted.encode_with_added_tokens_flat(&corpus, &mut actual);

            let gens = wipe_gens(&budgeted);
            assert!(
                gens >= 3,
                "expected several wipes at a 4 MB budget, got {gens} (ranked={ranked})"
            );
            assert_ids_eq(&actual, &expected, &format!("budgeted (ranked={ranked})"));

            let (short_len, short_cap, _, _, long_key_bytes, arena_len, _) =
                budgeted.cache_mem_stats();
            let b = budgeted.cache_budget.as_ref().unwrap();
            assert!(short_cap <= short_slots, "short table grew past its ceiling");
            assert!(short_len * 4 <= short_cap * 3, "short table past 3/4 load");
            assert!(
                arena_len <= arena_entries + b.max_encoding + 256,
                "arena_len {arena_len} exceeds its budget {arena_entries}"
            );
            assert!(
                b.long_bytes_used <= long_bytes + 4096,
                "long map {} exceeds its budget {long_bytes}",
                b.long_bytes_used
            );
            assert!(long_key_bytes <= b.long_bytes_used);

            let encode_raw = |t: &mut Tokenizer, input: &[u8]| -> Vec<TokenId> {
                let mut out = Vec::new();
                t.memoized_encode(crate::pretokenize::pretokenize_as_iter(input), |toks| {
                    out.extend_from_slice(toks)
                });
                out
            };
            assert_eq!(
                encode_raw(&mut budgeted, b"the"),
                vec![TokenId(301)],
                "wipe lost the added-token overwrite (ranked={ranked})"
            );
            assert_eq!(encode_raw(&mut unbounded, b"the"), vec![TokenId(301)]);

            // Dropping the budget restores unbounded semantics without
            // touching cache contents.
            budgeted.set_max_cache_bytes(None);
            assert_eq!(wipe_gens(&budgeted), 0);
            assert!(budgeted.cache_budget.is_none());
        }
    }

    /// HEADROOM: a vocab whose short-entry count lands just under
    /// `required_capacity`'s 3/4 bound must get its ceiling doubled, or
    /// the table wipes every few misses.
    #[test]
    fn budgeted_wipe_headroom_near_seed_boundary() {
        let extra = (0..48_500u32).map(|i| format!("v{i:05}").into_bytes()).collect();
        let mut tok = synth_tokenizer(extra, &[], &[], false);
        tok.set_max_cache_bytes(Some(4 << 20));
        let short_slots = tok.cache_budget.as_ref().unwrap().short_slots;
        assert!(
            tok.pretoken_cache.len() * 8 <= short_slots * 5,
            "seed {} over 5/8 of {short_slots} slots",
            tok.pretoken_cache.len()
        );

        // ~124k distinct 3-letter words: 4-token inline values, so the
        // short table is the only wipe trigger.
        let mut rng = test_util::XorShift64(0x0DDB_1A5E_5BAD_5EED);
        let corpus = random_words(&mut rng, 300_000, 3, MIXED);
        let mut out: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(&corpus, &mut out);
        assert!(!out.is_empty());

        let gens = wipe_gens(&tok);
        assert!(gens >= 1, "corpus no longer fills the table");
        assert!(gens <= 10, "wipe-thrash: {gens} wipes for ~124k distinct pretokens");
        let (_, short_cap, ..) = tok.cache_mem_stats();
        assert_eq!(short_cap, short_slots, "a wiping table sits exactly at its ceiling");
    }

    /// GIANT PRETOKEN: a recurring pretoken whose encoding alone exceeds
    /// the arena sub-budget must not force a wipe per occurrence.
    #[test]
    fn budgeted_wipe_giant_pretoken_no_thrash() {
        let mut rng = test_util::XorShift64(0x61A7_0000_C0FF_EE00);
        // One 300k-letter pretoken; 'h' excluded so the added token
        // "the" cannot split it into many medium segments (a different
        // overflow shape than the single-encoding one pinned here).
        let giant = random_words(&mut rng, 1, 300_000, b"abcdefgijklmnopqrstuvwxyz");
        let occurrences = 30usize;
        let mut corpus = Vec::new();
        for i in 0..occurrences {
            corpus.extend_from_slice(&giant);
            corpus.extend_from_slice(format!(" filler{i} words here\n").as_bytes());
        }

        let mut unbounded = budget_test_tokenizer(false);
        unbounded.set_max_cache_bytes(None);
        let mut expected: Vec<u32> = Vec::new();
        unbounded.encode_with_added_tokens_flat(&corpus, &mut expected);

        let mut tok = budget_test_tokenizer(false);
        tok.set_max_cache_bytes(Some(4 << 20));
        // Test premise: the giant's ~297k-token encoding exceeds the
        // arena sub-budget outright.
        let arena_entries = tok.cache_budget.as_ref().unwrap().arena_entries;
        assert!(
            arena_entries < 290_000,
            "premise broken: arena sub-budget {arena_entries} no longer \
             below the giant's encoding"
        );
        let mut out: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(&corpus, &mut out);
        assert_eq!(out, expected, "budgeted output diverged");

        let gens = wipe_gens(&tok);
        assert!(
            (gens as usize) <= 2,
            "wipe per giant occurrence: {gens} wipes for {occurrences} occurrences"
        );
    }

    /// REDERIVE: loader-phase mutations after a budget is set must
    /// recompute the split, or a seed grown past a stale arena floor
    /// wipes on every miss.
    #[test]
    fn budgeted_rederive_after_loader_mutation() {
        let mut unbounded = budget_test_tokenizer(false);
        unbounded.set_max_cache_bytes(None);
        let mut tok = budget_test_tokenizer(false);
        // Tight budget: the short table swallows all of it, so the arena
        // sub-budget sits at its seed-derived floor.
        tok.set_max_cache_bytes(Some(2 << 20));
        let stale_arena = tok.cache_budget.as_ref().unwrap().arena_entries;
        assert!(stale_arena < 8192, "premise: floor-bound arena sub-budget, got {stale_arena}");

        // 2000 added tokens whose 11-byte contents seed as 11-token
        // arena spills: 22k entries, far past the stale 4096 floor.
        let added: Vec<(Vec<u8>, TokenId)> = (0..2000u32)
            .map(|i| (format!("«tok{i:04}»").into_bytes(), TokenId(1000 + i)))
            .collect();
        tok.add_special_tokens(added.clone());
        unbounded.add_special_tokens(added);

        let b = tok.cache_budget.as_ref().unwrap();
        assert_eq!(b.total_bytes, 2 << 20, "budget lost across loader mutation");
        assert_eq!(b.generations, 0);
        let arena_entries = b.arena_entries;
        assert!(
            tok.token_arena.len() <= arena_entries,
            "re-derived arena floor {arena_entries} below the new seed {}",
            tok.token_arena.len()
        );
        assert!(arena_entries > stale_arena);

        let corpus = budget_test_corpus(50_000, 0xFEED_FACE_CAFE_F00D);
        let mut expected: Vec<u32> = Vec::new();
        unbounded.encode_with_added_tokens_flat(&corpus, &mut expected);
        let mut out: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(&corpus, &mut out);
        assert_eq!(out, expected, "budgeted output diverged");

        let gens = wipe_gens(&tok);
        assert!(gens >= 1);
        assert!(
            gens <= 40,
            "wipe-thrash after loader mutation: {gens} wipes for 50k words"
        );
        let b = tok.cache_budget.as_ref().unwrap();
        let (_, _, _, _, _, arena_len, _) = tok.cache_mem_stats();
        assert!(arena_len <= b.arena_entries + b.max_encoding + 256);
    }

    /// DEFAULT LIFECYCLE: a fresh tokenizer reports the default budget,
    /// builds its table once at seed size (no ceiling presize, no eager
    /// arena prealloc), doubles below the ceiling with zero wipes, and
    /// wipes only at 3/4 load AT the ceiling.
    #[test]
    fn default_budget_lifecycle() {
        // The fixture runs a loader-shaped sequence: constructor,
        // set_pretokenizer_type, add_special_tokens.
        let mut tok = budget_test_tokenizer(false);
        assert_eq!(tok.max_cache_bytes(), Some(Tokenizer::DEFAULT_MAX_CACHE_BYTES));
        assert_eq!(tok.fork().max_cache_bytes(), tok.max_cache_bytes());
        assert_eq!(
            tok.pretoken_cache.capacity(),
            1 << 16,
            "default budget must not presize the table"
        );
        assert!(tok.token_arena.capacity() < 1 << 20, "eager arena preallocation");
        // Loader-phase mutation re-derives the split.
        tok.add_special_tokens([(b"<|extra|>".to_vec(), TokenId(400))]);
        assert_eq!(tok.pretoken_cache.capacity(), 1 << 16);
        assert_eq!(tok.max_cache_bytes(), Some(Tokenizer::DEFAULT_MAX_CACHE_BYTES));

        // 16 MiB gives a 2^18-slot ceiling to observe growth against;
        // the ceiling-growth mechanics are budget-independent.
        tok.set_max_cache_bytes(Some(16 << 20));
        let ceiling = tok.cache_budget.as_ref().unwrap().short_slots;
        assert_eq!(ceiling, 1 << 18);
        assert_eq!(tok.pretoken_cache.capacity(), 1 << 16, "budget must not presize either");

        let mut rng = test_util::XorShift64(0xCE11_1216_5107_C047);
        let mut out: Vec<u32> = Vec::new();
        // ~35k distinct 3-letter words: under the seed table's 3/4
        // threshold (inline 4-token values — no arena traffic).
        tok.encode_with_added_tokens_flat(&random_words(&mut rng, 40_000, 3, MIXED), &mut out);
        assert_eq!(wipe_gens(&tok), 0);
        assert_eq!(tok.pretoken_cache.capacity(), 1 << 16, "grew too early");
        // ~140k distinct: grows 2^16 -> 2^17 -> 2^18 with zero wipes
        // (the ceiling's 196k threshold is unreachable with 3 letters).
        tok.encode_with_added_tokens_flat(&random_words(&mut rng, 600_000, 3, MIXED), &mut out);
        assert_eq!(wipe_gens(&tok), 0, "wiped below the ceiling");
        assert_eq!(tok.pretoken_cache.capacity(), ceiling, "must reach the ceiling");
        // ~120k more distinct 4-letter words push load past 3/4 AT the
        // ceiling (spills stay well under the arena sub-budget).
        tok.encode_with_added_tokens_flat(&random_words(&mut rng, 150_000, 4, LOWER), &mut out);
        assert!(wipe_gens(&tok) >= 1, "no wipe at the ceiling");
        assert_eq!(tok.pretoken_cache.capacity(), ceiling, "wipe must keep the ceiling");

        tok.set_max_cache_bytes(None);
        assert_eq!(tok.max_cache_bytes(), None);
        assert_eq!(tok.fork().max_cache_bytes(), None);
    }
}

/// Heavy differentials of the cached encode paths against the uncached
/// per-pretoken reference: OWT-scale tests (`#[ignore]`d, each with its
/// cargo command), plus fast tests for `pack_pretoken_key` at every page
/// offset and the vocab-seed merge-decomposition rule.
#[cfg(test)]
mod verify_heavy {
    use super::test_util::{XorShift64, assert_ids_eq, gpt2_path, plain_encode_pretoken};
    use super::*;
    use crate::load_tokenizer::hf::load_hf_bpe;
    use std::io::Read;

    fn load_owt(max_bytes: usize) -> Vec<u8> {
        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let f = std::fs::File::open(&path).expect("open ~/data/owt_train.txt");
        let mut input = Vec::new();
        f.take(max_bytes as u64).read_to_end(&mut input).unwrap();
        while !input.is_empty() && std::str::from_utf8(&input).is_err() {
            input.pop();
        }
        input
    }

    /// Cut `input` at the first newline at or after `at` (whole input if none).
    fn cut_at_newline(input: &[u8], at: usize) -> &[u8] {
        let at = at.min(input.len());
        match memchr::memchr(b'\n', &input[at..]) {
            Some(off) => &input[..at + off + 1],
            None => input,
        }
    }

    /// Token-for-token comparison of the cached public path against a
    /// per-pretoken plain-merge walk of the same piece stream (the
    /// added-token split itself is covered by `join_differential`).
    fn compare_cached_vs_reference(tok: &mut Tokenizer, input: &[u8], label: &str, verbose: bool) {
        let mut cached: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(input, &mut cached);

        let mut idx = 0usize;
        let mut scratch: Vec<u32> = Vec::new();
        tok.for_each_piece(input, |this, piece| match piece {
            Piece::Segment(segment, pos) => {
                let mut seg_off = 0usize;
                for pretoken in this.pretokenizer_type.pretokenize(segment) {
                    scratch.clear();
                    plain_encode_pretoken(this, pretoken.0, &mut scratch);
                    let got = cached.get(idx..idx + scratch.len());
                    if got != Some(&scratch[..]) {
                        // Approximate: NFC or a prepended prefix space can
                        // shift lengths vs the raw input.
                        let byte_off = pos + seg_off;
                        let ctx_start = byte_off.saturating_sub(40).min(input.len());
                        let ctx_end = (byte_off + pretoken.0.len() + 40).min(input.len());
                        panic!(
                            "{label}: encode mismatch at byte offset ~{byte_off} (input len {}), token index {idx}\n  \
                             pretoken ({} bytes): {:?}\n  expected ids: {:?}\n  cached ids:   {:?}\n  context: {:?}",
                            input.len(),
                            pretoken.0.len(),
                            String::from_utf8_lossy(pretoken.0),
                            scratch,
                            &cached[idx.min(cached.len())..(idx + scratch.len() + 4).min(cached.len())],
                            String::from_utf8_lossy(&input[ctx_start..ctx_end]),
                        );
                    }
                    idx += scratch.len();
                    seg_off += pretoken.0.len();
                }
            }
            Piece::Added(id) => {
                assert_eq!(
                    cached.get(idx).copied(),
                    Some(id.0),
                    "{label}: added-token id mismatch at token index {idx}"
                );
                idx += 1;
            }
        });
        assert_eq!(
            idx,
            cached.len(),
            "{label}: cached stream has {} extra trailing tokens",
            cached.len() - idx
        );
        if verbose {
            eprintln!(
                "{label}: all {idx} tokens match on {:.1} MB",
                input.len() as f64 / 1e6
            );
        }
    }

    /// Added-token differential: join ~1 MB corpus pieces with the first
    /// added token and check the public encode equals
    /// concat(plain-encode(piece), sep_id, ...), built without the matcher.
    fn join_differential(tok: &mut Tokenizer, corpus: &[u8], label: &str) {
        let Some((sep, sep_id)) = tok.added_tokens.first().map(|t| (t.content.to_vec(), t.id)) else {
            eprintln!("{label}: no added tokens registered; skipping join differential");
            return;
        };
        // Mask every added-token occurrence so pieces are separator-free.
        let mut corpus: Vec<u8> = corpus.to_vec();
        for t in tok.added_tokens.clone() {
            let content = &t.content;
            let hits: Vec<usize> = memchr::memmem::find_iter(&corpus, &content[..]).collect();
            for pos in hits {
                corpus[pos] = b'~';
            }
        }
        let corpus = &corpus[..];
        let mut expected: Vec<u32> = Vec::new();
        let mut joined: Vec<u8> = Vec::new();
        let mut nfc_buf = String::new();
        let mut start = 0usize;
        let mut pieces = 0usize;
        while start < corpus.len() {
            let target = (start + (1 << 20)).min(corpus.len());
            let end = match memchr::memchr(b'\n', &corpus[target..]) {
                Some(off) => target + off + 1,
                None => corpus.len(),
            };
            let piece = &corpus[start..end];
            start = end;
            if tok
                .added_tokens
                .iter()
                .any(|t| memchr::memmem::find(piece, &t.content).is_some())
            {
                continue;
            }
            joined.extend_from_slice(piece);
            joined.extend_from_slice(&sep);
            let seg = if tok.normalize_nfc {
                nfc_segment(piece, &mut nfc_buf)
            } else {
                piece
            };
            for pretoken in tok.pretokenizer_type.pretokenize(seg) {
                plain_encode_pretoken(tok, pretoken.0, &mut expected);
            }
            expected.push(sep_id.0);
            pieces += 1;
        }
        let mut cached: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(&joined, &mut cached);
        assert_ids_eq(&cached, &expected, &format!("{label}: join differential"));
        assert!(pieces > 0, "{label}: join differential ran on zero pieces (vacuous)");
        eprintln!(
            "{label}: join differential ok — {pieces} pieces, {} tokens, sep {:?} id {}",
            cached.len(),
            String::from_utf8_lossy(&sep),
            sep_id.0
        );
    }

    /// The callback path (`memoized_encode`) vs the uncached reference on
    /// 50 MB of OWT.
    /// `cargo test --release verify_memoized_encode_matches_reference_owt_50m -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 50 MB of OWT; run explicitly in release mode"]
    fn verify_memoized_encode_matches_reference_owt_50m() {
        let mut tokenizer = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let all = load_owt(50_000_000);
        let input = &all[..];

        let mut cached: Vec<u32> = Vec::new();
        tokenizer
            .memoized_encode(crate::pretokenize::pretokenize_as_iter(input), |tokens| {
                cached.extend(tokens.iter().map(|t| t.0))
            });

        let mut idx = 0usize;
        for (pi, pretoken) in crate::pretokenize::pretokenize_as_iter(input).enumerate() {
            let mut reference = Vec::new();
            plain_encode_pretoken(&tokenizer, pretoken.0, &mut reference);
            assert!(
                cached[idx..(idx + reference.len()).min(cached.len())] == reference[..],
                "pretoken {pi} ({:?}) diverged: cached {:?} vs reference {:?}",
                String::from_utf8_lossy(pretoken.0),
                &cached[idx..(idx + reference.len()).min(cached.len())],
                reference,
            );
            idx += reference.len();
        }
        assert_eq!(idx, cached.len(), "cached encode produced extra tokens");
        eprintln!("all {idx} tokens match on {} MB", input.len() / 1_000_000);
    }

    /// 1 GB of OWT through the public GPT-2 path vs the uncached reference,
    /// plus a 100 MB join differential.
    /// `cargo test --release verify_gpt2_public_encode_matches_reference_owt_1g -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 1 GB of OWT; run explicitly in release mode"]
    fn verify_gpt2_public_encode_matches_reference_owt_1g() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let input = load_owt(1_000_000_000);
        assert!(input.len() > 900_000_000, "corpus too small: {}", input.len());
        compare_cached_vs_reference(&mut tok, &input, "gpt2-raw-1g", true);
        let mut tok2 = load_hf_bpe(gpt2_path()).unwrap();
        join_differential(&mut tok2, cut_at_newline(&input, 100_000_000), "gpt2-join-100m");
    }

    /// ~200 MB of OWT through the public path of olmo3/qwen2/qwen3_5/
    /// deepseek_v3 vs the uncached reference, plus a 25 MB join differential each.
    /// `cargo test --release verify_multi_public_encode_matches_reference_owt_200m -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 200 MB of OWT per tokenizer; run explicitly in release mode"]
    fn verify_multi_public_encode_matches_reference_owt_200m() {
        let input = load_owt(200_000_000);
        assert!(input.len() > 190_000_000, "corpus too small: {}", input.len());
        let mut ran = 0usize;
        // qwen3_5 (merge-unreachable vocab entries) last, so the clean
        // tokenizers report first on a regression.
        for (name, repo_id) in [
            ("olmo3", "allenai/Olmo-3-1025-7B"),
            ("qwen2", "Qwen/Qwen2-1.5B-Instruct"),
            ("deepseek_v3", "deepseek-ai/DeepSeek-V3"),
            ("qwen3_5", "Qwen/Qwen3.5-9B"),
        ] {
            let Some(path) = crate::test_hub::hf_tokenizer_json(repo_id) else {
                eprintln!("{name}: {repo_id} tokenizer.json not in the HF cache; skipping");
                continue;
            };
            let mut tok = match load_hf_bpe(&path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{name}: failed to load ({e}); skipping");
                    continue;
                }
            };
            eprintln!(
                "{name}: scheme {:?}, nfc {}, {} added tokens, vocab {}",
                tok.pretokenizer_type,
                tok.normalize_nfc,
                tok.added_tokens.len(),
                tok.vocab_size()
            );
            compare_cached_vs_reference(&mut tok, &input, name, true);
            let mut tok2 = load_hf_bpe(&path).unwrap();
            join_differential(&mut tok2, cut_at_newline(&input, 25_000_000), name);
            ran += 1;
        }
        assert!(ran >= 3, "only {ran} tokenizers loaded — expected at least olmo3/qwen2/deepseek_v3");
    }

    /// A merge-unreachable vocab entry (qwen3_5 has ~200, e.g. " Jap\u{f3}n")
    /// must encode as its merge decomposition, as HF `tokenizers` does
    /// without `ignore_merges` — the seed is precomputed misses, never the
    /// raw ID. HF ground truth: [604, 385, 3064].
    #[test]
    fn verify_vocab_seeded_cache_matches_merge_decomposition() {
        let Some(path) = crate::test_hub::hf_tokenizer_json("Qwen/Qwen3.5-9B") else {
            eprintln!("Skipping: Qwen/Qwen3.5-9B tokenizer.json not in the HF cache");
            return;
        };
        let mut tok = load_hf_bpe(&path).expect("load qwen3_5");
        let pretoken: &[u8] = " Jap\u{f3}n".as_bytes();

        let mut cached: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(pretoken, &mut cached);
        // HF `tokenizers` ground truth for this tokenizer.json.
        assert_eq!(
            cached,
            vec![604, 385, 3064],
            "seeded cache returned the merge-unreachable whole-vocab entry"
        );

        // Same rule on the long (> 15 byte) miss path, which used to take
        // a whole-pretoken vocab_inv shortcut: a merge-unreachable long
        // vocab entry must also encode as its decomposition. qwen3_5 id
        // 107517 is a 30-byte CJK phrase HF splits in 3.
        let long_entry = tok
            .vocab_entries()
            .find(|&(id, _)| id == 107517)
            .map(|(_, b)| b.to_vec())
            .expect("qwen3_5 vocab entry 107517");
        assert!(long_entry.len() > 15, "expected a long entry");
        let mut cached_long: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(&long_entry, &mut cached_long);
        assert_ne!(
            cached_long,
            vec![107517],
            "long miss path returned the merge-unreachable whole-vocab entry"
        );
    }

    /// `pack_pretoken_key`'s unaligned-16-byte fast path vs the naive lane
    /// copy at EVERY page offset (both branches of the page-boundary guard),
    /// all lengths 0..=15.
    #[test]
    fn verify_pack_pretoken_key_all_page_offsets() {
        use crate::pretokenize::pack_pretoken_key;
        let mut buf = vec![0u8; 12288];
        let mut rng = XorShift64(0x0123_4567_89AB_CDEF);
        for b in buf.iter_mut() {
            *b = rng.next_u64() as u8;
            if *b == 0 {
                *b = 1; // avoid zero lanes masking length-tag mistakes
            }
        }
        for start in 0..buf.len() - 16 {
            for n in 0..=15usize {
                let span = &buf[start..start + n];
                let key = pack_pretoken_key(span);
                let mut lanes = [0u8; 16];
                lanes[..n].copy_from_slice(span);
                let naive = if n == 0 {
                    0u128
                } else {
                    u128::from_le_bytes(lanes) | ((n as u128) << 120)
                };
                assert_eq!(
                    key,
                    Some(naive),
                    "pack_pretoken_key mismatch at buf offset {start} (page offset {}), len {n}",
                    (buf[start..].as_ptr() as usize) & 4095
                );
            }
        }
        // Length > 15 routes to the long map.
        assert_eq!(pack_pretoken_key(&buf[..16]), None);
    }
}

/// Alignment-invariance sweep: the walkers' output must not depend on the
/// span's heap address.
#[cfg(test)]
mod verify_alignment {
    use super::test_util::{XorShift64, gpt2_path, two_phase_lens};
    use super::*;
    use crate::load_tokenizer::hf::load_hf_bpe;

    #[test]
    fn verify_walker_alignment_invariance() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        // Inputs chosen to stress batch-edge machinery: long runs of one
        // class, class flips near multiples of 64, multi-byte chars
        // straddling batch edges, contractions, digit groups.
        let mut inputs: Vec<Vec<u8>> = Vec::new();
        for n in [63usize, 64, 65, 127, 128, 129, 255, 256, 300, 4096] {
            for fill in [&b"a"[..], b"5", b" ", b"!", b"\n", "\u{e9}".as_bytes(), "\u{597d}".as_bytes()] {
                let mut v = Vec::new();
                while v.len() < n {
                    v.extend_from_slice(fill);
                }
                inputs.push(v);
            }
        }
        let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
        const PIECES: &[&str] = &[
            " the", " a", "word", "05", "  ", "\n", "'s", "n't", ",", " \u{e9}t\u{e9}",
            "\u{597d}\u{597d}", " 123", "...", "\t", " I'm", "\u{2014}", "e", " ",
        ];
        for _ in 0..200 {
            let target = 80 + (rng.next_u64() % 400) as usize;
            let mut v = Vec::new();
            while v.len() < target {
                v.extend_from_slice(
                    PIECES[(rng.next_u64() % PIECES.len() as u64) as usize].as_bytes(),
                );
            }
            inputs.push(v);
        }

        for (which, input) in inputs.iter().enumerate() {
            // Copy the same bytes at every offset 0..64 of a fresh buffer;
            // walker output must be identical for all of them.
            let mut ref_lens: Option<Vec<usize>> = None;
            let mut ref_ids: Option<Vec<u32>> = None;
            for off in 0..64usize {
                let mut buf = vec![0u8; off + input.len() + 64];
                buf[off..off + input.len()].copy_from_slice(input);
                let span = &buf[off..off + input.len()];
                let lens = two_phase_lens(FastR50kPretokenizer::new(span));
                let mut ids: Vec<u32> = Vec::new();
                tok.memoized_encode_flat(FastR50kPretokenizer::new(span), &mut ids);
                match (&ref_lens, &ref_ids) {
                    (None, _) => {
                        ref_lens = Some(lens);
                        ref_ids = Some(ids);
                    }
                    (Some(rl), Some(ri)) => {
                        assert!(
                            &lens == rl,
                            "input {which}: pretoken lens differ at offset {off}\n  base: {rl:?}\n  off{off}: {lens:?}\n  input: {:?}",
                            String::from_utf8_lossy(input)
                        );
                        assert!(
                            &ids == ri,
                            "input {which}: token ids differ at offset {off} on {:?}",
                            String::from_utf8_lossy(input)
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }
        eprintln!("alignment invariance: {} inputs x 64 offsets ok", inputs.len());
    }
}

/// Walker edge conditions (truncated multi-byte UTF-8 at the buffer end,
/// invalid-UTF-8 garbage codepoints): every scheme must partition ARBITRARY
/// bytes contiguously, in bounds, identically on the Iterator and two-phase
/// paths, deterministically.
#[cfg(test)]
mod walker_edge {
    use super::test_util::{XorShift64, gpt2_path, plain_encode_pretoken, two_phase_lens};
    use super::*;
    use crate::load_tokenizer::hf::load_hf_bpe;

    /// Full public GPT-2 path vs the plain reference over r50k pretokens.
    fn check_public_gpt2_path(tok: &mut Tokenizer, span: &[u8]) {
        let mut expected: Vec<u32> = Vec::new();
        for p in FastR50kPretokenizer::new(span) {
            plain_encode_pretoken(tok, p.0, &mut expected);
        }
        let mut cached: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(span, &mut cached);
        assert!(
            cached == expected,
            "public path mismatch on {:?}: cached {:?} expected {:?}",
            String::from_utf8_lossy(span),
            cached,
            expected
        );
    }

    /// Assert a scheme's pretokens are a contiguous, non-empty, in-bounds
    /// partition of `span` (required for encode correctness; also catches
    /// walkers running past the buffer on truncated UTF-8).
    fn check_partition<'a>(
        span: &'a [u8],
        it: impl Iterator<Item = Pretoken<'a>>,
        scheme: &str,
    ) {
        let mut off = 0usize;
        for p in it {
            assert!(
                std::ptr::eq(p.0.as_ptr(), span[off..].as_ptr()),
                "{scheme}: non-contiguous pretoken at byte {off} of {:?}",
                String::from_utf8_lossy(span)
            );
            assert!(!p.0.is_empty(), "{scheme}: empty pretoken at byte {off}");
            off += p.0.len();
            assert!(
                off <= span.len(),
                "{scheme}: pretoken overruns span end ({off} > {}) on {:?}",
                span.len(),
                String::from_utf8_lossy(span)
            );
        }
        assert_eq!(
            off,
            span.len(),
            "{scheme}: pretokens cover {off} of {} bytes of {:?}",
            span.len(),
            String::from_utf8_lossy(span)
        );
    }

    /// Partition check (Iterator path) plus cached encode (two-phase
    /// `fill_spans_keyed` path) vs the plain per-pretoken reference over
    /// the Iterator's pretokens — so the two walker paths are compared
    /// against each other on every span.
    fn check_scheme_encode<'a, P>(
        tok: &mut Tokenizer,
        span: &'a [u8],
        make: impl Fn(&'a [u8]) -> P,
        scheme: &str,
    ) where
        P: PretokenSpans<'a>,
        P: Iterator<Item = Pretoken<'a>>,
    {
        check_partition(span, make(span), scheme);
        let mut got: Vec<u32> = Vec::new();
        tok.memoized_encode_flat(make(span), &mut got);
        let mut expected: Vec<u32> = Vec::new();
        for p in make(span) {
            plain_encode_pretoken(tok, p.0, &mut expected);
        }
        assert!(
            got == expected,
            "{scheme}: cached encode mismatch on {:?} (len {}):\n  cached   {:?}\n  expected {:?}",
            String::from_utf8_lossy(span),
            span.len(),
            got,
            expected
        );
    }

    fn check_all_schemes(tok: &mut Tokenizer, span: &[u8]) {
        check_scheme_encode(tok, span, FastR50kPretokenizer::new, "r50k");
        check_scheme_encode(tok, span, FastCl100kPretokenizer::new, "cl100k");
        check_scheme_encode(tok, span, FastQwen2Pretokenizer::new, "qwen2");
        check_scheme_encode(tok, span, FastQwen35Pretokenizer::new, "qwen3_5");
        check_scheme_encode(tok, span, FastOlmo3Pretokenizer::new, "olmo3");
        check_scheme_encode(tok, span, FastDeepSeekV3Pretokenizer::new, "deepseek_v3");
    }

    /// Truncated multi-byte UTF-8 at the buffer end, every shape: for each
    /// lead-byte length (2/3/4) every truncation point (1..len-1 available
    /// continuation bytes missing), plus lone continuation bytes and
    /// invalid 0xF5..=0xFF leads, behind assorted prefixes that put the
    /// truncated char after a letter run / digit run / space / whitespace
    /// run / punctuation / another unicode char. Exactly-sized heap
    /// allocations so any walker overrun is an observable OOB.
    #[test]
    fn walker_truncated_utf8_tail() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        // Leads: 2-byte (0xC3), 3-byte (0xE2, and 0xE0 low), 4-byte (0xF0,
        // 0xF4 high), invalid leads (0xF5, 0xF8, 0xFF), continuation (0x80,
        // 0xBF), and 0xC0/0xC1 (invalid 2-byte leads).
        let leads: &[&[u8]] = &[
            b"\xc3",
            b"\xe2",
            b"\xe2\x80",
            b"\xe0",
            b"\xe0\xa0",
            b"\xf0",
            b"\xf0\x9f",
            b"\xf0\x9f\x99",
            b"\xf4",
            b"\xf4\x8f",
            b"\xf4\x8f\xbf",
            b"\xf5",
            b"\xf8\x88",
            b"\xff",
            b"\xff\xff",
            b"\xff\xff\xff",
            b"\x80",
            b"\xbf",
            b"\xc0",
            b"\xc1",
        ];
        let prefixes: &[&[u8]] = &[
            b"",
            b"a",
            b"hello",
            b"hello ",
            b"123",
            b" ",
            b"  \n",
            b"!?",
            "é".as_bytes(),
            "好".as_bytes(),
            b"'s",
            b"\xff\xff\xff\xff", // complete invalid run before the tail
        ];
        for &lead in leads {
            for &prefix in prefixes {
                let mut buf = Vec::with_capacity(prefix.len() + lead.len());
                buf.extend_from_slice(prefix);
                buf.extend_from_slice(lead);
                check_all_schemes(&mut tok, &buf);
                check_public_gpt2_path(&mut tok, &buf);
            }
        }
    }

    /// Deterministic boundary fuzz: random spans (0-64 bytes; ASCII text,
    /// raw bytes incl. invalid UTF-8 and truncated tails, valid multi-byte
    /// UTF-8, whitespace runs) placed at the END of an exactly-sized
    /// allocation, through every scheme's walker (partition + cached-vs-
    /// plain encode) and the full public GPT-2 path. Fixed seed, no I/O
    /// beyond the tokenizer.
    #[test]
    fn walker_boundary_fuzz_memoized_vs_reference() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let mut rng = XorShift64(0x243F_6A88_85A3_08D3);
        const CHARS: &[&str] = &["é", "ü", "好", "日", "🙂", "ß", "—", "\u{0301}", "٣", "क"];
        let iters = if cfg!(debug_assertions) { 2_000 } else { 12_000 };
        for _ in 0..iters {
            let len = (rng.next_u64() % 65) as usize;
            let pad = (rng.next_u64() % 17) as usize;
            let mut buf: Vec<u8> = Vec::with_capacity(pad + len + 8);
            for _ in 0..pad {
                buf.push(rng.next_u64() as u8);
            }
            let span_start = buf.len();
            match rng.next_u64() % 4 {
                0 => {
                    // ASCII text: letters, digits, spaces, punct, contractions
                    const POOL: &[u8] = b" aetoAETO059'.,!-\n\t\"()s d";
                    while buf.len() - span_start < len {
                        buf.push(POOL[(rng.next_u64() % POOL.len() as u64) as usize]);
                    }
                }
                1 => {
                    // Raw bytes: full 0..=255, mostly invalid UTF-8,
                    // truncated tails included.
                    while buf.len() - span_start < len {
                        buf.push(rng.next_u64() as u8);
                    }
                }
                2 => {
                    // Valid UTF-8 mix: fill to >= len, then trim whole
                    // chars back to <= len so the span stays valid UTF-8.
                    while buf.len() - span_start < len {
                        if rng.next_u64().is_multiple_of(2) {
                            buf.push(b' ' + (rng.next_u64() % 94) as u8);
                        } else {
                            buf.extend_from_slice(
                                CHARS[(rng.next_u64() % CHARS.len() as u64) as usize].as_bytes(),
                            );
                        }
                    }
                    while buf.len() - span_start > len
                        || std::str::from_utf8(&buf[span_start..]).is_err()
                    {
                        buf.pop();
                    }
                }
                _ => {
                    // Whitespace-heavy
                    const POOL: &[u8] = b"   \n\n\t\r a5.";
                    while buf.len() - span_start < len {
                        buf.push(POOL[(rng.next_u64() % POOL.len() as u64) as usize]);
                    }
                }
            }
            buf.truncate(span_start + len.min(buf.len() - span_start));
            let span = &buf[span_start..];
            check_all_schemes(&mut tok, span);
            // <|endoftext|> (13 bytes) cannot occur in <= 64 random bytes.
            check_public_gpt2_path(&mut tok, span);
        }
        eprintln!("boundary fuzz: {iters} spans x 6 schemes ok");
    }

    /// Pretokens at the exact edge lengths of the key-packing and cache
    /// machinery (15/16 for the packed u128 key, 65535/65536 and beyond
    /// for long runs through the walkers), in letter/digit/space/punct/
    /// multi-byte/invalid fills, each in an exactly-sized allocation.
    #[test]
    fn walker_edge_length_pretokens() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let lens: &[usize] = &[
            1, 2, 7, 8, 14, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 255, 256, 4095, 4096,
            4097, 65_535, 65_536, 65_537, 70_003,
        ];
        let fills: &[&[u8]] = &[
            b"a",
            b"5",
            b" ",
            b"!",
            b"\n",
            "\u{e9}".as_bytes(),   // é (2-byte letter)
            "\u{597d}".as_bytes(), // 好 (3-byte letter)
            b"\xff",               // invalid UTF-8
        ];
        for &n in lens {
            for fill in fills {
                // Repeat fill to >= n bytes, then cut to n only for 1-byte
                // fills (multi-byte fills keep whole chars).
                let reps = n / fill.len() + usize::from(n % fill.len() != 0);
                if reps == 0 {
                    continue;
                }
                let exact = if fill.len() == 1 { n } else { reps * fill.len() };
                let mut buf: Vec<u8> = Vec::with_capacity(exact);
                while buf.len() < exact {
                    buf.extend_from_slice(fill);
                }
                buf.truncate(exact);
                check_all_schemes(&mut tok, &buf);
                // Space-prefixed variant hits the space-fused starts.
                let mut buf2: Vec<u8> = Vec::with_capacity(exact + 1);
                buf2.push(b' ');
                buf2.extend_from_slice(&buf);
                check_all_schemes(&mut tok, &buf2);
                // 0xFF run with a letter tail: the exact shape of the
                // >65 KB nondeterminism (the last garbage 4-byte decode
                // straddles into the letters).
                if fill == b"\xff" && exact >= 4 {
                    let mut buf3 = buf.clone();
                    let e = buf3.len();
                    buf3[e - 3..].copy_from_slice(b"cba");
                    check_all_schemes(&mut tok, &buf3);
                }
            }
        }
        eprintln!("edge-length pretokens ok");
    }

    /// Regression: 65534 x 0xFF + "cba", bare and space-prefixed, walked
    /// repeatedly on both paths while background threads churn the heap
    /// (an out-of-table class load once made the partition depend on
    /// neighbouring heap memory). Every round must agree.
    #[test]
    fn walker_ff_run_paths_agree_under_heap_churn() {
        // 16 spare bytes of capacity: `pack_pretoken_key`'s page-guarded
        // 16-byte load may overread within the page, which ASAN would flag
        // on an exactly-sized allocation.
        let mut a = Vec::with_capacity(65537 + 16);
        a.resize(65534, 0xFFu8);
        a.extend_from_slice(b"cba"); // len 65537
        let mut b = Vec::with_capacity(65538 + 16);
        b.push(b' ');
        b.extend_from_slice(&a); // len 65538
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let churners: Vec<_> = (0..4)
            .map(|t| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut keep: Vec<Vec<u8>> = Vec::new();
                    let mut i = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        // Vary size and content so the pages after any
                        // fresh allocation keep changing.
                        let sz = 1 << (12 + (i + t) % 8); // 4 KB .. 512 KB
                        keep.push(vec![(i as u8) ^ 0x5A; sz]);
                        if keep.len() > 8 {
                            keep.clear();
                        }
                        i += 1;
                    }
                })
            })
            .collect();
        let iter_lens = |span: &[u8]| -> Vec<usize> {
            FastR50kPretokenizer::new(span).map(|p| p.0.len()).collect()
        };
        let mut reference: Option<[Vec<usize>; 2]> = None;
        for round in 0..40 {
            let got = [
                {
                    let (i, t) = (iter_lens(&a), two_phase_lens(FastR50kPretokenizer::new(&a)));
                    assert_eq!(i, t, "round {round} span A: iterator vs two-phase");
                    i
                },
                {
                    let (i, t) = (iter_lens(&b), two_phase_lens(FastR50kPretokenizer::new(&b)));
                    assert_eq!(i, t, "round {round} span B: iterator vs two-phase");
                    i
                },
            ];
            match &reference {
                None => reference = Some(got),
                Some(r) => assert_eq!(
                    r,
                    &got,
                    "round {round}: partition changed between rounds (nondeterminism)"
                ),
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for c in churners {
            c.join().unwrap();
        }
    }
}
