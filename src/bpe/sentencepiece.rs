use crate::bpe::{RankedMerges, bpe_merge_symbols_ranked};
use crate::pretokenize::pack_pretoken_key;
use crate::token::TokenId;
use rustc_hash::FxBuildHasher;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// SentencePiece uses U+2581 (▁) as a space marker.
const SENTENCEPIECE_SPACE: char = '\u{2581}';
const SENTENCEPIECE_SPACE_STR: &str = "\u{2581}";
const SP_MARK: [u8; 3] = [0xE2, 0x96, 0x81];

/// How text divides into independently-encodable (and cacheable) word units.
/// Computed once at load from the pre-tokenizer config and the vocab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WordSplit {
    /// A unit starts at each ▁ that follows a non-▁ char: units look like
    /// `▁▁▁word`. Valid when no vocab piece contains a ▁ after a non-▁ char,
    /// so no merge can cross a unit boundary and per-unit BPE equals the
    /// global merge.
    SpaceRuns,
    /// Metaspace `split=true`: a unit starts at every ▁ (HF splits there, so
    /// this is exact regardless of the vocab).
    EveryMark,
    /// The vocab has boundary-crossing pieces; merge whole chunks, uncached.
    None,
}

/// How the raw fast path prepends the dummy-prefix ▁ to a chunk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RawPrepend {
    Never,
    /// `Prepend` normalizer: unconditional (Llama 2).
    Unguarded,
    /// Metaspace `always`: skipped when the chunk already starts with a mark.
    GuardedAlways,
    /// Metaspace `first`: like `GuardedAlways`, first chunk of the input only.
    GuardedFirst,
}

/// One step of a tokenizer.json `normalizer` sequence, applied in order to
/// each text chunk between added tokens (HF normalizes those independently).
pub enum NormOp {
    /// `Prepend`: unconditional prefix, e.g. Llama 2's "▁". HF's Prepend
    /// leaves empty chunks empty.
    Prepend(String),
    /// `Replace` with a literal `String` pattern.
    Replace { pattern: String, content: String },
    /// `Replace` with the `" {2,}"` regex (transformers' SpmConverter emits it
    /// for `remove_extra_whitespaces`): each run of 2+ ASCII spaces becomes
    /// `content`.
    CollapseSpaces { content: String },
    /// `Strip` Unicode whitespace.
    Strip { left: bool, right: bool },
    /// `Precompiled` charsmap (sentencepiece's nmt_nfkc and friends).
    Precompiled(PrecompiledCharsmap),
}

/// A precompiled charsmap plus lookup tables that let ASCII-dominant text
/// skip the per-grapheme trie walk (which runs at ~50 MB/s).
pub struct PrecompiledCharsmap {
    pre: spm_precompiled::Precompiled,
    /// `transform` of each single ASCII char; `None` = pass through.
    ascii_map: [Option<Box<str>>; 128],
    /// The "\r\n" grapheme's mapping (`None` = pass through).
    crlf: Option<Box<str>>,
    /// True when no printable ASCII char is remapped, so the SIMD clean-run
    /// scan only has to stop on control bytes and non-ASCII.
    fast_scan: bool,
}

impl PrecompiledCharsmap {
    pub(crate) fn new(pre: spm_precompiled::Precompiled) -> Self {
        let mut ascii_map: [Option<Box<str>>; 128] = std::array::from_fn(|_| None);
        let mut fast_scan = true;
        let mut buf = [0u8; 4];
        for b in 0..128u8 {
            let ch = b as char;
            let s = ch.encode_utf8(&mut buf);
            if let Some(norm) = pre.transform(s)
                && norm != s
            {
                if (0x20..0x7F).contains(&b) {
                    // A remapped printable would make clean runs unsound.
                    fast_scan = false;
                }
                ascii_map[b as usize] = Some(norm.into());
            }
        }
        let crlf = pre.transform("\r\n").map(Into::into);
        PrecompiledCharsmap {
            pre,
            ascii_map,
            crlf,
            fast_scan,
        }
    }

    /// Exactly `spm_precompiled::Precompiled::normalize_string`, but ASCII
    /// runs bypass the grapheme walk: printable ASCII chars are standalone
    /// grapheme clusters (only CR×LF joins, and only non-ASCII extends a
    /// cluster), and control chars break unconditionally on both sides, so
    /// the walk is only needed for spans around non-ASCII bytes — including
    /// one ASCII margin byte on each side, which non-ASCII prepend/combining
    /// characters can absorb into their cluster.
    pub(crate) fn normalize_into(&self, input: &str, out: &mut String) {
        use std::simd::prelude::*;

        if !self.fast_scan {
            out.push_str(&self.pre.normalize_string(input));
            return;
        }
        let bytes = input.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // SIMD hop to the next attention byte (non-ASCII, control, DEL).
            let mut j = i;
            'scan: {
                while j + 16 <= bytes.len() {
                    let v = u8x16::from_slice(&bytes[j..]);
                    let attention = v.simd_ge(u8x16::splat(0x7F)) | v.simd_lt(u8x16::splat(0x20));
                    let m = attention.to_bitmask();
                    if m != 0 {
                        j += m.trailing_zeros() as usize;
                        break 'scan;
                    }
                    j += 16;
                }
                while j < bytes.len() && (0x20..0x7F).contains(&bytes[j]) {
                    j += 1;
                }
            }

            if j >= bytes.len() {
                out.push_str(&input[i..]);
                return;
            }
            let b = bytes[j];
            if b < 0x80 {
                // Control or DEL: a standalone grapheme except CR before LF.
                out.push_str(&input[i..j]);
                if b == b'\r' && bytes.get(j + 1) == Some(&b'\n') {
                    match &self.crlf {
                        Some(norm) => out.push_str(norm),
                        None => out.push_str("\r\n"),
                    }
                    i = j + 2;
                } else {
                    match &self.ascii_map[b as usize] {
                        Some(norm) => out.push_str(norm),
                        None => out.push(b as char),
                    }
                    i = j + 1;
                }
                continue;
            }
            // Non-ASCII span: pull in one preceding printable-ASCII byte (a
            // combining mark would extend its cluster), then extend until a
            // printable-ASCII byte whose successor is also ASCII (or a
            // control, which always breaks).
            let span_start = if j > i { j - 1 } else { j };
            out.push_str(&input[i..span_start]);
            let mut k = j;
            let span_end = loop {
                if k >= bytes.len() {
                    break bytes.len();
                }
                let c = bytes[k];
                if c < 0x20 || c == 0x7F {
                    break k; // control: hard break before it
                }
                if c < 0x80 && bytes.get(k + 1).is_none_or(|&n| n < 0x80) {
                    break k + 1; // ASCII with ASCII successor: safe cut after
                }
                k += 1;
            };
            out.push_str(&self.pre.normalize_string(&input[span_start..span_end]));
            i = span_end;
        }
    }
}

/// Position of a section (the text between added-token matches) within its
/// document, which decides whether the raw fast path ▁-prefixes the
/// section's first unit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SectionPos {
    /// The very start of the document (Metaspace `first` prepends only here).
    First,
    /// After an added-token match; per-section prepends still apply.
    Middle,
    /// Continues a section begun in an earlier fragment of a split document
    /// (see [`SentencePieceBPE::safe_fragment_ranges`]): never prefixed —
    /// the section's prefix, if any, was emitted with its first unit.
    Continuation,
}

/// When the Metaspace pre-tokenizer prepends ▁ to a chunk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrependScheme {
    Never,
    Always,
    /// Only the chunk at the very start of the input; chunks after an added
    /// token don't count.
    First,
}

/// tokenizer.json `Metaspace` pre-tokenizer: replaces spaces with ▁, then
/// prepends ▁ per `prepend` (unless the chunk already starts with ▁), and —
/// with `split` — keeps BPE merges from crossing ▁ word boundaries.
pub struct Metaspace {
    pub prepend: PrependScheme,
    pub split: bool,
}

/// An added token, matched atomically in text before encoding.
pub struct AddedTokenSpec {
    /// What to find in the text. For `normalized` tokens this is the content
    /// after running it through the normalizer ops (HF matches those against
    /// normalized text).
    pub content: String,
    pub id: TokenId,
    /// Consume whitespace immediately before the match.
    pub lstrip: bool,
    /// Consume whitespace immediately after the match.
    pub rstrip: bool,
}

/// A tokenizer that mirrors SentencePiece BPE with `byte_fallback`. Holds
/// the immutable model; an [`Encoder`] adds the per-thread cache and scratch.
pub struct SentencePieceBPE {
    /// `ranked_merge_key(a, b) → (merged, rank)`.
    pub(crate) merges: RankedMerges,
    pub(crate) vocab: Vec<Arc<[u8]>>,
    pub(crate) vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher>,
    /// Token ID of each byte's `<0xHH>` fallback piece; `None` when the
    /// vocab lacks it (such bytes then only occur inside chars it covers).
    pub(crate) byte_fallback_ids: [Option<TokenId>; 256],
    /// Added tokens with `normalized: false`, matched in the raw input.
    pub(crate) added_tokens: Vec<AddedTokenSpec>,
    /// Added tokens with `normalized: true`, matched (pre-normalized)
    /// against normalizer output, before the Metaspace step.
    pub(crate) norm_added_tokens: Vec<AddedTokenSpec>,
    /// The tokenizer.json normalizer sequence, applied per chunk in order.
    pub(crate) norm_ops: Vec<NormOp>,
    /// The Metaspace pre-tokenizer, if any (`None`: `norm_ops` handles
    /// spaces and merges may cross word boundaries).
    pub(crate) metaspace: Option<Metaspace>,
    // Everything below is derived by `finalize_speed_paths`.
    /// Unit-splitting mode; see [`WordSplit`].
    pub(crate) word_split: WordSplit,
    /// `Some` when the normalizer pipeline reduces to optional ▁-prepend +
    /// space→▁, so raw text splits into units directly.
    pub(crate) raw_prepend: Option<RawPrepend>,
    /// Initial symbol(s) for the ▁ marker (vocab piece, or byte fallback).
    pub(crate) space_init: Vec<TokenId>,
    /// Initial symbol per ASCII char (vocab piece or byte fallback; `None`
    /// = no token).
    pub(crate) ascii_init: [Option<TokenId>; 128],
    /// Leftmost-longest automaton over `added_tokens` (pattern index == vec
    /// index).
    pub(crate) added_matcher: Option<aho_corasick::AhoCorasick>,
    /// Frequent ASCII punctuation to split units before (0 = unused slot),
    /// keeping word×punctuation combinations out of the cache.
    pub(crate) split_bytes: [u8; NUM_SPLIT_BYTES],
    /// Per split byte, a bitset over the previous byte for when the split
    /// is safe (no vocab piece contains the pair, so no merge spans it).
    pub(crate) split_safe: Vec<[u64; 4]>,
    /// Vocab pieces that can cross a `▁▁▁word` unit boundary, as `(pre,
    /// post)` around one interior ▁; see `piece_spans_boundary`.
    pub(crate) cross_pieces: Vec<(Box<[u8]>, Box<[u8]>)>,
    /// Bitset over the last byte of any `cross_pieces` pre, for a one-load
    /// rejection in the scanner.
    pub(crate) cross_prev: [u64; 4],
    /// Cache budget snapshot by each [`Self::encoder`]; `None` = unbounded.
    pub(crate) max_cache_bytes: Option<usize>,
}

/// How many distinct punctuation bytes the unit splitter checks for.
pub(crate) const NUM_SPLIT_BYTES: usize = 8;

impl std::fmt::Debug for SentencePieceBPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SentencePieceBPE {{ vocab_size: {}, merges_count: {} }}",
            self.vocab.len(),
            self.merges.len(),
        )
    }
}

impl SentencePieceBPE {
    /// Assemble a model from its loaded tables; `norm_added_tokens` are
    /// given raw and normalized here. Derived fast-path state is computed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        merges: RankedMerges,
        vocab: Vec<Arc<[u8]>>,
        vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher>,
        byte_fallback_ids: [Option<TokenId>; 256],
        added_tokens: Vec<AddedTokenSpec>,
        norm_added_tokens: Vec<AddedTokenSpec>,
        norm_ops: Vec<NormOp>,
        metaspace: Option<Metaspace>,
    ) -> Self {
        let mut model = Self {
            merges,
            vocab,
            vocab_inv,
            byte_fallback_ids,
            added_tokens,
            norm_added_tokens: Vec::new(),
            norm_ops,
            metaspace,
            word_split: WordSplit::None,
            raw_prepend: None,
            space_init: Vec::new(),
            ascii_init: [None; 128],
            added_matcher: None,
            split_bytes: [0; NUM_SPLIT_BYTES],
            split_safe: Vec::new(),
            cross_pieces: Vec::new(),
            cross_prev: [0; 4],
            max_cache_bytes: Some(crate::bpe::Tokenizer::DEFAULT_MAX_CACHE_BYTES),
        };
        model.norm_added_tokens = norm_added_tokens
            .into_iter()
            .map(|mut spec| {
                spec.content = model.apply_norm_ops(&spec.content).into_owned();
                spec
            })
            .collect();
        model.finalize_speed_paths();
        model
    }

    /// Apply the normalizer ops and Metaspace replacement/prepend to one text
    /// chunk. `first_chunk` is true only for the chunk at the very start of
    /// the input (the Metaspace "first" prepend scheme needs it).
    pub fn normalize<'a>(&self, input: &'a str, first_chunk: bool) -> Cow<'a, str> {
        self.apply_metaspace(self.apply_norm_ops(input), first_chunk)
    }

    /// The tokenizer.json normalizer sequence only (no Metaspace step).
    pub(crate) fn apply_norm_ops<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let mut s: Cow<'a, str> = Cow::Borrowed(input);
        for op in &self.norm_ops {
            match op {
                NormOp::Prepend(prefix) => {
                    if !s.is_empty() {
                        let mut out = String::with_capacity(prefix.len() + s.len());
                        out.push_str(prefix);
                        out.push_str(&s);
                        s = Cow::Owned(out);
                    }
                }
                NormOp::Replace { pattern, content } => {
                    if s.contains(pattern.as_str()) {
                        let replaced = s.replace(pattern.as_str(), content);
                        s = Cow::Owned(replaced);
                    }
                }
                NormOp::CollapseSpaces { content } => {
                    if s.contains("  ") {
                        s = Cow::Owned(collapse_space_runs(&s, content));
                    }
                }
                NormOp::Strip { left, right } => {
                    let trimmed = match (left, right) {
                        (true, true) => s.trim(),
                        (true, false) => s.trim_start(),
                        (false, true) => s.trim_end(),
                        (false, false) => &s,
                    };
                    if trimmed.len() != s.len() {
                        s = Cow::Owned(trimmed.to_string());
                    }
                }
                NormOp::Precompiled(charsmap) => {
                    let mut out = String::with_capacity(s.len() + 16);
                    charsmap.normalize_into(&s, &mut out);
                    s = Cow::Owned(out);
                }
            }
        }
        s
    }

    /// The Metaspace pre-tokenizer's space replacement and ▁ prepend.
    pub(crate) fn apply_metaspace<'a>(
        &self,
        input: Cow<'a, str>,
        first_chunk: bool,
    ) -> Cow<'a, str> {
        let mut s = input;
        if let Some(ms) = &self.metaspace {
            if s.contains(' ') {
                s = Cow::Owned(s.replace(' ', SENTENCEPIECE_SPACE_STR));
            }
            let prepend = match ms.prepend {
                PrependScheme::Always => true,
                PrependScheme::First => first_chunk,
                PrependScheme::Never => false,
            };
            if prepend && !s.is_empty() && !s.starts_with(SENTENCEPIECE_SPACE) {
                let mut out = String::with_capacity(SENTENCEPIECE_SPACE_STR.len() + s.len());
                out.push(SENTENCEPIECE_SPACE);
                out.push_str(&s);
                s = Cow::Owned(out);
            }
        }
        s
    }

    /// Whether encoding puts a ▁ in front of ordinary text — decode then
    /// strips the resulting leading space, like HF's decoder does.
    fn prepends_space(&self) -> bool {
        self.metaspace
            .as_ref()
            .is_some_and(|ms| ms.prepend != PrependScheme::Never)
            || self
                .norm_ops
                .iter()
                .any(|op| matches!(op, NormOp::Prepend(_)))
    }

    /// Compute the encode fast-path configuration (`word_split`,
    /// `raw_prepend`, `space_init`) from the assembled model. Must be called
    /// once by the loader after all other fields are final.
    pub(crate) fn finalize_speed_paths(&mut self) {
        self.space_init = match self.vocab_inv.get(SP_MARK.as_slice()) {
            Some(&id) => vec![id],
            None => SP_MARK
                .iter()
                .filter_map(|&b| self.byte_fallback_ids[b as usize])
                .collect(),
        };

        for b in 0u8..128 {
            self.ascii_init[b as usize] = self
                .vocab_inv
                .get([b].as_slice())
                .copied()
                .or(self.byte_fallback_ids[b as usize]);
        }

        self.added_matcher = (!self.added_tokens.is_empty()).then(|| {
            aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                // DFA: the scan visits every byte; large added vocabs need
                // O(1)/byte.
                .kind(Some(aho_corasick::AhoCorasickKind::DFA))
                .build(self.added_tokens.iter().map(|t| t.content.as_bytes()))
                .expect("added-token automaton")
        });

        self.cross_pieces = Vec::new();
        self.cross_prev = [0u64; 4];
        self.word_split = if self.metaspace.as_ref().is_some_and(|ms| ms.split) {
            WordSplit::EveryMark
        } else if let Some(cross) = self.unit_crossing_pieces() {
            for (pre, _) in &cross {
                let b = *pre.last().expect("interior ▁ always has a predecessor");
                self.cross_prev[(b >> 6) as usize] |= 1 << (b & 63);
            }
            self.cross_pieces = cross;
            WordSplit::SpaceRuns
        } else {
            WordSplit::None
        };

        self.raw_prepend = self.compute_raw_prepend();

        // Interior byte adjacency across all vocab pieces: splitting a unit
        // between bytes (x, y) is safe exactly when no piece contains x
        // immediately before y (a merge's result is always a vocab piece, so
        // no merge can then span the boundary; initial symbols are single
        // chars and cannot either).
        self.split_safe = vec![[0u64; 4]; 256];
        if self.word_split != WordSplit::None {
            let mut adjacent = vec![[0u64; 4]; 256]; // adjacent[x] bitset over y
            for piece in &self.vocab {
                for pair in piece.windows(2) {
                    adjacent[pair[0] as usize][(pair[1] >> 6) as usize] |= 1 << (pair[1] & 63);
                }
            }
            // The most frequent English punctuation, best value per SIMD
            // compare in the scanner.
            for (slot, &b) in [b'.', b',', b'"', b')', b';', b':', b'!', b'?']
                .iter()
                .enumerate()
            {
                let mut safe = [u64::MAX; 4];
                for x in 0..256usize {
                    if adjacent[x][(b >> 6) as usize] & (1 << (b & 63)) != 0 {
                        safe[x >> 6] &= !(1 << (x & 63));
                    }
                }
                // Only worth a compare if splitting is ever allowed. A byte
                // whose split is never safe keeps slot 0 (never matches).
                if safe != [0u64; 4] {
                    self.split_bytes[slot] = b;
                    self.split_safe[b as usize] = safe;
                }
            }
        }
    }

    /// Vocab pieces that can cross a `▁▁▁word` unit boundary (an interior ▁
    /// after a non-▁ char), as `(pre, post)`. `Some(vec![])`: units are
    /// unconditionally safe; `None`: a piece is too complex for the guard
    /// (a ▁ or raw space inside pre/post) or there are too many, so
    /// whole-chunk merging must stay.
    fn unit_crossing_pieces(&self) -> Option<Vec<(Box<[u8]>, Box<[u8]>)>> {
        // Beyond this the per-boundary guard stops being "a handful of
        // memcmps on a rare prev byte" and whole-chunk merging is safer.
        const MAX_CROSS_PIECES: usize = 32;
        let mut out: Vec<(Box<[u8]>, Box<[u8]>)> = Vec::new();
        for piece in &self.vocab {
            let Ok(s) = std::str::from_utf8(piece) else {
                // Byte-fallback pieces are single bytes; they can't hold a ▁.
                continue;
            };
            let mut prev_is_mark = true; // leading ▁s are fine
            for (pos, c) in s.char_indices() {
                if c == SENTENCEPIECE_SPACE {
                    if !prev_is_mark {
                        // Interior ▁: the piece spans from `pre`'s unit into
                        // the one starting at this mark.
                        let (pre, rest) = s.split_at(pos);
                        let post = &rest[SP_MARK.len()..];
                        if pre.contains(SENTENCEPIECE_SPACE)
                            || post.contains(SENTENCEPIECE_SPACE)
                            || pre.contains(' ')
                            || post.contains(' ')
                            || out.len() == MAX_CROSS_PIECES
                        {
                            return None;
                        }
                        out.push((pre.as_bytes().into(), post.as_bytes().into()));
                    }
                    prev_is_mark = true;
                } else {
                    prev_is_mark = false;
                }
            }
        }
        Some(out)
    }

    /// Does some crossing vocab piece occur spanning the candidate unit
    /// boundary at `mark_pos` (mark of `width` bytes)? Only such an
    /// occurrence can let a merge cross the boundary; everywhere else the
    /// split is exact.
    #[inline(always)]
    fn piece_spans_boundary(&self, bytes: &[u8], mark_pos: usize, width: usize) -> bool {
        let prev = bytes[mark_pos - 1];
        if self.cross_prev[(prev >> 6) as usize] & (1 << (prev & 63)) == 0 {
            return false;
        }
        let after = &bytes[mark_pos + width..];
        self.cross_pieces
            .iter()
            .any(|(pre, post)| bytes[..mark_pos].ends_with(pre) && after.starts_with(post))
    }

    /// Whether an oversized document can be split into independently
    /// encoded fragments (see [`Self::safe_fragment_ranges`]). Only the raw
    /// fast path encodes per unit, so only there is a unit boundary a safe
    /// cut.
    pub(crate) fn supports_fragment_split(&self) -> bool {
        self.raw_prepend.is_some()
    }

    /// Split raw text into ranges of roughly `target` bytes at cuts the raw
    /// scanner provably treats as unit boundaries, so encoding the first
    /// range with `encode_raw_cb` and the rest with
    /// `encode_raw_fragment_cb` equals one encode of `text`. Requires
    /// [`Self::supports_fragment_split`].
    ///
    /// A cut at `p` is safe when:
    /// - `p` starts a mark (raw space or complete ▁) that, under
    ///   `SpaceRuns`, neither extends a mark run nor has a crossing piece
    ///   spanning it (`piece_spans_boundary`); every other scanner decision
    ///   is local to one side.
    /// - no added-token occurrence blocks `p` (`added_token_cut_blocks`), so
    ///   leftmost-longest matching restarts cleanly and lstrip/rstrip
    ///   trimming stays within one fragment.
    ///
    /// Where no safe cut exists a range can exceed `target` (output stays
    /// exact, only parallelism degrades).
    pub(crate) fn safe_fragment_ranges(
        &self,
        text: &str,
        target: usize,
    ) -> Vec<std::ops::Range<usize>> {
        debug_assert!(self.supports_fragment_split());
        let bytes = text.as_bytes();
        let len = bytes.len();
        if len == 0 {
            // Zero ranges would drop the document's output row.
            return vec![0..0];
        }
        let blocks = self.added_token_cut_blocks(text);
        let mut bi = 0usize; // cursor into `blocks`; probes are monotone
        let every_mark = self.word_split == WordSplit::EveryMark;
        let target = target.max(1);
        let mut out = Vec::new();
        let mut start = 0usize;
        'chunks: while start < len {
            let mut probe = start + target;
            while probe < len {
                let Some(width) = mark_width(bytes, probe, true) else {
                    probe += 1;
                    continue;
                };
                let extends_run = !every_mark
                    && (bytes[probe - 1] == b' '
                        || (probe >= 3 && bytes[probe - 3..probe] == SP_MARK));
                if extends_run || self.piece_spans_boundary(bytes, probe, width) {
                    probe += 1;
                    continue;
                }
                while bi < blocks.len() && blocks[bi].end <= probe {
                    bi += 1;
                }
                if bi < blocks.len() && blocks[bi].start <= probe {
                    // Inside a blocked interval: hop past it (the merged
                    // intervals are disjoint and sorted).
                    probe = blocks[bi].end;
                    continue;
                }
                out.push(start..probe);
                start = probe;
                continue 'chunks;
            }
            out.push(start..len);
            break;
        }
        out
    }

    /// Sorted, disjoint byte intervals in which no fragment cut may land,
    /// from every added-token occurrence (overlapping ones included: a
    /// conservative superset of the matches an encode selects):
    /// - inside an occurrence `[s, e)`;
    /// - at `e` itself, under `Unguarded` prepend only (the continuation
    ///   section would go un-prefixed);
    /// - `lstrip`: back through the whitespace run before `s`;
    /// - `rstrip`: forward through the whitespace run after `e`.
    ///
    /// A cut only lands on a mark start, so a token without a space or ▁
    /// byte, without lstrip/rstrip, on a non-`Unguarded` model cannot block
    /// anything and is skipped without a scan. The scans run on the rayon
    /// pool; only the parallel entry points fragment documents.
    fn added_token_cut_blocks(&self, text: &str) -> Vec<std::ops::Range<usize>> {
        use rayon::prelude::*;
        let bytes = text.as_bytes();
        let unguarded = self.raw_prepend == Some(RawPrepend::Unguarded);
        let mut blocks: Vec<std::ops::Range<usize>> = self
            .added_tokens
            .par_iter()
            .filter(|spec| {
                let content = spec.content.as_bytes();
                let interior = content.contains(&b' ') || content.contains(&0xE2);
                !content.is_empty() && (interior || unguarded || spec.lstrip || spec.rstrip)
            })
            .flat_map_iter(|spec| {
                let content = spec.content.as_bytes();
                let finder = memchr::memmem::Finder::new(content);
                let mut out = Vec::new();
                let mut from = 0usize;
                while let Some(off) = finder.find(&bytes[from..]) {
                    let s = from + off;
                    let e = s + content.len();
                    // A valid-UTF-8 needle only matches at char boundaries
                    // (its first byte is a lead byte, and a full match ends
                    // a complete char), so slicing `text` at `s`/`e` cannot
                    // panic.
                    let lo = if spec.lstrip {
                        text[..s].trim_end().len()
                    } else {
                        s + 1
                    };
                    let hi = if spec.rstrip {
                        e + (text[e..].len() - text[e..].trim_start().len())
                    } else if unguarded {
                        e
                    } else {
                        e - 1 // only the interior (s, e) blocks; `e` is safe
                    };
                    if lo <= hi {
                        out.push(lo..hi + 1);
                    }
                    from = s + 1; // step one byte: overlapping occurrences too
                }
                out
            })
            .collect();
        blocks.sort_unstable_by_key(|r| r.start);
        blocks.dedup_by(|next, prev| {
            if next.start <= prev.end {
                prev.end = prev.end.max(next.end);
                true
            } else {
                false
            }
        });
        blocks
    }

    /// Raw-fast-path eligibility: the normalizer sequence must reduce to
    /// optional ▁ prepend + literal space→▁, with no normalized added tokens
    /// (those are matched against materialized normalizer output).
    fn compute_raw_prepend(&self) -> Option<RawPrepend> {
        if self.word_split == WordSplit::None || !self.norm_added_tokens.is_empty() {
            return None;
        }
        let is_space_replace = |op: &NormOp| {
            matches!(op, NormOp::Replace { pattern, content }
                if pattern == " " && content == SENTENCEPIECE_SPACE_STR)
        };
        let from_metaspace = match self.metaspace.as_ref().map(|ms| ms.prepend) {
            None | Some(PrependScheme::Never) => RawPrepend::Never,
            Some(PrependScheme::Always) => RawPrepend::GuardedAlways,
            Some(PrependScheme::First) => RawPrepend::GuardedFirst,
        };
        match self.norm_ops.as_slice() {
            [NormOp::Prepend(p), op] if p == SENTENCEPIECE_SPACE_STR && is_space_replace(op) => {
                // Under EveryMark the unguarded prepend would fuse a "▁" that
                // HF splits off the first unit; take the materialized path.
                (self.word_split != WordSplit::EveryMark).then_some(RawPrepend::Unguarded)
            }
            [op] if is_space_replace(op) => Some(from_metaspace),
            [] if self.metaspace.is_some() => Some(from_metaspace),
            _ => None,
        }
    }

    /// Create an encoder for this model.
    pub fn encoder(&self) -> Encoder<'_> {
        Encoder {
            model: self,
            state: EncodeState::with_budget(self.max_cache_bytes),
        }
    }

    /// SentencePiece analog of [`super::Tokenizer::set_max_cache_bytes`],
    /// same default: bounds the caches of [`EncodeState`]s created
    /// afterward, which wipe back to empty (there is no vocab seed here).
    pub fn set_max_cache_bytes(&mut self, budget: Option<usize>) {
        self.max_cache_bytes = budget;
    }

    /// The configured cache budget in bytes; `None` when unbounded.
    pub fn max_cache_bytes(&self) -> Option<usize> {
        self.max_cache_bytes
    }

    /// Size of the vocabulary: one greater than the largest token ID,
    /// including added tokens (IDs with no assigned content count too).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Vocabulary entries as `(id, bytes)` pairs in ID order, including
    /// added tokens and skipping IDs with no assigned content.
    pub fn vocab_entries(&self) -> impl Iterator<Item = (u32, &[u8])> {
        super::vocab_entries(&self.vocab)
    }

    /// Merge rules as `(left, right)` byte pairs in rank order.
    pub fn merge_entries(&self) -> Vec<(&[u8], &[u8])> {
        super::ranked_merge_entries(&self.merges, &self.vocab)
    }

    /// Convenience: encode a single text through a temporary encoder.
    pub fn encode_raw(&self, input: &str) -> Vec<TokenId> {
        let mut out = Vec::new();
        self.encoder()
            .encode_raw_cb(input, &mut |tokens| out.extend_from_slice(tokens));
        out
    }

    /// Decode token IDs back to a UTF-8 string.
    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        let mut raw = Vec::new();
        for &t in tokens {
            let idx: usize = t.into();
            if idx < self.vocab.len() {
                raw.extend_from_slice(&self.vocab[idx]);
            }
        }
        let text = String::from_utf8_lossy(&raw);
        let mut out: Vec<u8> = text.replace(SENTENCEPIECE_SPACE, " ").into_bytes();
        if self.prepends_space() && out.first() == Some(&b' ') {
            out.remove(0);
        }
        out
    }
}

/// Replace each run of 2 or more ASCII spaces with `content`, like HF's
/// `Replace(Regex(" {2,}"), content)`.
fn collapse_space_runs(input: &str, content: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            if j - i >= 2 {
                out.push_str(content);
            } else {
                out.push(' ');
            }
            i = j;
        } else {
            let start = i;
            while i < bytes.len() && bytes[i] != b' ' {
                i += 1;
            }
            out.push_str(&input[start..i]);
        }
    }
    out
}

/// log2 of the direct-mapped front-cache size (entries).
const FRONT_BITS: u32 = 20;

/// Fixed footprint of the front cache (keys + vals), subtracted from the
/// budget before bounding the growing caches.
const FRONT_BYTES: usize = (1 << FRONT_BITS) * (16 + 8);
/// Estimated bytes per map entry, payload + SwissTable slack (`long` adds
/// its key bytes on top). Length-based, because the wipe keeps allocations.
const MAP_ENTRY_BYTES: usize = 48;

/// Index of `key` in the front cache: multiplicative hash, top bits.
#[inline(always)]
fn front_index(key: u128) -> usize {
    let folded = (key as u64) ^ ((key >> 64) as u64);
    let h = folded.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> (64 - FRONT_BITS)) as usize
}

/// Per-thread mutable encoding context: the unit cache plus scratch
/// buffers. Units repeat heavily in natural text, so the ranked merge only
/// runs on cache misses.
pub struct EncodeState {
    /// Append-only arena of encoded token IDs; cache entries are
    /// `(offset, len)` slices into it.
    arena: Vec<TokenId>,
    /// Direct-mapped front cache in front of `short`, structure-of-arrays
    /// so a probe touches only the key array. Key 0 = empty.
    front_keys: Vec<u128>,
    front_vals: Vec<(u32, u32)>,
    /// Cache for units of ≤ 15 key bytes (the overwhelming majority), keyed
    /// by the same packed `u128` scheme as the byte-level path.
    short: HashMap<u128, (u32, u32), FxBuildHasher>,
    /// Fallback cache for longer units.
    long: HashMap<Box<[u8]>, (u32, u32), FxBuildHasher>,
    /// Scratch for the merge loop.
    symbols: Vec<TokenId>,
    /// Scratch for composing keys with a virtual leading space.
    key_buf: Vec<u8>,
    /// Byte budget for the growing caches ([`Self::with_budget`]);
    /// `usize::MAX` = unbounded.
    budget: usize,
    /// Estimated `long` bytes: key bytes + `MAP_ENTRY_BYTES` per entry.
    long_bytes_used: usize,
    /// Largest single encoding, as wipe-trigger slack: one recurring giant
    /// unit must not force a wipe per occurrence. Kept across wipes.
    max_encoding: usize,
}

impl EncodeState {
    /// A state whose growing caches (short/long maps + arena; the front
    /// cache is fixed-size) wipe back to empty when their estimated
    /// footprint exceeds `max_bytes` minus the front cache, floored at
    /// 1 MiB so tiny budgets stay functional.
    pub fn with_budget(max_bytes: Option<usize>) -> Self {
        EncodeState {
            arena: Vec::new(),
            front_keys: vec![0u128; 1 << FRONT_BITS],
            front_vals: vec![(0u32, 0u32); 1 << FRONT_BITS],
            short: HashMap::with_hasher(FxBuildHasher),
            long: HashMap::with_hasher(FxBuildHasher),
            symbols: Vec::new(),
            key_buf: Vec::new(),
            budget: max_bytes
                .map_or(usize::MAX, |t| t.saturating_sub(FRONT_BYTES).max(1 << 20)),
            long_bytes_used: 0,
            max_encoding: 0,
        }
    }

    /// Number of cached units (for diagnostics).
    pub fn cache_size(&self) -> usize {
        self.short.len() + self.long.len()
    }

    fn over_budget(&self) -> bool {
        self.short.len() * MAP_ENTRY_BYTES + self.long_bytes_used + self.arena.len() * 4
            > self.budget.saturating_add(self.max_encoding * 4)
    }

    /// Wipe every cache back to empty, keeping allocations. The front keys
    /// must go too: their values are slices of the truncated arena.
    #[cold]
    #[inline(never)]
    fn wipe(&mut self) {
        self.arena.clear();
        self.front_keys.fill(0);
        self.short.clear();
        self.long.clear();
        self.long_bytes_used = 0;
    }
}

impl Default for EncodeState {
    fn default() -> Self {
        Self::with_budget(Some(super::Tokenizer::DEFAULT_MAX_CACHE_BYTES))
    }
}

/// An encoder that holds a reference to the model plus its own cache and
/// scratch. Create one per thread for parallel encoding.
pub struct Encoder<'a> {
    model: &'a SentencePieceBPE,
    state: EncodeState,
}

/// Leftmost match of any spec's content in `text` (longest on ties, like
/// HF's LeftmostLongest matching), as `(position, spec index)`.
fn find_added_token(specs: &[AddedTokenSpec], text: &str) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (i, spec) in specs.iter().enumerate() {
        if let Some(pos) = text.find(spec.content.as_str()) {
            let better = match best {
                None => true,
                Some((best_pos, best_i)) => {
                    pos < best_pos
                        || (pos == best_pos && spec.content.len() > specs[best_i].content.len())
                }
            };
            if better {
                best = Some((pos, i));
            }
        }
    }
    best
}

/// One piece of an added-token split: text between matches, or a matched
/// token's ID.
enum Piece<'t> {
    Section(&'t str),
    Token(TokenId),
}

/// Split `text` around added-token `matches` (`(start, end, spec index)`,
/// leftmost-longest, in order) with lstrip/rstrip, like HF's
/// AddedVocabulary: a match starting inside whitespace a previous rstrip
/// consumed is dropped.
fn split_added<'t>(
    specs: &[AddedTokenSpec],
    text: &'t str,
    matches: impl Iterator<Item = (usize, usize, usize)>,
    mut emit: impl FnMut(Piece<'t>),
) {
    let mut chunk_start = 0usize;
    for (start, end, idx) in matches {
        if start < chunk_start {
            continue;
        }
        let spec = &specs[idx];
        let mut chunk = &text[chunk_start..start];
        if spec.lstrip {
            chunk = chunk.trim_end();
        }
        if !chunk.is_empty() {
            emit(Piece::Section(chunk));
        }
        emit(Piece::Token(spec.id));
        chunk_start = end;
        if spec.rstrip {
            chunk_start += text[end..].len() - text[end..].trim_start().len();
        }
    }
    let chunk = &text[chunk_start..];
    if !chunk.is_empty() {
        emit(Piece::Section(chunk));
    }
}

impl<'a> Encoder<'a> {
    /// Encode raw (un-normalized) text with added-token splitting, emitting
    /// token runs through `f`.
    pub fn encode_raw_cb<F: FnMut(&[TokenId])>(&mut self, input: &str, f: &mut F) {
        self.model.encode_raw_cb(&mut self.state, input, f);
    }

    /// Like [`Self::encode_raw_cb`] for one fragment of a document split at
    /// scanner-safe boundaries (see
    /// [`SentencePieceBPE::safe_fragment_ranges`]). A non-first fragment's
    /// leading section continues one begun in an earlier fragment, so it is
    /// never ▁-prefixed.
    pub fn encode_raw_fragment_cb<F: FnMut(&[TokenId])>(
        &mut self,
        input: &str,
        first: bool,
        f: &mut F,
    ) {
        let pos = if first {
            SectionPos::First
        } else {
            SectionPos::Continuation
        };
        self.model.encode_raw_from(&mut self.state, input, pos, f);
    }
}

/// Is the byte at `pos` the start of a unit mark? In raw mode both a space
/// and a literal ▁ count; in normalized mode only ▁ does. Returns the mark's
/// byte width.
#[inline(always)]
fn mark_width(bytes: &[u8], pos: usize, raw: bool) -> Option<usize> {
    match bytes[pos] {
        b' ' if raw => Some(1),
        0xE2 if bytes.len() - pos >= 3 && bytes[pos + 1] == 0x96 && bytes[pos + 2] == 0x81 => {
            Some(3)
        }
        _ => None,
    }
}

impl SentencePieceBPE {
    /// Encode raw (un-normalized) text with added-token splitting: first the
    /// raw-matched (`normalized: false`) tokens, then — per remaining section —
    /// the normalizer ops, the normalized-matched tokens, and Metaspace + BPE.
    /// Emits token runs through `f`, like the byte-level path's
    /// `memoized_encode`.
    pub fn encode_raw_cb<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        input: &str,
        f: &mut F,
    ) {
        self.encode_raw_from(state, input, SectionPos::First, f);
    }

    /// [`Self::encode_raw_cb`] with the position of the input's leading
    /// section given explicitly: `First` for a whole document,
    /// `Continuation` for a non-first fragment of one (see
    /// [`Self::safe_fragment_ranges`]). Sections after an added-token match
    /// are always `Middle`.
    fn encode_raw_from<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        input: &str,
        start: SectionPos,
        f: &mut F,
    ) {
        let Some(matcher) = &self.added_matcher else {
            self.encode_section_cb(state, input, start, f);
            return;
        };
        let mut pos = start;
        let matches = matcher
            .find_iter(input.as_bytes())
            .map(|m| (m.start(), m.end(), m.pattern().as_usize()));
        split_added(&self.added_tokens, input, matches, |piece| match piece {
            Piece::Section(chunk) => self.encode_section_cb(state, chunk, pos, f),
            Piece::Token(id) => {
                f(&[id]);
                pos = SectionPos::Middle;
            }
        });
    }

    /// Encode one raw section: the raw fast path when eligible, otherwise
    /// normalizer ops → normalized added-token splitting → Metaspace → BPE.
    /// HF does not re-normalize the parts around a normalized added-token
    /// match.
    fn encode_section_cb<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        text: &str,
        pos: SectionPos,
        f: &mut F,
    ) {
        if let Some(prepend) = self.raw_prepend {
            self.encode_chunk_raw(state, text, pos, prepend, f);
            return;
        }

        // Fragment splitting is gated on the raw fast path
        // (`supports_fragment_split`), so a continuation never reaches the
        // materialized-normalizer path below — its whole-section ops
        // (Strip, per-section Prepend, ...) have no continuation form.
        debug_assert!(pos != SectionPos::Continuation);
        let first_chunk = pos == SectionPos::First;
        let normed = self.apply_norm_ops(text);

        if self.norm_added_tokens.is_empty() {
            let final_text = self.apply_metaspace(normed, first_chunk);
            self.encode_normalized_cb(state, &final_text, f);
            return;
        }

        let specs = &self.norm_added_tokens;
        let mut first = first_chunk;
        let mut from = 0usize;
        let matches = std::iter::from_fn(|| {
            let (pos, idx) = find_added_token(specs, &normed[from..])?;
            let start = from + pos;
            from = start + specs[idx].content.len();
            Some((start, from, idx))
        });
        split_added(specs, &normed, matches, |piece| match piece {
            Piece::Section(part) => {
                let final_text = self.apply_metaspace(Cow::Borrowed(part), first);
                self.encode_normalized_cb(state, &final_text, f);
            }
            Piece::Token(id) => {
                f(&[id]);
                first = false;
            }
        });
    }

    /// Raw fast path: split un-normalized text into units directly, mapping
    /// spaces to ▁ on the fly. The dummy-prefix ▁ becomes a virtual leading
    /// space on the first unit, which keys and encodes identically.
    fn encode_chunk_raw<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        chunk: &str,
        pos: SectionPos,
        prepend: RawPrepend,
        f: &mut F,
    ) {
        if chunk.is_empty() {
            return;
        }
        let bytes = chunk.as_bytes();
        let starts_with_mark = mark_width(bytes, 0, true).is_some();
        let virtual_prefix = match prepend {
            // A continuation resumes a section mid-way: its prefix, if any,
            // was emitted with the section's first unit in an earlier
            // fragment.
            _ if pos == SectionPos::Continuation => false,
            RawPrepend::Unguarded => true,
            RawPrepend::GuardedAlways => !starts_with_mark,
            RawPrepend::GuardedFirst => pos == SectionPos::First && !starts_with_mark,
            RawPrepend::Never => false,
        };
        self.encode_units::<true, F>(state, bytes, virtual_prefix, f);
    }

    /// Encode already-normalized text: unit split (per `word_split`) with the
    /// pretoken cache, or a whole-chunk merge when units aren't safe.
    pub fn encode_normalized_cb<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        input: &str,
        f: &mut F,
    ) {
        match self.word_split {
            WordSplit::None => self.bpe_chunk(state, input, f),
            _ => self.encode_units::<false, F>(state, input.as_bytes(), false, f),
        }
    }

    /// Split a chunk into word units and encode each through the cache,
    /// walking 32-byte SIMD blocks whose mark-candidate bitmask is drained
    /// bit by bit. `RAW` selects raw-mode marks (space or ▁) vs ▁ only;
    /// `virtual_prefix` logically prepends one space to the first unit.
    fn encode_units<const RAW: bool, F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        virtual_prefix: bool,
        f: &mut F,
    ) {
        use std::simd::prelude::*;

        let every_mark = self.word_split == WordSplit::EveryMark;
        let mut unit_start = 0usize;
        let mut last_mark_end = usize::MAX;
        let mut first_unit = true;
        // Close the current unit at `end` and start the next one there.
        macro_rules! close_unit {
            ($end:expr) => {{
                let end = $end;
                self.encode_unit(
                    state,
                    &bytes[unit_start..end],
                    virtual_prefix && first_unit,
                    RAW,
                    f,
                );
                first_unit = false;
                unit_start = end;
            }};
        }

        let splats = self.split_bytes.map(u8x16::splat);

        let mut block = 0usize;
        while block < bytes.len() {
            let mut mask: u32;
            if block + 32 <= bytes.len() {
                let lo = u8x16::from_slice(&bytes[block..]);
                let hi = u8x16::from_slice(&bytes[block + 16..]);
                let mut m_lo = lo.simd_eq(u8x16::splat(0xE2));
                let mut m_hi = hi.simd_eq(u8x16::splat(0xE2));
                if RAW {
                    m_lo |= lo.simd_eq(u8x16::splat(b' '));
                    m_hi |= hi.simd_eq(u8x16::splat(b' '));
                }
                // Split-punct candidates (unused slots are 0x00 splats; NUL
                // bytes then take the punct path and split_safe[0] is empty).
                for splat in splats {
                    m_lo |= lo.simd_eq(splat);
                    m_hi |= hi.simd_eq(splat);
                }
                mask = (m_lo.to_bitmask() as u32) | ((m_hi.to_bitmask() as u32) << 16);
            } else {
                mask = 0;
                for (i, &b) in bytes[block..].iter().enumerate() {
                    if b == 0xE2 || (RAW && b == b' ') || self.split_bytes.contains(&b) {
                        mask |= 1 << i;
                    }
                }
            }

            while mask != 0 {
                let mark_pos = block + mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let byte = bytes[mark_pos];
                if (RAW && byte == b' ') || byte == 0xE2 {
                    // 0xE2 also starts other three-byte chars.
                    let Some(width) = mark_width(bytes, mark_pos, RAW) else {
                        continue;
                    };
                    // SpaceRuns: a mark that extends a run, or one a crossing
                    // vocab piece spans, is not a boundary.
                    let boundary = every_mark || mark_pos != last_mark_end;
                    if boundary
                        && mark_pos != unit_start
                        && !self.piece_spans_boundary(bytes, mark_pos, width)
                    {
                        close_unit!(mark_pos);
                    }
                    last_mark_end = mark_pos + width;
                } else if mark_pos != unit_start {
                    // Split punctuation: a unit boundary only after a
                    // vocab-verified safe predecessor. A raw space acts as ▁,
                    // so check its final UTF-8 byte.
                    let mut prev = bytes[mark_pos - 1];
                    if RAW && prev == b' ' {
                        prev = SP_MARK[2];
                    }
                    let safe = &self.split_safe[byte as usize];
                    if safe[(prev >> 6) as usize] & (1 << (prev & 63)) != 0 {
                        close_unit!(mark_pos);
                    }
                }
            }
            block += 32;
        }
        // `unit_start` only ever advances to a mark position < len, so a
        // non-empty chunk always has a final unit.
        if unit_start < bytes.len() {
            close_unit!(bytes.len());
        }
    }

    /// Encode one unit through the pretoken cache; run the ranked merge only
    /// on a miss. The cache key is the unit's bytes (raw or normalized —
    /// byte-equal keys always encode identically), with a `b' '` prefix for
    /// the virtual leading space.
    #[inline]
    fn encode_unit<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        unit: &[u8],
        virtual_prefix: bool,
        raw: bool,
        f: &mut F,
    ) {
        let packed = if virtual_prefix {
            state.key_buf.clear();
            state.key_buf.push(b' ');
            state.key_buf.extend_from_slice(unit);
            pack_pretoken_key(&state.key_buf)
        } else {
            pack_pretoken_key(unit)
        };
        // Front cache first: one L1/L2 compare resolves Zipf-hot units.
        let mut front_idx = 0;
        if let Some(key) = packed {
            front_idx = front_index(key);
            // SAFETY: the front arrays have 2^FRONT_BITS entries and
            // `front_index` returns FRONT_BITS bits.
            if unsafe { *state.front_keys.get_unchecked(front_idx) } == key {
                let (offset, len) = unsafe { *state.front_vals.get_unchecked(front_idx) };
                let start = offset as usize;
                // SAFETY: entries are recorded right after appending `len`
                // tokens at `offset`; the arena only clears in a budget
                // wipe, which also zeroes every front key.
                f(unsafe { state.arena.get_unchecked(start..start + len as usize) });
                return;
            }
        }
        let cached = match packed {
            Some(key) => state.short.get(&key).copied(),
            None if virtual_prefix => state.long.get(state.key_buf.as_slice()).copied(),
            None => state.long.get(unit).copied(),
        };
        if let Some((offset, len)) = cached {
            if let Some(key) = packed {
                state.front_keys[front_idx] = key;
                state.front_vals[front_idx] = (offset, len);
            }
            let start = offset as usize;
            // SAFETY: as above (a wipe also clears both maps).
            f(unsafe { state.arena.get_unchecked(start..start + len as usize) });
            return;
        }

        // Miss: budget check first (the lookups above already missed, and
        // stay misses against the freshly wiped caches), then character
        // init → ranked merge, then record in the arena.
        if state.over_budget() {
            state.wipe();
        }
        state.symbols.clear();
        if virtual_prefix {
            state.symbols.extend_from_slice(&self.space_init);
        }
        // SAFETY: units start at mark starts and end at mark starts or the
        // chunk end, all of which are char boundaries of the original &str.
        let unit_str = unsafe { std::str::from_utf8_unchecked(unit) };
        self.init_symbols(unit_str, raw, &mut state.symbols);
        bpe_merge_symbols_ranked(&self.merges, &mut state.symbols);

        let offset = state.arena.len() as u32;
        let len = state.symbols.len() as u32;
        state.arena.extend_from_slice(&state.symbols);
        state.max_encoding = state.max_encoding.max(len as usize);
        match packed {
            Some(key) => {
                state.short.insert(key, (offset, len));
                state.front_keys[front_idx] = key;
                state.front_vals[front_idx] = (offset, len);
            }
            None => {
                let key: Box<[u8]> = if virtual_prefix {
                    state.key_buf.as_slice().into()
                } else {
                    unit.into()
                };
                state.long_bytes_used += key.len() + MAP_ENTRY_BYTES;
                state.long.insert(key, (offset, len));
            }
        }
        f(&state.symbols);
    }

    /// Whole-chunk merge without caching (vocabs with boundary-crossing
    /// pieces).
    fn bpe_chunk<F: FnMut(&[TokenId])>(&self, state: &mut EncodeState, chunk: &str, f: &mut F) {
        state.symbols.clear();
        self.init_symbols(chunk, false, &mut state.symbols);
        bpe_merge_symbols_ranked(&self.merges, &mut state.symbols);
        f(&state.symbols);
    }

    /// Character init: each char's vocab piece, or its UTF-8 bytes through
    /// byte fallback. In raw mode spaces initialize as the ▁ marker.
    #[inline]
    fn init_symbols(&self, text: &str, raw: bool, symbols: &mut Vec<TokenId>) {
        for ch in text.chars() {
            if raw && ch == ' ' {
                symbols.extend_from_slice(&self.space_init);
                continue;
            }
            if (ch as u32) < 128 {
                if let Some(id) = self.ascii_init[ch as usize] {
                    symbols.push(id);
                }
                continue;
            }
            let mut buf = [0u8; 4];
            let ch_bytes = ch.encode_utf8(&mut buf).as_bytes();
            if let Some(&id) = self.vocab_inv.get(ch_bytes) {
                symbols.push(id);
            } else {
                for &b in ch_bytes {
                    if let Some(id) = self.byte_fallback_ids[b as usize] {
                        symbols.push(id);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_space_runs() {
        assert_eq!(collapse_space_runs("a  b   c d  ", "▁"), "a▁b▁c d▁");
        assert_eq!(collapse_space_runs("a \t  b", "▁"), "a \t▁b");
        assert_eq!(collapse_space_runs("  ", "▁"), "▁");
        assert_eq!(collapse_space_runs("no runs", "▁"), "no runs");
    }

    /// PARITY + BOUNDS: a budgeted EncodeState that wipes mid-corpus must
    /// produce the exact token stream of an unbounded one, while the
    /// estimated footprint of the growing caches stays within the budget
    /// (plus the documented giant-encoding slack).
    #[test]
    fn budgeted_wipe_matches_unbounded() {
        let Some(path) = crate::test_hub::hf_tokenizer_json("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else {
            eprintln!("Skipping: TinyLlama tokenizer.json not in the HF cache");
            return;
        };
        let model = crate::load_tokenizer::hf::load_hf_sentencepiece(&path).unwrap();
        assert_eq!(
            model.max_cache_bytes(),
            Some(super::super::Tokenizer::DEFAULT_MAX_CACHE_BYTES)
        );

        // ~100k distinct short words plus interleaved > 15-byte words (the
        // long-map path) and repeats of a common head.
        let mut text = String::new();
        let mut x = 0x1234_5678_9ABC_DEF0u64;
        let mut rand = |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        for i in 0..100_000u32 {
            let len = if i % 17 == 0 { 16 + rand(24) } else { 3 + rand(8) };
            for _ in 0..len {
                text.push((b'a' + rand(26) as u8) as char);
            }
            text.push(' ');
            if i % 5 == 0 {
                text.push_str("the quick brown fox ");
            }
        }

        let mut unbounded = EncodeState::with_budget(None);
        let mut expected = Vec::new();
        model.encode_raw_cb(&mut unbounded, &text, &mut |t| expected.extend_from_slice(t));

        // 25 MiB total = ~1 MiB effective past the 24 MiB front cache:
        // several wipes over ~100k distinct units.
        let mut budgeted = EncodeState::with_budget(Some(25 << 20));
        let mut actual = Vec::new();
        model.encode_raw_cb(&mut budgeted, &text, &mut |t| actual.extend_from_slice(t));
        assert_eq!(actual, expected, "budgeted output diverged");
        assert!(budgeted.long_bytes_used > 0, "corpus never hit the long map");
        // Both saw ~100k distinct units, so a small survivor count proves
        // wipes happened, and the estimate must end up within budget.
        assert!(
            budgeted.cache_size() < unbounded.cache_size() / 3,
            "survivors {} vs unbounded {}",
            budgeted.cache_size(),
            unbounded.cache_size()
        );
        let used = budgeted.short.len() * MAP_ENTRY_BYTES
            + budgeted.long_bytes_used
            + budgeted.arena.len() * 4;
        let limit = budgeted.budget + budgeted.max_encoding * 4;
        assert!(used <= limit + 4096, "estimate {used} exceeds budget {limit}");
    }
}
