//! Fast pretokenizers, one submodule per scheme, sharing the byte
//! predicates, UTF-8 decode, and run scans below.

pub(crate) mod cl100k_family;
pub(crate) mod mask;
pub(crate) mod o200k_family;

pub mod cl100k;
pub mod deepseek_v3;
pub mod kimi;
pub mod nemotron;
pub mod o200k;
pub mod olmo3;
pub mod qwen2;
pub mod qwen3_5;
pub mod r50k;

pub use cl100k::FastCl100kPretokenizer;
pub use deepseek_v3::FastDeepSeekV3Pretokenizer;
pub use kimi::FastKimiPretokenizer;
pub use nemotron::FastNemotronPretokenizer;
pub use o200k::FastO200kPretokenizer;
pub use olmo3::FastOlmo3Pretokenizer;
pub use qwen2::FastQwen2Pretokenizer;
pub use qwen3_5::FastQwen35Pretokenizer;
pub use r50k::FastR50kPretokenizer;

use crate::pretokenize::SpanBatch;
use crate::pretokenize::unicode;

// Shared chunked span pull for the mask-scanner pretokenizers

/// `PretokenSpans::fill_spans_keyed` for every `(bytes, MaskState)`
/// pretokenizer: the two-phase chunk walker when a SIMD scanner is
/// available, otherwise one fused pull loop over `next_span`.
/// `#[inline(never)]` keeps each monomorphization's register allocation
/// away from the encode loop that calls it (measured, do not fold).
#[inline(never)]
pub(crate) fn fill_spans_keyed_mask<'a, S: mask::MaskScheme>(
    bytes: &'a [u8],
    state: &mut mask::MaskState,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if mask::simd_scanner_available() {
        return state.fill_spans_two_phase::<S>(bytes, batch, prefetch);
    }
    crate::pretokenize::fill_spans_keyed_with_buf(
        bytes,
        // next_span returns in-bounds, nonempty span boundaries.
        || state.next_span::<S>(bytes),
        batch,
        prefetch,
    )
}

/// Define a mask-scanner pretokenizer: a `{ bytes, state: MaskState }`
/// struct driving `$scheme` through the shared walker, with its
/// constructors, `Iterator`, and `PretokenSpans` implementations.
macro_rules! define_mask_pretokenizer {
    ($pretokenizer:ident, $scheme:ty) => {
        #[doc = concat!(
            "Pretokenizer for `", stringify!($scheme),
            "`: the mask scanner where SIMD is available, the scheme's scalar advance elsewhere."
        )]
        pub struct $pretokenizer<'a> {
            bytes: &'a [u8],
            state: crate::pretokenize::fast::mask::MaskState,
        }

        impl<'a> $pretokenizer<'a> {
            #[inline]
            pub fn new(bytes: &'a [u8]) -> Self {
                Self::with_pos(bytes, 0)
            }

            /// Resume iteration at a byte offset previously returned by [`Self::pos`].
            #[inline]
            pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
                Self { bytes, state: crate::pretokenize::fast::mask::MaskState::new(pos) }
            }

            /// Current position as a byte offset into the input.
            #[inline]
            pub fn pos(&self) -> usize {
                self.state.pos
            }
        }

        impl<'a> Iterator for $pretokenizer<'a> {
            type Item = crate::pretokenize::Pretoken<'a>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let (start, end) = self.state.next_span::<$scheme>(self.bytes)?;
                Some(crate::pretokenize::Pretoken(&self.bytes[start..end]))
            }
        }

        // SAFETY: delegates to `fill_spans_keyed_mask`, whose bodies write
        // exactly the first `n` entries from live spans of `self.bytes`.
        unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for $pretokenizer<'a> {
            #[inline]
            fn fill_spans_keyed(
                &mut self,
                batch: &mut crate::pretokenize::SpanBatch<'a>,
                prefetch: &impl Fn(u64),
            ) -> usize {
                crate::pretokenize::fast::fill_spans_keyed_mask::<$scheme>(
                    self.bytes,
                    &mut self.state,
                    batch,
                    prefetch,
                )
            }
        }
    };
}
pub(crate) use define_mask_pretokenizer;

// Branchless byte predicates

#[inline(always)]
pub(crate) fn is_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
pub(crate) fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline(always)]
pub(crate) fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

/// Decode one non-ASCII scalar at `pos` (`pos < bytes.len()`,
/// `bytes[pos] >= 0x80`) as `(codepoint, byte length)`. Invalid input
/// decodes deterministically: the read never passes `len`, a sequence
/// truncated by the buffer end consumes the remainder as [`CP_INVALID`],
/// and the result is always `<= 0x10FFFF` so packed-table lookups stay in
/// bounds.
#[inline(always)]
pub(crate) unsafe fn decode_cp(bytes: &[u8], pos: usize) -> (u32, usize) {
    if pos + 4 > bytes.len() {
        // Within 3 bytes of the buffer end: the only region where a
        // sequence can be truncated. Cold: interior calls (the hot ones)
        // never take it, and the branch predicts not-taken.
        return decode_cp_near_end(bytes, pos);
    }
    // SAFETY: pos + 4 <= len just checked.
    unsafe { decode_cp_inbounds(bytes, pos) }
}

/// [`decode_cp`] without the buffer-end guard: the caller must guarantee
/// `pos + 4 <= bytes.len()` (the batch classifiers' `scan + 70 <= len`
/// guard covers every call site).
#[inline(always)]
pub(crate) unsafe fn decode_cp_inbounds(bytes: &[u8], pos: usize) -> (u32, usize) {
    unsafe {
        let b0 = *bytes.get_unchecked(pos) as u32;
        let b1 = (*bytes.get_unchecked(pos + 1) & 0x3F) as u32;
        if b0 < 0xE0 {
            return (((b0 & 0x1F) << 6) | b1, 2);
        }
        let b2 = (*bytes.get_unchecked(pos + 2) & 0x3F) as u32;
        if b0 < 0xF0 {
            return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
        }
        let b3 = (*bytes.get_unchecked(pos + 3) & 0x3F) as u32;
        (
            (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3).min(CP_INVALID),
            4,
        )
    }
}

/// U+10FFFF, reported for truncated tails and beyond-Unicode garbage:
/// unassigned, so class `Other` in every table.
pub(crate) const CP_INVALID: u32 = 0x10FFFF;

/// [`decode_cp`]'s bounds-checked slow path for `pos + 4 > bytes.len()`.
#[cold]
#[inline(never)]
fn decode_cp_near_end(bytes: &[u8], pos: usize) -> (u32, usize) {
    let len = bytes.len();
    let b0 = bytes[pos] as u32;
    let need = if b0 < 0xE0 {
        2
    } else if b0 < 0xF0 {
        3
    } else {
        4
    };
    if pos + need > len {
        // Truncated tail: consume the rest of the buffer as one
        // unclassifiable char so every walker path terminates the final
        // pretoken at `len` the same way.
        return (CP_INVALID, len - pos);
    }
    let b1 = (bytes[pos + 1] & 0x3F) as u32;
    if need == 2 {
        return (((b0 & 0x1F) << 6) | b1, 2);
    }
    let b2 = (bytes[pos + 2] & 0x3F) as u32;
    if need == 3 {
        return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
    }
    let b3 = (bytes[pos + 3] & 0x3F) as u32;
    (
        (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3).min(CP_INVALID),
        4,
    )
}

/// `[\r\n]*`: advance past a run of CR/LF bytes (trailing newlines after a
/// punctuation run in the cl100k-family schemes).
#[inline(always)]
pub(crate) fn scan_newlines(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if b == b'\r' || b == b'\n' {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// If the char at `pos` is a letter (`\p{L}` under the 4-way `CharClass`
/// classifier), return the offset just past it.
#[inline(always)]
pub(crate) fn letter_end_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some(pos + 1);
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        if unicode::class_of(cp) == unicode::CharClass::Letter {
            return Some(pos + l);
        }
    }
    None
}

// SWAR

pub(crate) const HI: u64 = 0x8080_8080_8080_8080;

/// Returns the high bit set in each lane that is NOT an ASCII letter,
/// computed directly (rather than as the complement of a letter mask) so
/// the scan loop can branch on `!= 0` and reuse the value for `trailing_zeros`.
#[inline(always)]
pub(crate) fn swar64_letter_nonmask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lowered);
    !(ge_a & le_z) & HI
}

/// SWAR letter scan: advances `pos` past ASCII letters.
/// Returns the updated pos.
#[inline(always)]
pub(crate) fn swar_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    // SWAR: 8 bytes at a time
    while pos + 8 <= len {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HI != 0 {
            break;
        }
        let nonletter = swar64_letter_nonmask(word);
        if nonletter != 0 {
            return pos + nonletter.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    // Scalar tail
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_letter(b) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// Shared run scans (`\p{L}+`, `\p{N}+`, `\p{N}{1,3}`, `[^\s\p{L}\p{N}]+`, `\s...`)

/// `\p{N}{1,3}`: extend a number run that already matched `consumed` chars
/// to at most 3 chars total.
#[inline(always)]
pub(crate) fn scan_numbers_max3(bytes: &[u8], mut pos: usize, mut consumed: u32) -> usize {
    let len = bytes.len();
    while consumed < 3 && pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_digit(b) {
            pos += 1;
            consumed += 1;
            continue;
        }
        if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, pos) };
            if unicode::class_of(cp) == unicode::CharClass::Number {
                pos += l;
                consumed += 1;
                continue;
            }
        }
        break;
    }
    pos
}

/// `\p{N}{1,3}` or `\p{N}` starting at a number char ending at `first_end`.
#[inline(always)]
pub(crate) fn digit_token_end<const DIGITS3: bool>(bytes: &[u8], first_end: usize) -> usize {
    if DIGITS3 {
        scan_numbers_max3(bytes, first_end, 1)
    } else {
        first_end
    }
}

#[inline(always)]
pub(crate) fn scan_letters_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Letter {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_digits_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len && is_digit(unsafe { *bytes.get_unchecked(p) }) {
            p += 1;
        }
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Number {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 { break; }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) { return p; }
            p += 1;
        }
        if p < len {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Other {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// Whitespace-led token starting at `start`: `\s*[\r\n]+` | `\s+(?!\S)` |
/// `\s+`, in that priority; with `EOS_WS_WHOLE` (cl100k's `\s++$`)
/// trailing whitespace stays one token even when it contains a newline.
/// `is_ws` classifies non-ASCII codepoints. Precondition: the letter-prefix
/// and space+punct alternatives were ruled out.
#[inline(always)]
pub(crate) fn ws_token_end<const EOS_WS_WHOLE: bool>(
    bytes: &[u8],
    start: usize,
    is_ws: impl Fn(u32) -> bool,
) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl_end = 0usize; // 0 = run contains no \r\n
    let mut last_char_start = start;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if b == b'\r' || b == b'\n' {
            last_char_start = p;
            p += 1;
            last_nl_end = p;
        } else if is_ascii_ws(b) {
            last_char_start = p;
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if is_ws(cp) {
                last_char_start = p;
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if EOS_WS_WHOLE && p >= len {
        return p; // `\s++$`: trailing whitespace is one token
    }
    if last_nl_end != 0 {
        return last_nl_end; // through the last newline
    }
    if p >= len {
        return p; // `\s+(?!\S)`: lookahead succeeds at EOS
    }
    if last_char_start > start {
        return last_char_start; // `\s+(?!\S)`: all but the last ws char
    }
    p // `\s+`: single whitespace char before content
}

/// Helpers shared by the scheme tests: OWT loading, the fancy_regex
/// oracle, and scalar-vs-mask differential drivers.
#[cfg(test)]
pub(crate) mod test_support {
    use super::mask::{MaskScheme, MaskState};
    use crate::pretokenize::Pretoken;
    use itertools::Itertools;
    use std::io::Read;

    /// The first `max_bytes` of ~/data/owt_train.txt, truncated to a
    /// UTF-8 boundary (streamed; the full file is ~12 GB).
    pub(crate) fn load_owt_prefix(max_bytes: usize) -> Vec<u8> {
        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let f = std::fs::File::open(&path).expect("Could not open ~/data/owt_train.txt");
        let mut buf = Vec::new();
        f.take(max_bytes as u64).read_to_end(&mut buf).unwrap();
        while !buf.is_empty() && std::str::from_utf8(&buf).is_err() {
            buf.pop();
        }
        buf
    }

    pub(crate) fn regex_tokens(pattern: &str, s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(pattern).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    pub(crate) fn fast_tokens<'a>(it: impl Iterator<Item = Pretoken<'a>>) -> Vec<String> {
        it.map(|t| String::from_utf8_lossy(t.0).into_owned()).collect()
    }

    /// `fast(case)` must equal `reference(case)` for every case.
    pub(crate) fn assert_cases(
        cases: &[&str],
        fast: impl Fn(&str) -> Vec<String>,
        reference: impl Fn(&str) -> Vec<String>,
    ) {
        for case in cases {
            assert_eq!(fast(case), reference(case), "Mismatch on case {case:?}");
        }
    }

    /// Random codepoint soup drawn from `pools`, `fast` against `reference`.
    pub(crate) fn assert_random_soup(
        pools: &[&[char]],
        seed: u64,
        rounds: usize,
        fast: impl Fn(&str) -> Vec<String>,
        reference: impl Fn(&str) -> Vec<String>,
    ) {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        for round in 0..rounds {
            let len = rng.random_range(1..40);
            let s: String = (0..len)
                .map(|_| {
                    let pool = pools.choose(&mut rng).unwrap();
                    *pool.choose(&mut rng).unwrap()
                })
                .collect();
            assert_eq!(fast(&s), reference(&s), "Mismatch on round {round}, case {s:?}");
        }
    }

    /// Token-by-token comparison of `fast` against the reference regex on
    /// `text`, streaming (no full token lists held in memory).
    pub(crate) fn assert_matches_regex<'a>(
        pattern: &str,
        text: &'a str,
        fast: impl Iterator<Item = Pretoken<'a>>,
    ) {
        let re = fancy_regex::Regex::new(pattern).unwrap();
        let fast = fast.map(|t| String::from_utf8_lossy(t.0).into_owned());
        let reference = re
            .find_iter(text)
            .map(|m| m.expect("regex match error").as_str().to_owned());
        let mut recent: std::collections::VecDeque<(String, String)> = Default::default();
        let mut n = 0usize;
        for (i, pair) in fast.zip_longest(reference).enumerate() {
            let (f, r) = pair.map_any(Some, Some).or_default();
            assert_eq!(f, r, "Mismatch at token {i}; recent tokens: {recent:?}");
            if recent.len() == 10 {
                recent.pop_front();
            }
            recent.push_back((f.unwrap(), r.unwrap()));
            n = i + 1;
        }
        eprintln!("All {n} tokens match.");
    }

    pub(crate) fn scalar_tokens<S: MaskScheme>(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut pos = 0;
        let mut out = vec![];
        while pos < bytes.len() {
            let e = S::advance(bytes, pos);
            out.push(bytes[pos..e].to_vec());
            pos = e;
        }
        out
    }

    pub(crate) fn mask_tokens<S: MaskScheme>(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut st = MaskState::new(0);
        let mut out = vec![];
        while let Some((s, e)) = st.next_span::<S>(bytes) {
            out.push(bytes[s..e].to_vec());
        }
        out
    }

    /// The mask scanner must reproduce the scheme's scalar tokenization.
    #[track_caller]
    pub(crate) fn check_scalar_vs_mask<S: MaskScheme>(buf: &[u8], scheme: &str) {
        let a = scalar_tokens::<S>(buf);
        let b = mask_tokens::<S>(buf);
        if a != b {
            let i = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
            panic!(
                "{scheme} diverged at token {i} on {:?}\n  scalar: {:?}\n  mask:   {:?}",
                String::from_utf8_lossy(buf),
                a.get(i).map(|t| String::from_utf8_lossy(t).into_owned()),
                b.get(i).map(|t| String::from_utf8_lossy(t).into_owned()),
            );
        }
    }

    /// Streaming scalar-vs-mask check for large inputs.
    pub(crate) fn check_streaming<S: MaskScheme>(bytes: &[u8], scheme: &str) {
        let mut st = MaskState::new(0);
        let mut pos = 0usize;
        let mut idx = 0usize;
        while pos < bytes.len() {
            let scalar_end = S::advance(bytes, pos);
            match st.next_span::<S>(bytes) {
                Some((s, e)) => assert!(
                    s == pos && e == scalar_end,
                    "{scheme} diverged at token {idx} (byte {pos}): scalar {pos}..{scalar_end} \
                     mask {s}..{e}: {:?} vs {:?}",
                    String::from_utf8_lossy(&bytes[pos..scalar_end]),
                    String::from_utf8_lossy(&bytes[s..e]),
                ),
                None => panic!("{scheme} ended early at token {idx} (byte {pos})"),
            }
            pos = scalar_end;
            idx += 1;
        }
        assert!(st.next_span::<S>(bytes).is_none(), "{scheme} produced extra tokens");
        eprintln!("{scheme}: all {idx} tokens match");
    }

    /// Deterministic xorshift generator for the differential fuzzers.
    pub(crate) fn xorshift(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// Random buffer of `pieces` concatenated to at least `target` bytes.
    pub(crate) fn soup(pieces: &[&str], target: usize, rng: &mut impl FnMut() -> u64) -> Vec<u8> {
        let mut buf = Vec::new();
        while buf.len() < target {
            buf.extend_from_slice(pieces[(rng() % pieces.len() as u64) as usize].as_bytes());
        }
        buf
    }

    /// Small cases shared by the cl100k-family schemes (the reference
    /// regex decides each scheme's expectation).
    pub(crate) const CL100K_FAMILY_CASES: &[&str] = &[
        "hello",
        " hello",
        "hello world",
        "  hello",
        "   hello",
        "\thello",
        "\t\thello",
        "\nhello",
        "\n\nhello",
        "\n\n   hello",
        "!hello",
        "!!hello",
        "?!x",
        "don't",
        "DON'T",
        "they'LL go",
        "it'S he'Ll",
        "we'Ve THEY'RE",
        "'sound",
        "'lx",
        "'hello",
        " 'hello",
        " 's",
        "x'0",
        "123",
        "1234",
        "1234567",
        "12345678901",
        " 123",
        " 1234",
        "  123",
        "3rd",
        "abc1234def",
        "3.14159",
        "hello, world!",
        "hi!\n\ndef",
        "hi !!\n\ndef",
        " !!!",
        "a-b",
        "a - b",
        "...",
        "hello\n",
        "hello \n",
        "hello \nx",
        "hello\n x",
        "hello  \n\n  ",
        "x \n\n ",
        "x  ",
        "x \t",
        "  \n  hello",
        "\r\nhello",
        "a\r\n",
        "a\r\n ",
        "a\n \n",
        "a \n \t",
        "\n\n",
        "\n\n\t",
        "   ",
        " ",
        "",
        "café",
        " café",
        "\u{a0}word",
        "voilà ¡hola!",
        "١٢٣٤٥",
        " ١٢٣٤٥",
        "e\u{301}f",
        "日本語のテキスト",
        " 日本語",
        "1٢3x",
        "1٢34",
        "tab\tsep\tvals",
        "\x0bword",
        "a\u{2028}b",
        "a\u{2028}\n",
        "price: $5.99!",
        "'ſ",
        "it'ſ fine",
        // Marks: `[\p{L}\p{M}]+` runs (Qwen3.5) vs marks as punctuation.
        "cafe\u{301} de\u{301}composed",
        "\u{301}leading mark",
        "\u{301}\u{301}two marks",
        " \u{301}abc",
        "\t\u{301}abc",
        "!\u{301}",
        "!\u{301}!",
        "!!\u{301}x",
        "1\u{301}2",
        "'\u{301}s",
        "देवनागरी में परीक्षण",
        "अंग्रेज़ी",
        "టెస్ట్ తెలుగు",
        "עִבְרִית נִקּוּד",
        "الْعَرَبِيَّة",
        "a\u{20dd}b",
        " \u{20dd}",
        "\u{200b}\u{301}x",
    ];

    /// Small cases shared by the o200k-family schemes: casing, suffix
    /// contractions, `/` tails, marks, and (Kimi) Han runs, numerals and
    /// symbols.
    pub(crate) const O200K_FAMILY_CASES: &[&str] = &[
        "hello",
        "Hello",
        "HELLO",
        "HeLLo",
        "camelCase",
        "PascalCase",
        "HTTPResponse",
        "HTTPresponse",
        "parseHTMLDocument",
        "XMLHttpRequest",
        "aB",
        "aBc",
        "ABc",
        "ABCdef GHIjkl",
        " hello",
        " Hello World",
        "hello world",
        "  hello",
        "\thello",
        "\tHello",
        "\nhello",
        "\n\nHello",
        "!hello",
        "!Hello",
        "!!hello",
        "?!x",
        "don't",
        "DON'T",
        "Don'T",
        "don'ts",
        "can'ts more",
        "they'LL go",
        "it'S he'Ll",
        "we'Ve THEY'RE",
        "x'll'd",
        "don't's",
        "o'clock",
        "don'x",
        "x'lm",
        "x'm'm",
        "'sound",
        "'Sound",
        "'lx",
        "'hello",
        " 'hello",
        " 's",
        " 'S",
        "x'0",
        "3's",
        "3'ts",
        "123",
        "1234",
        "1234567",
        " 123",
        "  123",
        "3rd",
        "abc1234def",
        "hello, world!",
        "hi!\n\ndef",
        "hi !!\n\ndef",
        " !!!",
        "a-b",
        "a - b",
        "...",
        "a/b",
        "a//b",
        "http://x.com/path",
        ".\n//x",
        "!\n/",
        "!\n/\n/x",
        "\n/",
        "//\n/",
        "x/\n",
        "x\n/",
        "hello\n",
        "hello \n",
        "hello \nx",
        "hello\n x",
        "hello  \n\n  ",
        "x \n\n ",
        "x  ",
        "x \t",
        "  \n  hello",
        "\r\nhello",
        "a\r\n",
        "a\r\n ",
        "a\n \n",
        "a \n \t",
        "\n\n",
        "\n\n\t",
        "   ",
        " ",
        "",
        "café",
        "Café",
        "CAFÉ",
        "cafÉ",
        " café",
        "\u{a0}word",
        "voilà ¡hola!",
        "ΑΒΓδε",
        "αβΓΔ",
        "Привет Мир",
        "ПРИВЕТ мир",
        "ẞßẞ",
        "ǅungla ǄUNGLA ǆungla",
        "١٢٣٤٥",
        "1٢3x",
        "١٢٣٤٥٦٧",
        "tab\tsep\tvals",
        "\x0bword",
        "a\u{2028}b",
        "a\u{2028}\n",
        "price: $5.99!",
        "'ſ",
        "x'ſ fine",
        "日本語のテキスト",
        " 日本語",
        "日本語ABC",
        "abc日本語Def",
        "e\u{301}f",
        "cafe\u{301} de\u{301}composed",
        "\u{301}leading mark",
        "\u{301}\u{301}two marks",
        " \u{301}abc",
        "\t\u{301}abc",
        "!\u{301}",
        "!\u{301}!",
        "!!\u{301}x",
        "!!\u{301}X",
        "1\u{301}2",
        "'\u{301}s",
        "A\u{301}B",
        "a\u{301}B",
        "x\u{301}'s",
        "деВНАгарІ",
        "देवनागरी में परीक्षण",
        "עִבְרִית נִקּוּד",
        "الْعَرَبِيَّة",
        "a\u{20dd}b",
        " \u{20dd}",
        "\u{200b}\u{301}x",
        // Han: run splits at script edges, numerals in digit groups,
        // symbols in punct runs, and the `[\r\n]*`-only Kimi tail.
        "中文",
        "中文模型",
        " 中文",
        "  中文",
        "\t中文",
        "\n中文",
        "!中文",
        "中English文",
        "abc中文def",
        "ABC中文",
        "中文ABC",
        "中'se",
        "中's",
        "中'S x",
        "中文's",
        "日本語のテキスト漢字かな",
        "漢字とひらがな",
        "中。文",
        "中，文！",
        "中文123",
        "123中文",
        "1〇2",
        "〇〇",
        "中〇文",
        "1〇〇〇〇",
        "1234〇",
        " 〇",
        "〇1",
        "⼀⼁⼂",
        "!⼀x",
        " ⼀",
        "中⼀文",
        "a⼀b",
        "々中",
        "中々",
        "〆切",
        "㐀㿿",
        "𠀀𠀁",
        "中\u{16FF0}文",
        "!\u{16FF0}",
        "}\n///doc",
        "*/\n/**",
        "`\n//! bindings",
        "path/to/file",
        "中\n文",
        "中\r\n文",
        "中 文",
        "中  文",
        "中\n\n文",
        "。\n中",
        "中。\n\n/x",
    ];
}
