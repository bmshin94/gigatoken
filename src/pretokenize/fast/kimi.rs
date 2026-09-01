//! Fast pretokenizer for the Kimi regex (moonshotai Kimi-K2 family, from
//! `tokenization_kimi.py`):
//! `[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The o200k scheme with a leading `[\p{Han}]+` alternative, Han excluded
//! from the letter brackets, and no `/` in the absorbed punct tail. See
//! `o200k_family` (`CONTRACTIONS = true`, `DIGITS3 = true`,
//! `SLASH = false`, `HAN = true`).

use super::mask::MaskScheme;
use super::o200k_family;

pub(crate) struct KimiScheme;

impl MaskScheme for KimiScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<true, true, false, true>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<true, true, false, true>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { o200k_family::batch_masks_x86::<AVX512, true, true, false, true>(bytes, scan) }
    }
}

super::define_mask_pretokenizer!(FastKimiPretokenizer, KimiScheme);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::test_support::*;

    /// The Kimi pattern verbatim (from `tokenization_kimi.py`; only greedy
    /// quantifiers, so it runs directly under fancy-regex, which shares
    /// regex-syntax's `&&` class intersection).
    const KIMI_REF_REGEX: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn fast(s: &str) -> Vec<String> {
        fast_tokens(FastKimiPretokenizer::new(s.as_bytes()))
    }

    fn reference(s: &str) -> Vec<String> {
        regex_tokens(KIMI_REF_REGEX, s)
    }

    #[test]
    fn kimi_small_cases() {
        assert_cases(O200K_FAMILY_CASES, fast, reference);
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes
    /// (including the Han classes).
    #[test]
    fn kimi_matches_regex_random() {
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', 'ひ', 'カ'],   // lower/caseless (non-Han)
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],                  // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers (non-Han)
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
            &['s', 't', 'm', 'd', 'l', 'v', 'r', 'e', 'S', 'T', 'L'], // suffix letters
            &['中', '文', '日', '本', '語', '々', '〆', '㐀', '𠀀'], // Han letters
            &['〇', '〡', '〢', '㆒'],                          // Han numerals (Nl)
            &['⼀', '⼁', '⺀', '\u{16FF0}'],                   // Han symbols/marks (So/Mc)
        ];
        assert_random_soup(pools, 0x93E3_5EEC, 6000, fast, reference);
    }

    #[test]
    fn kimi_matches_regex_owt() {
        let input = load_owt_prefix(5_000_000);
        let text = std::str::from_utf8(&input).unwrap();
        assert_matches_regex(KIMI_REF_REGEX, text, FastKimiPretokenizer::new(&input));
    }
}
