//! Fast pretokenizer for the Nemotron-3 regex (nvidia Nemotron-3 family):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The o200k scheme without contraction suffixes and with single-char
//! `\p{N}` digit tokens. See `o200k_family` (`CONTRACTIONS = false`,
//! `DIGITS3 = false`).

use super::mask::MaskScheme;
use super::o200k_family;

pub(crate) struct NemotronScheme;

impl MaskScheme for NemotronScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<false, false, true, false>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<false, false, true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { o200k_family::batch_masks_x86::<AVX512, false, false, true, false>(bytes, scan) }
    }
}

super::define_mask_pretokenizer!(FastNemotronPretokenizer, NemotronScheme);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::test_support::*;

    /// The Nemotron pattern verbatim — no possessive quantifiers, so it
    /// runs directly under fancy-regex.
    const NEMOTRON_REF_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn fast(s: &str) -> Vec<String> {
        fast_tokens(FastNemotronPretokenizer::new(s.as_bytes()))
    }

    fn reference(s: &str) -> Vec<String> {
        regex_tokens(NEMOTRON_REF_REGEX, s)
    }

    /// The o200k-family case list applies verbatim (contraction cases just
    /// tokenize differently, which the reference regex reflects).
    #[test]
    fn nemotron_small_cases() {
        assert_cases(O200K_FAMILY_CASES, fast, reference);
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes.
    #[test]
    fn nemotron_matches_regex_random() {
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', '日'],      // lower/caseless
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],                  // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
        ];
        assert_random_soup(pools, 0x93E3_5EEE, 3000, fast, reference);
    }
}
