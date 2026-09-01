//! Fast pretokenizer for the cl100k_base (GPT-3.5/GPT-4) regex:
//! `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s+`
//!
//! Differences from the r50k scheme:
//! - contractions are case-insensitive (`'S`, `'Ll`, ...)
//! - a letter run absorbs ONE preceding char of any kind except `\r`, `\n`,
//!   letters, and numbers (not just a space: `!word`, `\tword`, `\u{A0}word`)
//! - number runs are at most 3 chars and never absorb a leading space
//! - a punctuation run absorbs trailing `\r`/`\n` chars (`"!!\n\n"`)
//! - a whitespace run containing a newline splits right after its LAST
//!   newline (`\s*[\r\n]`); trailing whitespace at EOS stays one token
//!
//! See `cl100k_family` for the shared scalar walker and mask-scanner
//! boundary algebra (`DIGITS3 = true`, `EOS_WS_WHOLE = true`).

use super::cl100k_family;
use super::mask::MaskScheme;

pub(crate) struct Cl100kScheme;

impl MaskScheme for Cl100kScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        cl100k_family::advance_pos::<true, true>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        cl100k_family::batch_masks::<true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { cl100k_family::batch_masks_x86::<AVX512, true, false>(bytes, scan) }
    }
}

super::define_mask_pretokenizer!(FastCl100kPretokenizer, Cl100kScheme);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::test_support::*;

    /// Backtracking-compatible equivalent of the possessive cl100k pattern.
    /// Each possessive quantifier is safe to relax because nothing after it
    /// can match a char its class rejected, and `$` only matches at EOS.
    const CL100K_REF_REGEX: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s+";

    fn fast(s: &str) -> Vec<String> {
        fast_tokens(FastCl100kPretokenizer::new(s.as_bytes()))
    }

    #[test]
    fn cl100k_small_cases() {
        assert_cases(CL100K_FAMILY_CASES, fast, |s| regex_tokens(CL100K_REF_REGEX, s));
    }

    #[test]
    fn cl100k_matches_regex_owt() {
        let input = load_owt_prefix(5_000_000);
        let text = std::str::from_utf8(&input).unwrap();
        assert_matches_regex(CL100K_REF_REGEX, text, FastCl100kPretokenizer::new(&input));
    }
}
