//! Fast pretokenizer for the Qwen2/Qwen3 regex:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! Differences from the cl100k scheme:
//! - `\p{N}` matches exactly ONE number char (cl100k allows up to 3)
//! - `\s*[\r\n]+` outranks the end-of-input whitespace rule: a whitespace
//!   run containing a newline always splits right after its LAST newline,
//!   even at EOS, and the remaining whitespace becomes a separate token
//!   (cl100k keeps trailing whitespace at EOS as one token via `\s+$`)
//!
//! See `cl100k_family` (`DIGITS3 = false`, `EOS_WS_WHOLE = false`).

use super::cl100k_family;
use super::mask::MaskScheme;

pub(crate) struct Qwen2Scheme;

impl MaskScheme for Qwen2Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        cl100k_family::advance_pos::<false, false>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        cl100k_family::batch_masks::<false, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { cl100k_family::batch_masks_x86::<AVX512, false, false>(bytes, scan) }
    }
}

super::define_mask_pretokenizer!(FastQwen2Pretokenizer, Qwen2Scheme);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::test_support::*;

    /// The Qwen2 pattern verbatim — it contains no possessive quantifiers,
    /// so it runs directly under fancy-regex.
    const QWEN2_REF_REGEX: &str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn fast(s: &str) -> Vec<String> {
        fast_tokens(FastQwen2Pretokenizer::new(s.as_bytes()))
    }

    #[test]
    fn qwen2_small_cases() {
        assert_cases(CL100K_FAMILY_CASES, fast, |s| regex_tokens(QWEN2_REF_REGEX, s));
    }

    #[test]
    fn qwen2_matches_regex_owt() {
        let input = load_owt_prefix(5_000_000);
        let text = std::str::from_utf8(&input).unwrap();
        assert_matches_regex(QWEN2_REF_REGEX, text, FastQwen2Pretokenizer::new(&input));
    }
}
