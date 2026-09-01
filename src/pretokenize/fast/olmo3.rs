//! Fast pretokenizer for the Olmo 2/3 (dolma2) regex:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The Qwen2 scheme with cl100k's number rule: `\p{N}{1,3}` matches runs
//! of up to THREE number chars (Qwen2 matches exactly one). See
//! `cl100k_family` (`DIGITS3 = true`, `EOS_WS_WHOLE = false`).

use super::cl100k_family;
use super::mask::MaskScheme;

pub(crate) struct Olmo3Scheme;

impl MaskScheme for Olmo3Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        cl100k_family::advance_pos::<true, false>(bytes, pos)
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

super::define_mask_pretokenizer!(FastOlmo3Pretokenizer, Olmo3Scheme);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::test_support::*;

    /// The Olmo3/dolma2 pattern verbatim — no possessive quantifiers, so it
    /// runs directly under fancy-regex.
    const OLMO3_REF_REGEX: &str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn fast(s: &str) -> Vec<String> {
        fast_tokens(FastOlmo3Pretokenizer::new(s.as_bytes()))
    }

    #[test]
    fn olmo3_small_cases() {
        assert_cases(CL100K_FAMILY_CASES, fast, |s| regex_tokens(OLMO3_REF_REGEX, s));
    }

    #[test]
    fn olmo3_matches_regex_owt() {
        let input = load_owt_prefix(5_000_000);
        let text = std::str::from_utf8(&input).unwrap();
        assert_matches_regex(OLMO3_REF_REGEX, text, FastOlmo3Pretokenizer::new(&input));
    }

    /// Full-OWT (~12 GB) comparison against the reference regex, parallelized
    /// with rayon over ~32 MB chunks cut at newline boundaries (both sides
    /// tokenize the identical chunk, so splitting is safe). Run with:
    /// `cargo test --release olmo3_matches_regex_owt_full -- --ignored --nocapture`
    #[test]
    #[ignore = "reads the full ~12 GB OWT file; run explicitly in release mode"]
    fn olmo3_matches_regex_owt_full() {
        use rayon::prelude::*;

        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let file = std::fs::File::open(&path).expect("Could not open ~/data/owt_train.txt");
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let bytes: &[u8] = &mmap;

        const CHUNK: usize = 32 * 1024 * 1024;
        let mut boundaries = vec![0usize];
        while *boundaries.last().unwrap() < bytes.len() {
            let target = (*boundaries.last().unwrap() + CHUNK).min(bytes.len());
            let end = if target == bytes.len() {
                target
            } else {
                // Cut at the next newline (ASCII, so always a UTF-8 boundary).
                match memchr::memchr(b'\n', &bytes[target..]) {
                    Some(off) => target + off + 1,
                    None => bytes.len(),
                }
            };
            boundaries.push(end);
        }
        eprintln!(
            "Comparing olmo3 fast pretokenizer vs regex on {:.2} GB in {} chunks",
            bytes.len() as f64 / 1e9,
            boundaries.len() - 1
        );
        boundaries.par_windows(2).for_each(|w| {
            let chunk = &bytes[w[0]..w[1]];
            let text = std::str::from_utf8(chunk).expect("chunk is not valid UTF-8");
            eprintln!("chunk at byte {}", w[0]);
            assert_matches_regex(OLMO3_REF_REGEX, text, FastOlmo3Pretokenizer::new(chunk));
        });
    }
}
