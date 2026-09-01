//! Fast pretokenizer for the Qwen3.5 regex:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! Differences from the Qwen2/Qwen3 scheme:
//! - `\p{M}` joins letter runs (`[\p{L}\p{M}]+` instead of `\p{L}+`), so a
//!   combining mark extends a word and a bare mark run is a word of its own
//! - `\p{M}` is excluded from the punctuation run (`[^\s\p{L}\p{M}\p{N}]+`),
//!   so a mark after punctuation terminates the run
//!
//! The mask scanner is `cl100k_family`'s with the mark-joining classifier
//! (`MARKS_JOIN = true`); the scalar walker below mirrors the family's with
//! `[\p{L}\p{M}]` run membership.

use super::cl100k_family;
use super::mask::MaskScheme;
use super::{
    decode_cp, is_ascii_ws, is_digit, is_letter, scan_newlines, swar_scan_letters, ws_token_end,
};
use crate::pretokenize::unicode::{DsCharClass, ds_class_of};

pub(crate) struct Qwen35Scheme;

impl MaskScheme for Qwen35Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance_pos(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        cl100k_family::batch_masks::<false, true>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { cl100k_family::batch_masks_x86::<AVX512, false, true>(bytes, scan) }
    }
}

super::define_mask_pretokenizer!(FastQwen35Pretokenizer, Qwen35Scheme);

/// If the char at `pos` is `\p{L}` or `\p{M}`, return the offset just past it.
#[inline(always)]
fn lm_end_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some(pos + 1);
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        if matches!(ds_class_of(cp), DsCharClass::Letter | DsCharClass::Mark) {
            return Some(pos + l);
        }
    }
    None
}

/// `[\p{L}\p{M}]+` from `pos`.
#[inline(always)]
fn scan_lm_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if matches!(ds_class_of(cp), DsCharClass::Letter | DsCharClass::Mark) {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// `[^\s\p{L}\p{M}\p{N}]+` from `pos` (punctuation, symbols, controls —
/// everything except letters, marks, numbers, and whitespace).
#[inline(always)]
fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 {
                break;
            }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) {
                return p;
            }
            p += 1;
        }
        if p < len {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if matches!(ds_class_of(cp), DsCharClass::PunctSym | DsCharClass::Other) {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
fn ws_end(bytes: &[u8], start: usize) -> usize {
    ws_token_end::<false>(bytes, start, |cp| ds_class_of(cp) == DsCharClass::Whitespace)
}

/// Advance past one token starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
fn advance_pos(bytes: &[u8], pos: usize) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter — `[\p{L}\p{M}]+` with empty prefix
    if is_letter(b0) {
        return scan_lm_from(bytes, pos + 1);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space (`\s+(?!\S)` at EOS)
        };
        if is_letter(b1) {
            return scan_lm_from(bytes, pos + 2); // " word"
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // numbers never absorb the space
            }
            if is_ascii_ws(b1) {
                return ws_end(bytes, pos);
            }
            // ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
            let p = scan_other_from(bytes, pos + 2);
            return scan_newlines(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        match ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => return scan_lm_from(bytes, p1),
            DsCharClass::Whitespace => return ws_end(bytes, pos),
            DsCharClass::Number => return pos + 1,
            DsCharClass::PunctSym | DsCharClass::Other => {
                let p = scan_other_from(bytes, p1);
                return scan_newlines(bytes, p);
            }
        }
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        match ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => return scan_lm_from(bytes, p0),
            DsCharClass::Number => return p0, // `\p{N}`: exactly one char
            // Any non-letter/mark/number char except \r\n may prefix a run
            class => {
                if let Some(p) = lm_end_at(bytes, p0) {
                    return scan_lm_from(bytes, p);
                }
                if class == DsCharClass::Whitespace {
                    return ws_end(bytes, pos);
                }
                let p = scan_other_from(bytes, p0);
                return scan_newlines(bytes, p);
            }
        }
    }

    // ASCII digit: `\p{N}` matches exactly one char
    if is_digit(b0) {
        return pos + 1;
    }

    // Apostrophe: case-insensitive contractions
    if b0 == b'\'' {
        match bytes.get(pos + 1).map(u8::to_ascii_lowercase) {
            Some(b's' | b'd' | b'm' | b't') => return pos + 2,
            Some(b'l') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'l') => {
                return pos + 3;
            }
            Some(b'v') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            Some(b'r') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            _ => {}
        }
        // U+017F LATIN SMALL LETTER LONG S case-folds to 's' under `(?i)`
        if bytes.get(pos + 1) == Some(&0xC5) && bytes.get(pos + 2) == Some(&0xBF) {
            return pos + 3;
        }
        // Not a contraction: `'` can still prefix a letter/mark run
        if let Some(p) = lm_end_at(bytes, pos + 1) {
            return scan_lm_from(bytes, p);
        }
        let p = scan_other_from(bytes, pos + 1);
        return scan_newlines(bytes, p);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_end(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter/mark run
    if is_ascii_ws(b0) {
        if let Some(p) = lm_end_at(bytes, pos + 1) {
            return scan_lm_from(bytes, p);
        }
        return ws_end(bytes, pos);
    }

    // ASCII punctuation/symbol/control
    if let Some(p) = lm_end_at(bytes, pos + 1) {
        return scan_lm_from(bytes, p);
    }
    let p = scan_other_from(bytes, pos + 1);
    scan_newlines(bytes, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::test_support::*;

    /// The Qwen3.5 pattern verbatim — it contains no possessive quantifiers,
    /// so it runs directly under fancy-regex.
    const QWEN35_REF_REGEX: &str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn fast(s: &str) -> Vec<String> {
        fast_tokens(FastQwen35Pretokenizer::new(s.as_bytes()))
    }

    fn reference(s: &str) -> Vec<String> {
        regex_tokens(QWEN35_REF_REGEX, s)
    }

    #[test]
    fn qwen35_small_cases() {
        assert_cases(CL100K_FAMILY_CASES, fast, reference);
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes.
    #[test]
    fn qwen35_matches_regex_random() {
        let pools: &[&[char]] = &[
            &['a', 'Z', 'é', 'ß', 'Ж', 'ا', '한', '日'],      // letters
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
        ];
        assert_random_soup(pools, 0x93E3_5EEC, 2000, fast, reference);
    }

    #[test]
    fn qwen35_matches_regex_owt() {
        let input = load_owt_prefix(5_000_000);
        let text = std::str::from_utf8(&input).unwrap();
        assert_matches_regex(QWEN35_REF_REGEX, text, FastQwen35Pretokenizer::new(&input));
    }
}
