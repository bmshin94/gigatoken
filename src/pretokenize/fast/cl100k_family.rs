//! Shared scalar walker and mask-scanner boundary algebra for the cl100k
//! regex family: cl100k, olmo3, qwen2, and qwen3.5. Their patterns share
//! the shape
//!
//! `'(?i:contractions)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3} or \p{N}|
//!  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! and differ in the digit-group size (`DIGITS3`), whether trailing
//! whitespace at end of input stays whole (`EOS_WS_WHOLE`, cl100k's
//! `\s++$`; only the scalar tail ever sees end of input), and whether
//! `\p{M}` joins letter runs (`MARKS_JOIN`, qwen3.5).
//!
//! Boundary rules:
//! - A letter starts a token unless it continues a letter run, follows
//!   space/tab-class whitespace (which absorbs one following letter run
//!   via the `[^\r\n\p{L}\p{N}]?` prefix), or follows a punct char that is
//!   itself at a boundary — i.e. whose own predecessor is neither punct nor
//!   a space (a two-chars-back test, char-aware for multi-byte chars).
//! - Digits split every 1 or 3 chars from each run start and never absorb
//!   a preceding space.
//! - A punct char starts a token unless it continues a punct run or
//!   follows a space (` ?[^\s\p{L}\p{N}]+`).
//! - Newlines directly after a punct run are absorbed (`[\r\n]*`).
//! - A whitespace run containing newlines emits one token through its LAST
//!   newline, then the r50k-style tail rules; NL-free runs split before
//!   their last char when followed by non-ws (`\s+(?!\S)`). A run touching
//!   the batch end resolves in-batch when the char at byte 64 is non-ws;
//!   a run actually crossing the edge defers to the scalar path (its last
//!   newline may lie in a later batch).

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::mask::{self, AsciiMasks, smear_up};
use super::{
    decode_cp, digit_token_end, is_ascii_ws, is_digit, is_letter, letter_end_at,
    scan_letters_from, scan_newlines, scan_other_from, ws_token_end,
};
use crate::pretokenize::unicode::{self, CharClass};

// Scalar ground truth

#[inline(always)]
fn ws_end<const EOS_WS_WHOLE: bool>(bytes: &[u8], start: usize) -> usize {
    ws_token_end::<EOS_WS_WHOLE>(bytes, start, |cp| {
        unicode::class_of(cp) == CharClass::Whitespace
    })
}

/// Advance past one token starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
pub(crate) fn advance_pos<const DIGITS3: bool, const EOS_WS_WHOLE: bool>(
    bytes: &[u8],
    pos: usize,
) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter — `\p{L}+` with empty prefix
    if is_letter(b0) {
        return scan_letters_from(bytes, pos + 1);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space
        };
        if is_letter(b1) {
            return scan_letters_from(bytes, pos + 2); // " word"
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // numbers never absorb the space
            }
            if is_ascii_ws(b1) {
                return ws_end::<EOS_WS_WHOLE>(bytes, pos);
            }
            // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
            let p = scan_other_from(bytes, pos + 2);
            return scan_newlines(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        match unicode::class_of(cp) {
            CharClass::Letter => return scan_letters_from(bytes, p1),
            CharClass::Whitespace => return ws_end::<EOS_WS_WHOLE>(bytes, pos),
            CharClass::Number => return pos + 1,
            CharClass::Other => {
                let p = scan_other_from(bytes, p1);
                return scan_newlines(bytes, p);
            }
        }
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        let class = unicode::class_of(cp);
        if class == CharClass::Letter {
            return scan_letters_from(bytes, p0);
        }
        if class == CharClass::Number {
            return digit_token_end::<DIGITS3>(bytes, p0);
        }
        // Any non-letter/number char except \r\n may prefix a letter run
        if let Some(p) = letter_end_at(bytes, p0) {
            return scan_letters_from(bytes, p);
        }
        if class == CharClass::Whitespace {
            return ws_end::<EOS_WS_WHOLE>(bytes, pos);
        }
        let p = scan_other_from(bytes, p0);
        return scan_newlines(bytes, p);
    }

    // ASCII digit
    if is_digit(b0) {
        return digit_token_end::<DIGITS3>(bytes, pos + 1);
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
        // Not a contraction: `'` can still prefix a letter run
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        let p = scan_other_from(bytes, pos + 1);
        return scan_newlines(bytes, p);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_end::<EOS_WS_WHOLE>(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter run
    if is_ascii_ws(b0) {
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        return ws_end::<EOS_WS_WHOLE>(bytes, pos);
    }

    // ASCII punctuation/symbol
    if let Some(p) = letter_end_at(bytes, pos + 1) {
        return scan_letters_from(bytes, p);
    }
    let p = scan_other_from(bytes, pos + 1);
    scan_newlines(bytes, p)
}

// Mask-scanner boundary algebra

/// Boundary carries from the two chars before the batch: P1 ends at
/// `scan - 1`, P2 is the one before it (the two-chars-back absorb test).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Clone, Copy, Default)]
struct Carries {
    /// P1 is a letter / space (0x20) / non-newline non-space ws / punct /
    /// any ws / digit.
    pl: u64,
    ps: u64,
    pwt: u64,
    po: u64,
    pws: u64,
    pd: u64,
    /// P2 is punct-or-space, for a char lead at bit 0 (P1 entirely before
    /// the batch).
    c2_os: u64,
    /// Same test positioned at the first lead AFTER a P1 that straddles
    /// into the batch (P1's own prev is then P2).
    b2b_in: u64,
}

/// Pure-ASCII carries (hot, branchless). Requires `scan > 0` and
/// `bytes[scan-1] < 0x80` (and `bytes[scan-2] < 0x80` when present).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn ascii_carries(bytes: &[u8], scan: usize) -> Carries {
    let b = bytes[scan - 1];
    let bit = |c: bool| u64::from(c);
    let (l, d, w) = (super::is_letter(b), super::is_digit(b), is_ascii_ws(b));
    let n = b == b'\r' || b == b'\n';
    let c2_os = if scan >= 2 {
        let b2 = bytes[scan - 2];
        bit(b2 == b' '
            || (!super::is_letter(b2) && !super::is_digit(b2) && !is_ascii_ws(b2)))
    } else {
        0
    };
    Carries {
        pl: bit(l),
        ps: bit(b == b' '),
        pwt: bit(w && !n && b != b' '),
        po: bit(!l && !d && !w),
        pws: bit(w),
        pd: bit(d),
        c2_os,
        b2b_in: 0,
    }
}

/// `(usable, bad)` for `bytes[scan..scan+64]` under the cl100k-family
/// rules (`DIGITS3`: `\p{N}{1,3}` vs `\p{N}`; `MARKS_JOIN`: `\p{M}` joins
/// letter runs). NEON classifies the ASCII classes and the pure-ASCII
/// boundary algebra stays inline; batches with any non-ASCII byte in or
/// just before them take [`family_extended_masks`], `#[inline(never)]` so
/// the hot path's register allocation stays clean.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn batch_masks<const DIGITS3: bool, const MARKS_JOIN: bool>(
    bytes: &[u8],
    scan: usize,
) -> (u64, u64) {
    use std::arch::aarch64::*;
    let len = bytes.len();
    if scan + 70 > len {
        // Not enough lookahead for the batch-edge char classification
        // (up to a 4-byte char starting at scan + 66); scalar batch.
        return (0, u64::MAX);
    }
    unsafe {
        let p = bytes.as_ptr().add(scan);
        let zero = vdupq_n_u8(0);
        let mut lv = [zero; 4];
        let mut dv = [zero; 4];
        let mut sv = [zero; 4];
        let mut wsv = [zero; 4];
        let mut nv = [zero; 4];
        let mut hiv = [zero; 4];
        let mut apv = [zero; 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            lv[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            dv[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            sv[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            wsv[i] = vorrq_u8(
                sv[i],
                vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)),
            );
            nv[i] = vorrq_u8(
                vceqq_u8(v, vdupq_n_u8(b'\r')),
                vceqq_u8(v, vdupq_n_u8(b'\n')),
            );
            hiv[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            apv[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
        }
        let l64 = mask::movemask64(lv[0], lv[1], lv[2], lv[3]);
        let d64 = mask::movemask64(dv[0], dv[1], dv[2], dv[3]);
        let s64 = mask::movemask64(sv[0], sv[1], sv[2], sv[3]);
        let wsa = mask::movemask64(wsv[0], wsv[1], wsv[2], wsv[3]);
        let n64 = mask::movemask64(nv[0], nv[1], nv[2], nv[3]);

        // Apostrophes only matter for the contraction fixup.
        let ap_any = vorrq_u8(vorrq_u8(apv[0], apv[1]), vorrq_u8(apv[2], apv[3]));
        let ap64 = if vmaxvq_u8(ap_any) != 0 {
            mask::movemask64(apv[0], apv[1], apv[2], apv[3])
        } else {
            0
        };

        let am = mask::AsciiMasks {
            l: l64,
            d: d64,
            s: s64,
            wt: wsa & !s64 & !n64,
            n: n64,
            hi: 0,
            ap: ap64,
        };

        // Any non-ASCII byte in the batch — or within the two carry bytes
        // before it — routes to the extended classifier.
        let hi_any = vorrq_u8(vorrq_u8(hiv[0], hiv[1]), vorrq_u8(hiv[2], hiv[3]));
        if vmaxvq_u8(hi_any) != 0
            || (scan >= 1 && bytes[scan - 1] >= 0x80)
            || (scan >= 2 && bytes[scan - 2] >= 0x80)
        {
            let mut am = am;
            am.hi = mask::movemask64(hiv[0], hiv[1], hiv[2], hiv[3]);
            return family_extended_masks::<DIGITS3, MARKS_JOIN>(bytes, scan, am);
        }

        let cr = if scan == 0 { Carries::default() } else { ascii_carries(bytes, scan) };
        family_algebra::<DIGITS3>(bytes, scan, am, cr, mask::UniClasses::default())
    }
}

/// x86-64 front-end: same contract as the NEON `batch_masks`, monomorphized
/// on the SIMD tier (see `MaskScheme::batch_masks_x86`). `#[inline(always)]`
/// with no `target_feature` of its own, so the body fuses into whichever
/// feature region calls it (LLVM declined to inline a `#[target_feature]`
/// form into the fill wrappers).
///
/// # Safety
///
/// The selected tier must have been runtime-detected
/// ([`mask::avx512_scanner_available`] /
/// [`mask::avx2_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn batch_masks_x86<
    const AVX512: bool,
    const DIGITS3: bool,
    const MARKS_JOIN: bool,
>(
    bytes: &[u8],
    scan: usize,
) -> (u64, u64) {
    let len = bytes.len();
    if scan + 70 > len {
        // Not enough lookahead for the batch-edge char classification
        // (up to a 4-byte char starting at scan + 66); scalar batch.
        return (0, u64::MAX);
    }
    let am = if AVX512 {
        // SAFETY: the caller detected the AVX-512 tier (fn contract).
        unsafe { mask::ascii_masks_avx512(bytes, scan) }
    } else {
        // SAFETY: the caller detected the AVX2 tier (fn contract).
        unsafe { mask::ascii_masks_avx2(bytes, scan) }
    };

    // Any non-ASCII byte in the batch — or within the two carry bytes
    // before it — routes to the extended classifier. (`am.hi` is exact
    // and already computed, unlike NEON's lazily-movemasked variant.)
    if am.hi != 0
        || (scan >= 1 && bytes[scan - 1] >= 0x80)
        || (scan >= 2 && bytes[scan - 2] >= 0x80)
    {
        // SAFETY: both detected tiers include the BMI1/BMI2/LZCNT/POPCNT
        // bit features `family_extended_masks` re-declares (fn contract).
        return unsafe { family_extended_masks::<DIGITS3, MARKS_JOIN>(bytes, scan, am) };
    }

    let cr = if scan == 0 { Carries::default() } else { ascii_carries(bytes, scan) };
    family_algebra::<DIGITS3>(bytes, scan, am, cr, mask::UniClasses::default())
}

/// Slow(er) path for batches with non-ASCII in or just before them: the
/// carries walk back through multi-byte chars and every unicode char in
/// the batch joins the effective class masks ([`mask::classify_uni_chars`]),
/// then the shared boundary algebra applies unchanged. Only number chars
/// (char-counted `\p{N}{1,3}` grouping), whitespace straddling the batch
/// end, and stray continuation bytes stay bad zones. `#[inline(never)]`
/// keeps the walker's register allocation clean; the x86 `target_feature`
/// keeps its bit scans on tzcnt/lzcnt/blsr in a baseline build (see
/// `mask.rs`).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "bmi1,bmi2,lzcnt,popcnt")
)]
#[inline(never)]
fn family_extended_masks<const DIGITS3: bool, const MARKS_JOIN: bool>(
    bytes: &[u8],
    scan: usize,
    am: mask::AsciiMasks,
) -> (u64, u64) {
    // Class-table LazyLock resolved once; the per-char classify below is
    // then a bare slice index.
    let ct = unicode::ClassTable::<MARKS_JOIN>::get();
    let class = move |cp| ct.class_of(cp);

    // A P1 straddling into the batch claims its continuation bytes with
    // its class; `b2b_in` is the two-back test for the char right after
    // it, whose predecessor chain starts before the batch.
    let mut cl = mask::UniClasses::default();
    let cr = if scan == 0 {
        Carries::default()
    } else if bytes[scan - 1] < 0x80 && (scan < 2 || bytes[scan - 2] < 0x80) {
        ascii_carries(bytes, scan)
    } else {
        // A multi-byte char within two bytes of the batch start.
        // SAFETY: scan > 0 on this branch, and the classifier's
        // scan + 70 <= len batch guard covers pos + 3 <= len.
        let (c1, j1, e1) = unsafe { mask::char_through(bytes, scan, class) };
        let pb = bytes[scan - 1];
        let chm = if e1 > scan { (1u64 << (e1 - scan)) - 1 } else { 0 };
        cl.cont = chm;
        let c2v = if j1 == 0 {
            0
        } else {
            // SAFETY: j1 > 0 just checked, and j1 < scan keeps the decode
            // within the classifier's scan + 70 <= len batch guard.
            let c2c = unsafe { mask::char_through(bytes, j1, class) }.0;
            u64::from(bytes[j1 - 1] == b' ' || c2c == CharClass::Other)
        };
        let mut c = Carries::default();
        if e1 > scan {
            c.b2b_in = c2v << (e1 - scan);
        } else {
            c.c2_os = c2v;
        }
        c.pd = u64::from(c1 == CharClass::Number);
        match c1 {
            CharClass::Letter => {
                cl.l = chm;
                c.pl = 1;
            }
            // A digit P1 sets no letter/punct carries (`\p{N}` groups
            // restart at token boundaries). One straddling into the batch
            // defeats the `pd` seed (bit 0 is its continuation byte, not
            // an ASCII digit), so its bytes defer via resid instead.
            CharClass::Number => {
                cl.n = chm;
                cl.resid |= chm;
            }
            CharClass::Other => {
                cl.o = chm;
                c.po = 1;
            }
            CharClass::Whitespace => {
                cl.ws = chm;
                if e1 > scan {
                    // Straddling-in ws: run bookkeeping crosses the edge.
                    cl.resid = chm;
                }
                c.ps = u64::from(pb == b' ');
                let nl = pb == b'\r' || pb == b'\n';
                c.pwt = u64::from(pb != b' ' && !nl);
                c.pws = 1;
            }
        }
        c
    };

    let mut uni = if am.hi != 0 {
        // SAFETY: this classifier's scan + 70 <= len batch guard is
        // exactly `classify_uni_chars`' contract.
        unsafe { mask::classify_uni_chars::<false, true>(bytes, scan, am.hi & !cl.cont, class) }
    } else {
        mask::UniClasses::default()
    };
    uni.l |= cl.l;
    uni.n |= cl.n;
    uni.o |= cl.o;
    uni.ws |= cl.ws;
    uni.cont |= cl.cont;
    uni.resid |= cl.resid;

    family_algebra::<DIGITS3>(bytes, scan, am, cr, uni)
}

/// The scheme family's shared u64 boundary algebra over per-byte class
/// masks. `uni` is all-zero on the pure-ASCII path (the constant folds
/// away every unicode term); the extended path passes real class masks
/// with straddle-in claims already merged in.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn family_algebra<const DIGITS3: bool>(
    bytes: &[u8],
    scan: usize,
    am: mask::AsciiMasks,
    cr: Carries,
    uni: mask::UniClasses,
) -> (u64, u64) {
    let Carries { pl, ps, pwt, po, pws, pd, c2_os, b2b_in } = cr;
    let contm = uni.cont;
    let resid = uni.resid;

    // Effective per-byte classes: every byte of a classified char carries
    // the char's class, so byte-adjacency == char-adjacency.
    let lb = am.l | uni.l;
    let sb = am.s; // the ` ?` / prefix "space" is ASCII 0x20 only
    let wtb = am.wt | uni.ws;
    let ob = !(am.l | am.d | am.s | am.wt | am.n | am.hi) | uni.o;
    let ws_all = sb | wtb | am.n;

    // --- Letters: `[^\r\n\p{L}\p{N}]?\p{L}+` -------------------------------
    // B: "the char two back is punct or space" — evaluated at each char's
    // lead by shifting the prev-byte test C by the PREV char's length.
    let len1 = !(contm | uni.lead2 | uni.lead3 | uni.lead4);
    let c_test = ((ob | sb) << 1) | po | ps; // bit 0: byte scan-1 in O|S
    let b2back = ((c_test & len1) << 1)
        | ((c_test & uni.lead2) << 2)
        | ((c_test & uni.lead3) << 3)
        | ((c_test & uni.lead4) << 4)
        | c2_os // prev char entirely before the batch
        | b2b_in; // prev char straddles in; its own prev is P2
    let p_l = (lb << 1) | pl;
    let p_s = (sb << 1) | ps;
    let p_wt = (wtb << 1) | pwt;
    let p_o = (ob << 1) | po;
    let absorb = p_o & !b2back;
    let b_letters = lb & !contm & !p_l & !p_s & !p_wt & !absorb;

    // --- Digits: `\p{N}{1,3}` or `\p{N}` -----------------------------------
    // The run-split hop loop only runs when a run of 2+ digits exists.
    let b_digits = if DIGITS3 && am.d & (am.d >> 1) != 0 {
        mask::digit_run_splits3(am.d)
    } else {
        am.d
    };

    // --- Punct: ` ?[^\s\p{L}\p{N}]+` ----------------------------------------
    let b_punct = ob & !contm & !p_o & !p_s;

    // --- Whitespace ---------------------------------------------------------
    // Newlines directly after a punct run are absorbed (`[\r\n]*`). The
    // smear only runs on a nonzero seed (most batches have no
    // punct-adjacent newline).
    let abs_seed = am.n & ((ob << 1) | po);
    let abs_n = if abs_seed == 0 { 0 } else { smear_up(abs_seed, am.n) };
    let ws_eff = ws_all & !abs_n;

    let mut bad = resid | resid << 1 | resid >> 1;

    // Byte-64 lookahead: is the char at the next batch's first byte
    // non-ws? Branchless for an ASCII byte 64 (a ~20% coin flip on natural
    // text); only a non-ASCII byte 64 branches. Guarded on `bad >> 63`: a
    // ws char straddling out makes byte 64 a continuation byte, not a lead.
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    let nn64 = if nb64 < 0x80 {
        !is_ascii_ws(nb64)
    } else {
        // SAFETY: the same scan + 70 <= len batch guard puts the decode at
        // scan + 64 in bounds (needs scan + 68 <= len).
        bad >> 63 == 0 && unsafe { mask::nn_at_full(bytes, scan + 64) }
    };
    let nn64m = u64::from(nn64).wrapping_neg(); // all-ones when non-ws

    // A punct-absorbed newline run touching the batch end: if the char at
    // byte 64 is ws, the token may continue (another newline), and even
    // when it doesn't, the next batch cannot tell the absorbed `\n` before
    // its bit 0 from a ws-run `\n` — defer to the scalar path. If byte 64
    // is non-ws, the punct token ends exactly at the batch edge.
    if abs_n >> 63 != 0 && !nn64 {
        bad |= 1u64 << 63;
    }

    // A ws run touching the batch end resolves in-batch when byte 64's
    // char is non-ws (the run's last newline and its `(?!\S)` split are
    // then all visible; `nn64m` feeds the lookahead bits below).
    // Otherwise it defers: its last newline (and the `\s+$`-style
    // end-of-input rules) may lie beyond this batch.
    let nonws = !ws_eff;
    if ws_eff >> 63 != 0 && !nn64 {
        if nonws == 0 {
            return (0, u64::MAX); // whole batch one ws run
        }
        let h = 63 - nonws.leading_zeros(); // highest non-ws bit (< 63)
        bad |= u64::MAX << (h + 1);
    }

    // A digit run crossing the batch END needs no deferral (its in-batch
    // splits are phased from its in-batch start). One whose phase did NOT
    // start in this batch — continuing from before it (`pd`) or following
    // a bad zone that may hold digit-class chars — defers, since
    // `digit_run_splits3` phases every run from its first in-batch digit.
    if DIGITS3 {
        let seed = (am.d & (bad << 1)) | (am.d & pd);
        if seed != 0 {
            bad |= smear_up(seed, am.d);
        }
    }

    // Base rule (correct for NL-free runs; NL runs are overridden below):
    // run start, or split before the last char when followed by non-ws.
    let ws_leads1 = (am.s | am.wt | am.n) & ws_eff;
    let ws_leads = (ws_leads1 | uni.w2 | uni.w3) & !abs_n;
    let p_ws = (ws_eff << 1) | pws; // prev byte ws (any kind)
    // Last-char `(?!\S)` split: in-batch via shifted nonws; the run
    // touching bit 63 uses the byte-64 lookahead (`nn64m`). A 2-byte ws
    // led at 62 or 3-byte ws led at 61 ends at the edge too.
    let edge_last = (ws_leads1 & (1 << 63)) | (uni.w2 & (1 << 62)) | (uni.w3 & (1 << 61));
    let split_ok = (ws_leads1 & (nonws >> 1))
        | (uni.w2 & (nonws >> 2))
        | (uni.w3 & (nonws >> 3))
        | (edge_last & nn64m);
    let mut b_ws = ws_leads & (!p_ws | split_ok);

    // Override every run that contains a (non-absorbed) newline: one token
    // through the run's last newline, then r50k-style tail rules. (A
    // branchless downward-smear formulation measured 0.95x; keep the loop.)
    let mut runs_n = am.n & ws_eff & !bad;
    while runs_n != 0 {
        let f = runs_n.trailing_zeros();
        let below_gap = nonws & ((1u64 << f) - 1);
        let a = if below_gap == 0 { 0 } else { 64 - below_gap.leading_zeros() };
        // First non-ws above f, or 64 for a run ending exactly at the
        // batch edge (only reachable when `nn64`).
        let e = (nonws & (u64::MAX << f)).trailing_zeros();
        let run_mask = (u64::MAX << a) & !u64::MAX.unbounded_shl(e);
        b_ws &= !run_mask;
        // Run start. Bit-0-leading runs with prev-byte ws cannot contain a
        // newline (scalar resumes only after `\s*[\r\n]+` tokens), so `a`
        // is always a true run start here.
        b_ws |= 1u64 << a;
        let q = 63 - (am.n & run_mask).leading_zeros(); // last NL in run
        if (q + 1) < e {
            // Tail after the last newline: starts a token, and its last
            // char splits off before the following non-ws char.
            b_ws |= 1u64 << (q + 1);
            let tail = run_mask & (u64::MAX << (q + 1));
            let tail_leads = ws_leads & tail;
            b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
        }
        runs_n &= !run_mask;
    }

    let mut boundary = b_letters | b_digits | b_punct | b_ws;

    // --- Contractions: `'(?i:[sdmt]|ll|ve|re)` ------------------------------
    // ('ſ — U+017F — is non-ASCII, so it already sits in a bad zone.)
    let mut cand = am.ap & boundary & !bad;
    while cand != 0 {
        let i = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        if i >= 61 {
            bad |= u64::MAX << i;
            break;
        }
        let b1 = bytes[scan + i + 1];
        if b1 >= 0x80 {
            // `(?i:'s)` also matches 'ſ (U+017F): an apostrophe before any
            // non-ASCII char defers to the scalar path.
            bad |= 0b111u64 << i;
            continue;
        }
        let k = match b1 | 0x20 {
            b's' | b'd' | b'm' | b't' => 2,
            b'l' if bytes[scan + i + 2] | 0x20 == b'l' => 3,
            b'v' if bytes[scan + i + 2] | 0x20 == b'e' => 3,
            b'r' if bytes[scan + i + 2] | 0x20 == b'e' => 3,
            _ => 0,
        };
        if k != 0 {
            boundary &= !(1u64 << (i + 1));
            boundary |= 1u64 << (i + k);
        }
    }

    (boundary & !bad, bad)
}

#[cfg(test)]
mod tests {
    use crate::pretokenize::fast::cl100k::Cl100kScheme;
    use crate::pretokenize::fast::olmo3::Olmo3Scheme;
    use crate::pretokenize::fast::qwen2::Qwen2Scheme;
    use crate::pretokenize::fast::qwen3_5::Qwen35Scheme;
    use crate::pretokenize::fast::test_support::*;

    #[track_caller]
    fn check_all(buf: &[u8]) {
        check_scalar_vs_mask::<Olmo3Scheme>(buf, "olmo3");
        check_scalar_vs_mask::<Cl100kScheme>(buf, "cl100k");
        check_scalar_vs_mask::<Qwen2Scheme>(buf, "qwen2");
        check_scalar_vs_mask::<Qwen35Scheme>(buf, "qwen3_5");
    }

    fn check_streaming_all(bytes: &[u8]) {
        check_streaming::<Olmo3Scheme>(bytes, "olmo3");
        check_streaming::<Cl100kScheme>(bytes, "cl100k");
        check_streaming::<Qwen2Scheme>(bytes, "qwen2");
        check_streaming::<Qwen35Scheme>(bytes, "qwen3_5");
    }

    /// The batch classifier must engage on plain ASCII (real token starts,
    /// no bad zones) rather than pass via the scalar fallback.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn family_classifier_engages_on_ascii() {
        use crate::pretokenize::fast::mask::{self, MaskScheme};
        if !mask::simd_scanner_available() {
            return;
        }
        let text = b"The quick brown fox jumps over the lazy dog while 42 geese watch on quietly";
        let (usable, bad) = Cl100kScheme::batch_masks(text, 0);
        assert_eq!(bad, 0, "plain ASCII must produce no bad zones");
        assert!(usable.count_ones() >= 10, "classifier found too few boundaries");
    }

    /// Crafted cases, padded so they cross the batch (not scalar-tail) path.
    #[test]
    fn family_mask_matches_scalar_padded_cases() {
        let pad = "The quick brown fox jumps over the lazy dog again and again. ";
        let cases = [
            "January 24, 2015 and 12345678 numbers 1 22 333 4444",
            "don't DON'T they'Ll 'sound 'lx x'y '' ' \u{2019}s",
            "!hello !!hello ?!x a-b ... !!!\n\nnext",
            "tabs\tand\nnewlines\r\n mixed  \n  runs \n\n\n deep",
            "hi!\n\ndef hi !!\n\nabc \"quoted\" (paren)",
            "caf\u{e9} r\u{e9}sum\u{e9} \u{201c}word\u{201d} \u{2014}dash\u{2013} \u{00a0}nbsp",
            "\u{2003}em \u{2009}thin\u{2028}ls x\u{e9}\u{e9}y",
            "price: $5.99! 100,000.00 3.14159 2nd 3rd 4th",
            "a\u{2028}b a\u{2028}\n \n\n\t x \n\n ",
            "mixed 1\u{662}3x \u{661}\u{662}\u{663} arabic",
        ];
        for case in cases {
            for lead in [0usize, 1, 37, 63, 64, 65] {
                let mut buf = pad.as_bytes().repeat(4)[..pad.len() * 2 + lead].to_vec();
                buf.extend_from_slice(case.as_bytes());
                buf.extend_from_slice(pad.as_bytes());
                buf.extend_from_slice(case.as_bytes());
                check_all(&buf);
            }
        }
    }

    /// A multi-byte digit char straddling a batch edge, followed by ASCII
    /// digits: the `\p{N}{1,3}` phase starts at the straddling char, which
    /// the `pd` seed alone cannot see (bit 0 is a continuation byte).
    #[test]
    fn family_straddling_digit_char_phase() {
        for lead in 100..200usize {
            let mut buf = vec![b'a'; lead];
            buf.extend_from_slice("\u{662}1234".as_bytes());
            buf.extend_from_slice(&vec![b'a'; 262 - 6 - lead][..]);
            check_all(&buf);
        }
    }

    /// Differential fuzz across all four family schemes.
    #[test]
    fn family_mask_matches_scalar_fuzz() {
        let pieces: &[&str] = &[
            "a", "B", "z", "9", "0", " ", "  ", "\n", "\t", "\r\n", "\r", "'", "'s", "'LL",
            "!", ".", ",", "(", "é", "ß", "日", "🎉", "\u{00A0}", "\u{2003}", "word", "12",
            "1234", "’", "“", "”", "–", "—", "…", "\u{2009}", "\u{200B}", "\u{2028}",
            "\u{202F}", "×", "÷", "«", "µ", "café", "éé", "naïve", "Α", "а", "\n\n", "!x",
            "\tx", " x", "?!", "\u{301}", "ſ", "'ſ", "'\u{301}", "\u{661}\u{662}",
            "\u{FF11}", "क", "\u{940}", "\u{1D54F}", "€", "™", "…\u{2028}",
        ];
        let mut rng = xorshift(0x243F6A8885A308D3);
        for round in 0..3000 {
            check_all(&soup(pieces, 80 + (round % 400), &mut rng));
        }
    }

    /// 100 MB of OWT, mask vs scalar, for all four family schemes.
    #[test]
    #[ignore]
    fn family_mask_matches_scalar_owt() {
        check_streaming_all(&load_owt_prefix(100_000_000));
    }

    /// Full-OWT (~12 GB) variant; ~4 min total.
    #[test]
    #[ignore = "reads the full ~12 GB OWT file"]
    fn family_mask_matches_scalar_owt_full() {
        check_streaming_all(&load_owt_prefix(usize::MAX));
    }
}
