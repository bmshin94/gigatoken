//! Fast pretokenizer for the GPT-2 (r50k_base) regex:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! With SIMD support (aarch64 NEON, x86_64 AVX-512/AVX2) iteration runs a
//! mask scanner: 64-byte batches classified into per-byte u64 class masks,
//! boundary bits from shifted-mask algebra, one bit popped per token. The
//! scalar `advance_pos` is the reference implementation, the no-SIMD
//! fallback, and the executor for bad zones and buffer tails. It is a
//! free function rather than a `&mut self` method: keeping the cursor in a
//! register instead of `self.pos` is worth ~30% throughput.

use super::mask::{self, MaskScheme};
use super::{
    decode_cp, is_ascii_ws, is_digit, is_letter, scan_digits_from, scan_letters_from,
    scan_other_from,
};
use crate::pretokenize::unicode::{self, CharClass};

/// Boundary and bad-zone bitmasks for `bytes[scan..scan+64]` (requires
/// `scan + 64 <= bytes.len()`): bit `k` of `usable` = a trustworthy token
/// start at `scan + k`; `bad` marks bytes whose boundaries `advance_pos`
/// must re-derive. Batches with any non-ASCII byte take [`extended_masks`]
/// (`#[inline(never)]`, measured: keeps the walker's registers clean).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
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
            hiv[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            apv[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
        }

        let lb = mask::movemask64(lv[0], lv[1], lv[2], lv[3]);
        let db = mask::movemask64(dv[0], dv[1], dv[2], dv[3]);
        let s64 = mask::movemask64(sv[0], sv[1], sv[2], sv[3]);
        let wsa = mask::movemask64(wsv[0], wsv[1], wsv[2], wsv[3]);
        // Apostrophes only matter for the contraction fixup below.
        let ap_any = vorrq_u8(vorrq_u8(apv[0], apv[1]), vorrq_u8(apv[2], apv[3]));
        let ap64 = if vmaxvq_u8(ap_any) != 0 {
            mask::movemask64(apv[0], apv[1], apv[2], apv[3])
        } else {
            0
        };

        // Any non-ASCII byte routes to the extended classifier, which
        // reuses the ASCII masks computed above.
        let hi_any = vorrq_u8(vorrq_u8(hiv[0], hiv[1]), vorrq_u8(hiv[2], hiv[3]));
        if vmaxvq_u8(hi_any) != 0 {
            let hi64 = mask::movemask64(hiv[0], hiv[1], hiv[2], hiv[3]);
            return extended_masks(bytes, scan, lb, db, s64, wsa, hi64, ap64);
        }

        ascii_batch_algebra(bytes, scan, lb, db, s64, wsa, ap64)
    }
}

/// x86 counterpart of the NEON `batch_masks`, monomorphized on the SIMD
/// tier ([`mask::ascii_masks_avx512`] / [`mask::ascii_masks_avx2`]); the
/// boundary algebra and the extended path are shared. `#[inline(always)]`
/// with no `target_feature` of its own, so the body fuses into whichever
/// feature region calls it.
///
/// # Safety
///
/// The selected tier must have been runtime-detected
/// ([`mask::avx512_scanner_available`] /
/// [`mask::avx2_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
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
    let wsa = am.s | am.wt | am.n;
    if am.hi != 0 {
        // SAFETY: both detected tiers include the BMI1/BMI2/LZCNT/POPCNT
        // bit features `extended_masks` re-declares (fn contract).
        return unsafe { extended_masks(bytes, scan, am.l, am.d, am.s, wsa, am.hi, am.ap) };
    }
    ascii_batch_algebra(bytes, scan, am.l, am.d, am.s, wsa, am.ap)
}

/// Pure-ASCII boundary algebra shared by the NEON and AVX-512 batch
/// classifiers (the batch has no non-ASCII byte; `wsa` = all ASCII
/// whitespace). Everything here is platform-independent u64 bit math.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn ascii_batch_algebra(
    bytes: &[u8],
    scan: usize,
    lb: u64,
    db: u64,
    s64: u64,
    wsa: u64,
    ap64: u64,
) -> (u64, u64) {
    let ob = !(lb | db | wsa); // hi == 0 on this path

    // Bit-0 carries from the char before the batch. This batch is
    // pure ASCII, so a multi-byte prev char always ends exactly at
    // the boundary and the walk-back gives true carries — no bad
    // zone.
    let (pl, pd, ps, pws, po) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else {
        carries_at(bytes, scan)
    };

    let cont_same =
        (lb & ((lb << 1) | pl)) | (db & ((db << 1) | pd)) | (ob & ((ob << 1) | po));
    let after_sp = (s64 << 1) | ps;
    let nb = !wsa & !cont_same & !after_sp;

    // Ws-run split (`\s+(?!\S)`); bit 63 needs the real lookahead
    // char. The ASCII case is branchless — "is byte 63 ws" is a
    // ~20% coin flip on natural text, so testing it costs a
    // mispredict every few batches. Only a non-ASCII lookahead
    // byte (rare) branches, for the table-backed ws check.
    let mut split_ok = wsa & (!wsa >> 1); // bit 63: shifted-in 0
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    if nb64 < 0x80 {
        split_ok |= (u64::from(!is_ascii_ws(nb64)) << 63) & wsa;
    } else if wsa >> 63 != 0
        // SAFETY: this classifier's scan + 70 <= len batch guard puts the
        // decode at scan + 64 in bounds (needs scan + 68 <= len).
        && unsafe { mask::nn_at_full(bytes, scan + 64) }
    {
        split_ok |= 1 << 63;
    }
    let pwsb = (wsa << 1) | pws;
    let wsboundary = wsa & (!pwsb | split_ok);
    let mut boundary = nb | wsboundary;

    let mut bad = 0u64;

    // Contraction fixup (see extended_masks for the rules).
    if ap64 != 0 {
        let mut cand = ap64 & boundary;
        while cand != 0 {
            let i = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            if i >= 61 {
                bad |= u64::MAX << i;
                break;
            }
            let k = match bytes[scan + i + 1] {
                b's' | b'd' | b'm' | b't' => 2,
                b'l' if bytes[scan + i + 2] == b'l' => 3,
                b'v' if bytes[scan + i + 2] == b'e' => 3,
                b'r' if bytes[scan + i + 2] == b'e' => 3,
                _ => 0,
            };
            if k != 0 {
                boundary &= !(1u64 << (i + 1));
                boundary |= 1u64 << (i + k);
            }
        }
    }
    (boundary & !bad, bad)
}

/// Slow(er) path for batches containing non-ASCII: every unicode char in
/// (or straddling into) the batch is classified with the packed table
/// ([`mask::classify_uni_chars`]) and joins the per-byte class masks, so
/// the u64 boundary algebra applies unchanged. Bad zones remain only for
/// whitespace straddling a batch edge, stray continuation bytes, and
/// contractions at the batch edge. `#[inline(never)]` keeps the walker's
/// register allocation clean; the x86 `target_feature` keeps the bit scans
/// on tzcnt/lzcnt/blsr in a baseline build (see `mask.rs`).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "bmi1,bmi2,lzcnt,popcnt")
)]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn extended_masks(
    bytes: &[u8],
    scan: usize,
    l64: u64,
    d64: u64,
    s64: u64,
    ws64: u64,
    hi64: u64,
    ap64: u64,
) -> (u64, u64) {
    let wsa = ws64;

    // Class-table LazyLock resolved once; the per-char classify below is
    // then a bare slice index.
    let ct = unicode::ClassTable::<false>::get();
    let class = move |cp| ct.class_of(cp);

    // Bit-0 carries via the prev-char walk-back; a char straddling into
    // this batch claims its continuation bytes with its class.
    let mut claim = mask::UniClasses::default();
    let (pl, pd, ps, pws, po) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else if bytes[scan - 1] < 0x80 {
        carries_at(bytes, scan)
    } else {
        // SAFETY: scan > 0 on this branch, and the classifier's
        // scan + 70 <= len batch guard covers pos + 3 <= len.
        let (cls, _lead, end) = unsafe { mask::char_through(bytes, scan, class) };
        let chm = if end > scan { (1u64 << (end - scan)) - 1 } else { 0 };
        claim.cont = chm;
        match cls {
            CharClass::Letter => {
                claim.l = chm;
                (1, 0, 0, 0, 0)
            }
            CharClass::Number => {
                claim.n = chm;
                (0, 1, 0, 0, 0)
            }
            CharClass::Other => {
                claim.o = chm;
                (0, 0, 0, 0, 1)
            }
            CharClass::Whitespace => {
                // A ws char straddling in defers to the scalar path (its
                // run-split bookkeeping needs the pre-batch extent) but
                // still marks its true class for neighbors' algebra.
                claim.ws = chm;
                claim.resid = chm;
                (0, 0, u64::from(bytes[scan - 1] == b' '), 1, 0)
            }
        }
    };

    // SAFETY: this classifier's scan + 70 <= len batch guard is exactly
    // `classify_uni_chars`' contract.
    let uni = unsafe {
        mask::classify_uni_chars::<true, false>(bytes, scan, hi64 & !claim.cont, class)
    };

    // Effective per-byte classes: every byte of a classified char carries
    // the char's class, so the same algebra as the pure-ASCII path
    // applies.
    let lb = l64 | claim.l | uni.l;
    let db = d64 | claim.n | uni.n;
    let wsb = wsa | claim.ws | uni.ws;
    let ob = !(l64 | d64 | wsa | hi64) | claim.o | uni.o;
    let contm = claim.cont | uni.cont;
    let resid = claim.resid | uni.resid;

    let cont_same =
        (lb & ((lb << 1) | pl)) | (db & ((db << 1) | pd)) | (ob & ((ob << 1) | po));
    let after_sp = (s64 << 1) | ps;
    let nb = !wsb & !cont_same & !after_sp & !contm;

    // Ws-run split: char-length-aware "followed by non-ws" test. All ws
    // chars whose lookahead crosses the batch edge look at byte 64: an
    // ASCII ws at 63, a 2-byte ws led at 62, a 3-byte ws led at 61
    // (later leads straddle out and are already bad zones). The ASCII
    // case is branchless as in the fast path; multi-byte edge leads are
    // rare enough to branch.
    let nn = !wsb;
    let mut split_ok = (wsa & (nn >> 1)) | (uni.w2 & (nn >> 2)) | (uni.w3 & (nn >> 3));
    let ws_leads = wsa | uni.w2 | uni.w3;
    let edge_mb = (uni.w2 & (1 << 62)) | (uni.w3 & (1 << 61));
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    if nb64 < 0x80 && edge_mb == 0 {
        split_ok = (split_ok & !(1 << 63)) | ((u64::from(!is_ascii_ws(nb64)) << 63) & wsa);
    } else {
        let edge = edge_mb | ((1 << 63) & wsa);
        if edge != 0 {
            // SAFETY: this classifier's scan + 70 <= len batch guard puts
            // the decode at scan + 64 in bounds (needs scan + 68 <= len).
            if unsafe { mask::nn_at_full(bytes, scan + 64) } {
                split_ok |= edge;
            } else {
                split_ok &= !edge;
            }
        }
    }
    let pwsb = (wsb << 1) | pws;
    let wsboundary = ws_leads & (!pwsb | split_ok);
    let mut boundary = nb | wsboundary;

    let mut bad = resid | resid << 1 | resid >> 1;

    // Contraction fixup: an apostrophe at a token start absorbs an
    // s/d/m/t/ll/ve/re suffix. One that could reach past bit 63 defers
    // to the scalar path (the next batch cannot see the moved boundary).
    let mut cand = ap64 & boundary & !bad;
    while cand != 0 {
        let i = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        if i >= 61 {
            bad |= u64::MAX << i;
            break;
        }
        let k = match bytes[scan + i + 1] {
            b's' | b'd' | b'm' | b't' => 2,
            b'l' if bytes[scan + i + 2] == b'l' => 3,
            b'v' if bytes[scan + i + 2] == b'e' => 3,
            b'r' if bytes[scan + i + 2] == b'e' => 3,
            _ => 0,
        };
        if k != 0 {
            boundary &= !(1u64 << (i + 1));
            boundary |= 1u64 << (i + k);
        }
    }

    (boundary & !bad, bad)
}

/// `(pl, pd, ps, pws, po)` boundary carries for the char ending at
/// `scan - 1` (`scan > 0`), multi-byte aware via [`mask::char_through`];
/// `ps` (the ` ?` absorb) is ASCII 0x20 only. The ASCII case is branchless
/// on purpose (a class if-chain mispredicts per batch).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn carries_at(bytes: &[u8], scan: usize) -> (u64, u64, u64, u64, u64) {
    let b = bytes[scan - 1];
    if b < 0x80 {
        let (l, d, w) = (is_letter(b), is_digit(b), is_ascii_ws(b));
        let bit = |c: bool| u64::from(c);
        return (bit(l), bit(d), bit(b == b' '), bit(w), bit(!l && !d && !w));
    }
    // SAFETY: scan > 0 (this fn's caller contract), and the calling batch
    // classifier's scan + 70 <= len guard covers pos + 3 <= len.
    match unsafe { mask::char_through(bytes, scan, unicode::class_of) }.0 {
        CharClass::Letter => (1, 0, 0, 0, 0),
        CharClass::Number => (0, 1, 0, 0, 0),
        CharClass::Whitespace => (0, 0, 0, 1, 0),
        CharClass::Other => (0, 0, 0, 0, 1),
    }
}

pub(crate) struct R50kScheme;

impl MaskScheme for R50kScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance_pos(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        batch_masks(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { batch_masks_x86::<AVX512>(bytes, scan) }
    }
}

super::define_mask_pretokenizer!(FastR50kPretokenizer, R50kScheme);

/// Advance past one token starting at `start` (`start < bytes.len()`, a
/// token boundary); returns the token's end. Comparison chains rather
/// than a LUT; the byte loads issue in parallel under speculation (a
/// single-load variant measured 0.84x).
#[inline(always)]
fn advance_pos(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let b0 = unsafe { *bytes.get_unchecked(start) };

    // Bare ASCII letter start (~5% of OWT tokens; most words carry a space)
    if is_letter(b0) {
        return scan_letters_from(bytes, start + 1);
    }

    // Hot path: space before content (~78% of tokens, ~75% space+letters)
    if b0 == b' ' {
        if start + 1 < len {
            let b1 = unsafe { *bytes.get_unchecked(start + 1) };
            if is_letter(b1) {
                return scan_letters_from(bytes, start + 2);
            }
            if is_digit(b1) {
                return scan_digits_from(bytes, start + 2);
            }
            if b1 >= 0x80 {
                let (cp, l) = unsafe { decode_cp(bytes, start + 1) };
                let p = start + 1 + l;
                return match unicode::class_of(cp) {
                    CharClass::Letter => scan_letters_from(bytes, p),
                    CharClass::Number => scan_digits_from(bytes, p),
                    CharClass::Whitespace => advance_ws(bytes, p, start),
                    CharClass::Other => scan_other_from(bytes, p),
                };
            }
            if is_ascii_ws(b1) {
                return advance_ws(bytes, start + 1, start);
            }
            return scan_other_from(bytes, start + 2);
        }
        return start + 1;
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, start) };
        let p = start + l;
        return match unicode::class_of(cp) {
            CharClass::Letter => scan_letters_from(bytes, p),
            CharClass::Number => scan_digits_from(bytes, p),
            CharClass::Whitespace => advance_ws(bytes, p, start),
            CharClass::Other => scan_other_from(bytes, p),
        };
    }

    // Digit
    if is_digit(b0) {
        return scan_digits_from(bytes, start + 1);
    }

    // Apostrophe / contraction
    if b0 == b'\'' {
        match bytes.get(start + 1) {
            Some(b's' | b'd' | b'm' | b't') => return start + 2,
            Some(b'l') if bytes.get(start + 2) == Some(&b'l') => return start + 3,
            Some(b'v') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            Some(b'r') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            _ => return scan_other_from(bytes, start + 1),
        }
    }

    // Whitespace (tab, newline, etc.)
    if b0.wrapping_sub(9) < 5 {
        return advance_ws(bytes, start + 1, start);
    }

    // Other (punctuation, symbols)
    scan_other_from(bytes, start + 1)
}

/// Advance through whitespace. `scan_pos` is where to continue scanning,
/// `token_start` is where the token began (for the split-off-last-char logic).
#[inline(always)]
fn advance_ws(bytes: &[u8], scan_pos: usize, token_start: usize) -> usize {
    let len = bytes.len();
    let mut p = scan_pos;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if is_ascii_ws(b) {
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == CharClass::Whitespace {
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if p < len {
        let ws_bytes = p - token_start;
        if ws_bytes >= 2 {
            let mut last = p - 1;
            while last > token_start && unsafe { *bytes.get_unchecked(last) } & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > token_start {
                return last;
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::fast::test_support::*;

    /// The batch classifier must engage on any machine with the assumed
    /// feature sets: plain ASCII text must yield real token starts and no
    /// bad zones (guards against silently passing via the scalar fallback).
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn batch_classifier_engages_on_ascii() {
        if !mask::simd_scanner_available() {
            eprintln!("no SIMD scanner on this CPU; skipping");
            return;
        }
        // Longer than scan + 70: the classifier needs byte-64+ lookahead.
        let text = b"The quick brown fox jumps over the lazy dog while 42 geese watch on quietly";
        let (usable, bad) = R50kScheme::batch_masks(text, 0);
        assert_eq!(bad, 0, "plain ASCII must produce no bad zones");
        let mut starts = vec![];
        let mut p = 0;
        while p < text.len() {
            starts.push(p);
            p = advance_pos(text, p);
        }
        for i in 0..64usize {
            if usable >> i & 1 == 1 {
                assert!(starts.contains(&i), "usable bit {i} is not a token start");
            }
        }
        assert!(usable.count_ones() >= 10, "classifier found too few boundaries");
    }

    /// Mask scanner vs scalar `advance_pos` on crafted edge cases.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn mask_iter_matches_shipped_edge_cases() {
        let cases: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b" ".to_vec(),
            b"a".to_vec(),
            b"hello world".to_vec(),
            b"  double  spaces  ".to_vec(),
            b"a\n\nb".to_vec(),
            b"a \n b".to_vec(),
            b"tabs\tand\nnewlines\r\n end".to_vec(),
            b"don't can't we'll they've you're I'm he's 'tis 'twas".to_vec(),
            b"DON'T CAN'T 'S 'LL".to_vec(),
            b"x'y z' 'a '' ' ".to_vec(),
            b"3.14 100,000 2nd a1b2".to_vec(),
            b"!!! ?! #hashtag @user (paren) [brack]".to_vec(),
            "café résumé naïve".as_bytes().to_vec(),
            "日本語のテキスト and English".as_bytes().to_vec(),
            "space\u{00A0}nbsp \u{00A0} runs".as_bytes().to_vec(),
            "emoji 🎉🎊 mix".as_bytes().to_vec(),
            "µ§±².5 ×÷".as_bytes().to_vec(),
            b"ws at end   ".to_vec(),
            b"   ws at start".to_vec(),
            // Exactly chunk-sized and chunk-straddling patterns.
            b"abcdefghijklmnop".to_vec(),
            b"abcdefghijklmno ".to_vec(),
            b"abcdefghijklmn 'll xyz".to_vec(),
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec(),
            b"a b c d e f g h i j k l m n o p q r s t u v w x".to_vec(),
            [b"word ".repeat(10), b"\xE2\x80\x82ws".to_vec()].concat(),
        ];
        for case in &cases {
            check_scalar_vs_mask::<R50kScheme>(case, "r50k");
        }
    }

    /// Differential fuzz: random mixes of letters, digits, ws, punctuation,
    /// apostrophes, and multi-byte UTF-8 at every length 0..~200.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn mask_iter_matches_shipped_fuzz() {
        let pieces: &[&str] = &[
            "a", "B", "z", "9", "0", " ", "  ", "\n", "\t", "\r\n", "'", "'s", "'ll", "'re",
            "!", ".", ",", "(", "é", "ß", "日", "🎉", "\u{00A0}", "\u{2003}", "word", "12",
            "’", "’s", "“", "”", "–", "—", "…", "\u{2009}", "\u{200B}", "\u{2028}",
            "\u{202F}", "×", "÷", "«", "µ", "café", "éé", "naïve", "Α", "а", "ſ", "'ſ",
            "\u{661}\u{662}", "\u{FF11}", "क", "\u{940}", "\u{1D54F}", "€", "™", "\u{301}",
        ];
        let mut rng = xorshift(0x243F6A8885A308D3);
        for round in 0..4000 {
            let buf = soup(pieces, (round % 200) + 1, &mut rng);
            check_scalar_vs_mask::<R50kScheme>(&buf, "r50k");
        }
    }

    /// Differential check on real OWT (100 MB), token for token.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    #[ignore]
    fn mask_iter_matches_shipped_owt() {
        check_streaming::<R50kScheme>(&load_owt_prefix(100_000_000), "r50k");
    }

    /// Differential check on the FULL OWT file (~12 GB), token for token.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    #[ignore = "reads the full ~12 GB OWT file"]
    fn mask_iter_matches_shipped_owt_full() {
        check_streaming::<R50kScheme>(&load_owt_prefix(usize::MAX), "r50k");
    }

    fn drive(bytes: &[u8], f: impl Fn(&[u8], usize) -> usize) -> (usize, u64) {
        let mut pos = 0usize;
        let mut n = 0usize;
        let mut acc = 0u64;
        while pos < bytes.len() {
            let end = f(bytes, pos);
            acc = acc.wrapping_add(end as u64);
            n += 1;
            pos = end;
        }
        (n, acc)
    }

    /// Interleaved same-binary A/B harness (min-of-7): swap an experimental
    /// `advance_pos` candidate in below and run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn ab_r50k_advance_interleaved() {
        let input = load_owt_prefix(100_000_000);
        let mb = input.len() as f64 / 1e6;
        eprintln!("input: {mb:.1} MB");

        // Experimental candidate under test; placeholder = shipped impl.
        let experimental = |b: &[u8], s: usize| advance_pos(b, s);

        // Warmup + equivalence check
        let (n_a, acc_a) = drive(&input, advance_pos);
        let (n_b, acc_b) = drive(&input, experimental);
        assert_eq!((n_a, acc_a), (n_b, acc_b), "variants disagree");

        let mut best_a = f64::INFINITY;
        let mut best_b = f64::INFINITY;
        for round in 0..7 {
            let t = std::time::Instant::now();
            let r = drive(&input, advance_pos);
            let da = t.elapsed().as_secs_f64();
            std::hint::black_box(r);

            let t = std::time::Instant::now();
            let r = drive(&input, experimental);
            let db = t.elapsed().as_secs_f64();
            std::hint::black_box(r);

            best_a = best_a.min(da);
            best_b = best_b.min(db);
            eprintln!(
                "round {round}: shipped {:.0} MB/s | experimental {:.0} MB/s",
                mb / da,
                mb / db
            );
        }
        eprintln!(
            "best: shipped {:.0} MB/s | experimental {:.0} MB/s | ratio {:.3}x",
            mb / best_a,
            mb / best_b,
            best_a / best_b
        );
    }
}
