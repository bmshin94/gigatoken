//! Shared infrastructure for mask-scanner pretokenizers.
//!
//! A mask scanner classifies 64-byte batches with SIMD into per-byte u64
//! class masks, derives "a token starts here" bits with shifted-mask
//! algebra, and pops one bit per token. A scheme plugs in two hooks
//! ([`MaskScheme`]): `advance`, the scalar ground truth (also the no-SIMD
//! path), and `batch_masks`, the `(usable, bad)` bits of one batch, where
//! `bad` marks zones the walker re-derives through `advance`. [`MaskState`]
//! is the scheme-agnostic walker behind `Iterator::next`;
//! [`MaskState::fill_spans_two_phase`] is the chunked pull the encode loop
//! uses (same masks, harvested a chunk at a time and emitted branch-free).

use crate::pretokenize::unicode::{self, CharClass};

// Platform SIMD primitives: aarch64 NEON (always present), x86_64 AVX-512 or
// AVX2 (runtime-detected; scalar fallback otherwise).

/// Does this x86_64 CPU have the AVX-512 scanner tier (Zen 4/5, Ice Lake+)?
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx512_scanner_available() -> bool {
    // std caches the CPUID result: an atomic load + bit test after the
    // first call.
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("popcnt")
}

/// The AVX-512 scanner tier plus VBMI2 (`vpcompressb` for
/// [`flatten_bits_avx512`]); Skylake-X lacks it and stays on the plain tier.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx512_fill_available() -> bool {
    avx512_scanner_available() && std::arch::is_x86_feature_detected!("avx512vbmi2")
}

/// Does this x86_64 CPU have the AVX2 scanner tier (Haswell+, all Zen)?
/// The bit features are detected explicitly since the boundary algebra's
/// codegen relies on them.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx2_scanner_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("popcnt")
}

/// Is a SIMD mask scanner usable on this machine? When false, [`MaskState`]
/// runs every token through the scheme's scalar `advance`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
    avx512_scanner_available() || avx2_scanner_available()
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
    cfg!(target_arch = "aarch64")
}

// The x86-64 `target_feature` sets below enable the bit features (BMI1/2,
// LZCNT, POPCNT) too, so inlined boundary algebra gets tzcnt/lzcnt/blsr;
// they must stay in sync with the `*_scanner_available` checks.

/// simdjson-style movemask: 4 mask vectors (64 lanes of 0x00/0xFF) -> u64,
/// bit i = lane i.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn movemask64(
    v0: std::arch::aarch64::uint8x16_t,
    v1: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
    v3: std::arch::aarch64::uint8x16_t,
) -> u64 {
    use std::arch::aarch64::*;
    unsafe {
        const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(W.as_ptr());
        let mut a0 = vandq_u8(v0, w);
        let a1 = vandq_u8(v1, w);
        let mut a2 = vandq_u8(v2, w);
        let a3 = vandq_u8(v3, w);
        // simdjson's 4-`addp` reduction tree, pinned as asm: written with
        // `vpaddq_u8`, LLVM rewrites the adds as uzp/orr triples (9 -> 17
        // ops per call). Only lane u64 0 is read.
        core::arch::asm!(
            "addp {a0:v}.16b, {a0:v}.16b, {a1:v}.16b",
            "addp {a2:v}.16b, {a2:v}.16b, {a3:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a2:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a0:v}.16b",
            a0 = inout(vreg) a0,
            a1 = in(vreg) a1,
            a2 = inout(vreg) a2,
            a3 = in(vreg) a3,
            options(pure, nomem, nostack, preserves_flags),
        );
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a0))
    }
}

/// One u64 mask (bit i = byte scan+i) per byte predicate, for 64 bytes.
/// The working currency of scheme boundary algebra: everything after this
/// is platform-independent u64 bit math.
#[derive(Clone, Copy, Default)]
pub(crate) struct AsciiMasks {
    /// ASCII letters.
    pub l: u64,
    /// ASCII digits.
    pub d: u64,
    /// Space (0x20) only.
    pub s: u64,
    /// Non-newline ASCII whitespace: \t, \x0b, \x0c.
    pub wt: u64,
    /// Newlines: \r, \n.
    pub n: u64,
    /// Non-ASCII bytes (>= 0x80).
    pub hi: u64,
    /// ASCII apostrophes.
    pub ap: u64,
}

/// Classify `bytes[scan..scan+64]` with AVX-512 (requires
/// `scan + 64 <= bytes.len()` and a detected AVX-512 tier). One k-register
/// compare per predicate: a `__mmask64` IS the u64 the algebra wants.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
pub(crate) fn ascii_masks_avx512(bytes: &[u8], scan: usize) -> AsciiMasks {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(bytes.as_ptr().add(scan) as *const _);
        let lowered = _mm512_or_si512(v, _mm512_set1_epi8(0x20));
        let l = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(lowered, _mm512_set1_epi8(b'a' as i8)),
            _mm512_set1_epi8(25),
        );
        let d = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(b'0' as i8)),
            _mm512_set1_epi8(9),
        );
        let s = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b' ' as i8));
        let n = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\r' as i8))
            | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8));
        // \t (9), \x0b (11), \x0c (12): ascii ws minus \r\n and space.
        let wt = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(9)),
            _mm512_set1_epi8(4),
        ) & !n;
        let hi = _mm512_movepi8_mask(v) as u64;
        let ap = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\'' as i8));
        AsciiMasks { l, d, s, wt, n, hi, ap }
    }
}

/// Classify `bytes[scan..scan+64]` with AVX2 (requires
/// `scan + 64 <= bytes.len()` and a detected AVX2 tier): one compare per
/// half plus a `vpmovmskb` ladder (`x <= lim` is `min_epu8(x, lim) == x`).
///
/// `#[inline(never)]` is load-bearing: inlined, LLVM's vector combiner
/// pulls the caller's scalar boundary algebra back into the byte-vector
/// domain (vpinsrb/vpextrb ladders, measured 3.5x slower end to end).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
pub(crate) fn ascii_masks_avx2(bytes: &[u8], scan: usize) -> AsciiMasks {
    use std::arch::x86_64::*;
    unsafe {
        // Closures inherit the enclosing fn's target features.
        let le = |v: __m256i, lim: __m256i| -> __m256i {
            _mm256_cmpeq_epi8(_mm256_min_epu8(v, lim), v)
        };
        let mm = |m0: __m256i, m1: __m256i| -> u64 {
            (_mm256_movemask_epi8(m0) as u32 as u64)
                | ((_mm256_movemask_epi8(m1) as u32 as u64) << 32)
        };

        let p = bytes.as_ptr().add(scan);
        let v0 = _mm256_loadu_si256(p as *const _);
        let v1 = _mm256_loadu_si256(p.add(32) as *const _);

        let x20 = _mm256_set1_epi8(0x20);
        let ca = _mm256_set1_epi8(b'a' as i8);
        let c25 = _mm256_set1_epi8(25);
        let l = mm(
            le(_mm256_sub_epi8(_mm256_or_si256(v0, x20), ca), c25),
            le(_mm256_sub_epi8(_mm256_or_si256(v1, x20), ca), c25),
        );
        let c0 = _mm256_set1_epi8(b'0' as i8);
        let c9 = _mm256_set1_epi8(9);
        let d = mm(
            le(_mm256_sub_epi8(v0, c0), c9),
            le(_mm256_sub_epi8(v1, c0), c9),
        );
        let sp = _mm256_set1_epi8(b' ' as i8);
        let s = mm(_mm256_cmpeq_epi8(v0, sp), _mm256_cmpeq_epi8(v1, sp));
        let cr = _mm256_set1_epi8(b'\r' as i8);
        let lf = _mm256_set1_epi8(b'\n' as i8);
        let n = mm(
            _mm256_or_si256(_mm256_cmpeq_epi8(v0, cr), _mm256_cmpeq_epi8(v0, lf)),
            _mm256_or_si256(_mm256_cmpeq_epi8(v1, cr), _mm256_cmpeq_epi8(v1, lf)),
        );
        // \t (9), \x0b (11), \x0c (12): ascii ws minus \r\n and space.
        let c4 = _mm256_set1_epi8(4);
        let wt = mm(
            le(_mm256_sub_epi8(v0, c9), c4),
            le(_mm256_sub_epi8(v1, c9), c4),
        ) & !n;
        let hi = mm(v0, v1); // vpmovmskb takes the sign bit directly
        let apc = _mm256_set1_epi8(b'\'' as i8);
        let ap = mm(_mm256_cmpeq_epi8(v0, apc), _mm256_cmpeq_epi8(v1, apc));
        AsciiMasks { l, d, s, wt, n, hi, ap }
    }
}

// Bit-domain helpers (platform-independent)

/// Is the char starting at `idx` NOT whitespace (`\S` for a `(?!\S)`
/// lookahead)?
///
/// # Safety
///
/// `idx < bytes.len()`, and `idx + 4 <= bytes.len()` when `bytes[idx]` is
/// non-ASCII (the batch classifiers' `scan + 70 <= len` guard covers
/// `idx = scan + 64`).
#[inline(always)]
pub(crate) unsafe fn nn_at_full(bytes: &[u8], idx: usize) -> bool {
    use super::{decode_cp_inbounds, is_ascii_ws};
    let b = bytes[idx];
    if b < 0x80 {
        return !is_ascii_ws(b);
    }
    // SAFETY: caller guarantees idx + 4 <= len for a non-ASCII byte here
    // (this fn's contract).
    let (cp, _) = unsafe { decode_cp_inbounds(bytes, idx) };
    unicode::class_of(cp) != CharClass::Whitespace
}

/// The char containing byte `pos - 1`: its class (per the scheme's
/// codepoint classifier `class`), lead index, and end (exclusive); `end >
/// pos` iff the char straddles across `pos`. Multi-byte chars walk back
/// to their lead, which is what lets a batch after a unicode char compute
/// true carries instead of a bad zone.
///
/// # Safety
///
/// `pos > 0`, and `pos + 3 <= bytes.len()` when `bytes[pos - 1]` is
/// non-ASCII (the walk-back lead `j <= pos - 1` is decoded guardless; the
/// batch classifiers' `scan + 70 <= len` guard covers `pos <= scan + 64`).
#[inline(always)]
pub(crate) unsafe fn char_through(
    bytes: &[u8],
    pos: usize,
    class: impl Fn(u32) -> CharClass,
) -> (CharClass, usize, usize) {
    use super::{decode_cp_inbounds, is_ascii_ws, is_digit, is_letter};
    let b = bytes[pos - 1];
    if b < 0x80 {
        let cls = if is_letter(b) {
            CharClass::Letter
        } else if is_digit(b) {
            CharClass::Number
        } else if is_ascii_ws(b) {
            CharClass::Whitespace
        } else {
            CharClass::Other
        };
        return (cls, pos - 1, pos);
    }
    let mut j = pos - 1;
    while j > 0 && bytes[j] & 0xC0 == 0x80 {
        j -= 1;
    }
    // SAFETY: j < pos and pos + 3 <= len (this fn's contract), so
    // j + 4 <= len.
    let (cp, l) = unsafe { decode_cp_inbounds(bytes, j) };
    (class(cp), j, j + l)
}

/// Per-byte class masks for a batch's unicode chars, classified with the
/// packed table (`unicode::class_of`) — the same lookup the scalar paths
/// do. Every byte of a classified char carries the char's class, so
/// byte-adjacency == char-adjacency and the schemes' u64 boundary
/// algebra applies unchanged.
#[derive(Clone, Copy, Default)]
pub(crate) struct UniClasses {
    /// Letter / number / other / whitespace bytes.
    pub l: u64,
    pub n: u64,
    pub o: u64,
    pub ws: u64,
    /// Whitespace lead bits by char length, for the char-length-aware
    /// `(?!\S)` shift tests. Deferred ws chars (see `resid`) are not
    /// included.
    pub w2: u64,
    pub w3: u64,
    /// Lead bits of all classified chars by length, for schemes that
    /// shift a test by the previous char's length (the cl100k family's
    /// two-chars-back rule).
    pub lead2: u64,
    pub lead3: u64,
    pub lead4: u64,
    /// Continuation bytes of classified chars.
    pub cont: u64,
    /// Bytes only the scalar path can decide: whitespace chars straddling
    /// the batch end (their run-split bookkeeping crosses the boundary),
    /// number chars when `NUMBERS` is false, and stray continuation
    /// bytes. Class masks stay truthful for these bytes so neighbors'
    /// algebra is exact; callers turn `resid` into bad zones (±1 smear).
    pub resid: u64,
}

/// Classify every unicode char whose lead bit is in `m` for
/// `bytes[scan..scan+64]`. A char spilling off the batch end gets class
/// bits for its in-batch bytes only (the next batch's `char_through`
/// walk-back covers the rest). `NUMBERS`: false when digit grouping is
/// char-counted (`\p{N}{1,3}`), so number chars defer. `LEADS`: fill the
/// per-length lead masks (for shift-by-prev-char-length rules).
///
/// The loop stays branchy on purpose (a branchless body measured 0.986x).
///
/// # Safety
///
/// `scan + 70 <= bytes.len()`: a lead at bit 63 is decoded guardless and
/// may touch through `scan + 67`.
#[inline(always)]
pub(crate) unsafe fn classify_uni_chars<const NUMBERS: bool, const LEADS: bool>(
    bytes: &[u8],
    scan: usize,
    mut m: u64,
    class: impl Fn(u32) -> CharClass,
) -> UniClasses {
    use super::decode_cp_inbounds;
    let mut u = UniClasses::default();
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        let b = bytes[scan + i];
        if b < 0xE0 {
            // 2-byte lane (leads 0xC2..0xDF, cp < 0x800): nearly every
            // non-ASCII char in western corpora, so this branch predicts
            // taken and skips the length ladder + general decode.
            if b < 0xC2 {
                u.resid |= 1 << i; // stray continuation byte (invalid UTF-8)
                continue;
            }
            let lead = 1u64 << i;
            let chm = 3u64 << i; // in-batch bytes (excess drops at bit 63)
            // SAFETY: scan + 70 <= len (this fn's # Safety contract),
            // i <= 63, so scan + i + 1 <= scan + 64 < len.
            let b1 = unsafe { *bytes.get_unchecked(scan + i + 1) };
            let cp = ((b as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F);
            match class(cp) {
                CharClass::Letter => u.l |= chm,
                CharClass::Number => {
                    u.n |= chm;
                    if !NUMBERS {
                        u.resid |= chm;
                    }
                }
                CharClass::Other => u.o |= chm,
                CharClass::Whitespace => {
                    u.ws |= chm;
                    if i + 2 > 64 {
                        // Straddling-out ws stays a bad zone; its true
                        // class marks keep neighbors' `(?!\S)` tests
                        // exact.
                        u.resid |= chm;
                    } else {
                        u.w2 |= lead;
                    }
                }
            }
            if LEADS {
                u.lead2 |= lead;
            }
            u.cont |= chm & !lead;
            m &= !chm;
            continue;
        }
        let l = if b < 0xF0 { 3 } else { 4 };
        let chm = ((1u64 << l) - 1) << i; // in-batch bytes (excess drops)
        let lead = 1u64 << i;
        // SAFETY: scan + 70 <= len (this fn's # Safety contract), i <= 63,
        // so scan + i + 4 <= len even for a 4-byte lead at bit 63.
        let (cp, _) = unsafe { decode_cp_inbounds(bytes, scan + i) };
        match class(cp) {
            CharClass::Letter => u.l |= chm,
            CharClass::Number => {
                u.n |= chm;
                if !NUMBERS {
                    u.resid |= chm;
                }
            }
            CharClass::Other => u.o |= chm,
            CharClass::Whitespace => {
                u.ws |= chm;
                if i + l > 64 || l == 4 {
                    // Straddling-out ws (and defensively: no 4-byte cp
                    // is ws in Unicode) stays a bad zone; its true class
                    // marks keep neighbors' `(?!\S)` tests exact.
                    u.resid |= chm;
                } else {
                    u.w3 |= lead;
                }
            }
        }
        if LEADS {
            if l == 3 {
                u.lead3 |= lead;
            } else {
                u.lead4 |= lead;
            }
        }
        u.cont |= chm & !lead;
        m &= !chm;
    }
    u
}

/// Token-start bits inside ASCII digit runs for `\p{N}{1,3}`: each run
/// splits into 3-char tokens, so boundaries sit at run start + 3k. (For a
/// plain `\p{N}` scheme every digit is a start — no helper needed.)
#[inline(always)]
pub(crate) fn digit_run_splits3(d: u64) -> u64 {
    let mut b = d & !(d << 1); // run starts
    // A start at p re-arms at p+3 while the run continues: hop condition
    // c = "p..p+3 all digits". Log-doubling covers 64-bit runs in 5 steps.
    let mut c = d & (d >> 1) & (d >> 2) & (d >> 3);
    let mut sh = 3u32;
    while sh < 64 {
        b |= (b & c) << sh;
        c &= c >> sh;
        sh <<= 1;
    }
    b
}

/// Smear `seed` upward (toward higher bits) through contiguous set bits of
/// `within`, in log steps.
#[inline(always)]
pub(crate) fn smear_up(seed: u64, within: u64) -> u64 {
    let mut a = seed;
    let mut m = within;
    let mut sh = 1u32;
    while sh < 64 {
        a |= (a << sh) & m;
        m &= m << sh;
        sh <<= 1;
    }
    a
}

// The batch walker

/// The two per-scheme hooks of a mask-scanner pretokenizer.
pub(crate) trait MaskScheme {
    /// Scalar ground truth: end of the token starting at `pos`
    /// (`pos < bytes.len()`, `pos` on a token boundary).
    fn advance(bytes: &[u8], pos: usize) -> usize;

    /// `(usable, bad)` for `bytes[scan..scan+64]` (`scan+64 <= len`):
    /// `usable` bit k = trustworthy token start at scan+k; `bad` bit k =
    /// byte scan+k needs the scalar path. `usable & bad` must be 0.
    #[cfg(target_arch = "aarch64")]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64);

    /// The x86_64 batch classifier, monomorphized on the SIMD tier
    /// (`AVX512`: the AVX-512 front-end, else AVX2); same contract as the
    /// aarch64 `batch_masks`. The fill wrappers instantiate it inside a
    /// matching `#[target_feature]` region so it inlines into the fill loop.
    ///
    /// # Safety
    ///
    /// The selected tier must have been runtime-detected
    /// ([`avx512_scanner_available`] / [`avx2_scanner_available`]).
    #[cfg(target_arch = "x86_64")]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64);

    /// Runtime-dispatched form of [`Self::batch_masks_x86`] for call sites
    /// outside a tier-monomorphized region (`next_span`): a cached tier
    /// check plus a call into a per-tier `#[target_feature]` wrapper. Only
    /// valid when [`simd_scanner_available`] ([`MaskState`] guarantees it).
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64)
    where
        Self: Sized,
    {
        debug_assert!(simd_scanner_available());
        if avx512_scanner_available() {
            // SAFETY: runtime AVX-512 detection right above.
            unsafe { batch_masks_dyn_avx512::<Self>(bytes, scan) }
        } else {
            // SAFETY: MaskState enables the mask-scanner path only after
            // runtime detection (simd_scanner_available); without AVX-512
            // that detection was the AVX2 tier's.
            unsafe { batch_masks_dyn_avx2::<Self>(bytes, scan) }
        }
    }
}

/// AVX-512 feature region for the runtime-dispatched
/// `MaskScheme::batch_masks`: the scheme's `batch_masks_x86` body fuses
/// into it and gets full-tier codegen (~25% slower without this region).
///
/// # Safety
///
/// The CPU must support the AVX-512 scanner tier
/// ([`avx512_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn batch_masks_dyn_avx512<S: MaskScheme>(bytes: &[u8], scan: usize) -> (u64, u64) {
    // SAFETY: the caller detected the AVX-512 tier (fn contract).
    unsafe { S::batch_masks_x86::<true>(bytes, scan) }
}

/// AVX2 counterpart of [`batch_masks_dyn_avx512`].
///
/// # Safety
///
/// The CPU must support the AVX2 scanner tier
/// ([`avx2_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn batch_masks_dyn_avx2<S: MaskScheme>(bytes: &[u8], scan: usize) -> (u64, u64) {
    // SAFETY: the caller detected the AVX2 tier (fn contract).
    unsafe { S::batch_masks_x86::<false>(bytes, scan) }
}

/// x86 SIMD-tier selector for the monomorphized fill bodies: `DYN` keeps
/// the per-batch runtime dispatch, the others pin the tier once per fill
/// inside a matching `#[target_feature]` wrapper (`AVX512_VBMI2` differs
/// from `AVX512` only in phase A's `vpcompressb` flatten). Always `DYN`
/// off x86_64.
pub(crate) const X86_TIER_DYN: u8 = 0;
pub(crate) const X86_TIER_AVX2: u8 = 1;
pub(crate) const X86_TIER_AVX512: u8 = 2;
pub(crate) const X86_TIER_AVX512_VBMI2: u8 = 3;

/// Scheme-agnostic mask-scanner state: pops trusted boundary bits, walks
/// bad zones through the scheme's scalar `advance`, runs the buffer tail
/// scalar, and precomputes one batch ahead so the SIMD chain retires under
/// the previous batch's pops. Without SIMD support (non-aarch64/x86_64
/// targets, or an x86_64 CPU without AVX-512 or AVX2) `scalar_until`
/// starts at `usize::MAX`, so every token takes the scalar path.
pub(crate) struct MaskState {
    /// Start of the pending (not yet emitted) token.
    pub pos: usize,
    /// Base of the next batch to scan.
    scan: usize,
    /// Base the `rem`/`batch_*` bits refer to.
    mask_base: usize,
    /// Boundary bits of the current segment (trusted, pop-ready).
    rem: u64,
    /// Full usable mask of the current batch (later segments).
    batch_usable: u64,
    /// Bad zones of the current batch not yet passed.
    batch_bad: u64,
    /// Emit tokens via the scalar advance while `pos < scalar_until`.
    scalar_until: usize,
    /// Eagerly computed masks for the batch at `pre_base` (usize::MAX =
    /// none).
    pre_base: usize,
    pre_usable: u64,
    pre_bad: u64,
}

impl MaskState {
    #[inline]
    pub(crate) fn new(pos: usize) -> Self {
        let scalar_until = if simd_scanner_available() { pos } else { usize::MAX };
        Self {
            pos,
            scan: pos,
            mask_base: pos,
            rem: 0,
            batch_usable: 0,
            batch_bad: 0,
            scalar_until,
            pre_base: usize::MAX,
            pre_usable: 0,
            pre_bad: 0,
        }
    }

    /// Load the segment of `batch_usable` bits in [from_bit, next bad run)
    /// into `rem` and aim `scalar_until` past that bad run at the next
    /// trusted boundary (or the batch end).
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline(always)]
    fn load_segment(&mut self, from_bit: u32) {
        let live = u64::MAX << from_bit;
        let seg_bad = self.batch_bad & live;
        if seg_bad == 0 {
            self.rem = self.batch_usable & live;
            self.batch_bad = 0;
        } else {
            let nb = seg_bad.trailing_zeros();
            self.rem = self.batch_usable & live & ((1u64 << nb) - 1);
            let rest = self.batch_usable & (u64::MAX << nb);
            self.scalar_until = if rest != 0 {
                self.mask_base + rest.trailing_zeros() as usize
            } else {
                self.mask_base + 64
            };
        }
        // A bit at the pending token's own start is not an end. Branchless:
        // whether the pending token starts exactly at this segment's first
        // bit is a ~20% coin flip on natural text.
        let at_start = self.pos == self.mask_base + from_bit as usize;
        self.rem &= !(u64::from(at_start) << from_bit);
    }

    /// The next token's byte range, or None at end of input.
    #[inline(always)]
    pub(crate) fn next_span<S: MaskScheme>(&mut self, bytes: &[u8]) -> Option<(usize, usize)> {
        let len = bytes.len();
        loop {
            if self.rem != 0 {
                let tz = self.rem.trailing_zeros() as usize;
                let end = self.mask_base + tz;
                self.rem &= self.rem - 1;
                let start = self.pos;
                self.pos = end;
                return Some((start, end));
            }
            if self.pos < self.scalar_until {
                if self.pos >= len {
                    return None;
                }
                let start = self.pos;
                let end = S::advance(bytes, start);
                self.pos = end;
                return Some((start, end));
            }
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            {
                // Continue with the current batch's next trusted segment
                // after a scalar gap (each batch is computed exactly once).
                if self.batch_bad != 0 && self.pos < self.mask_base + 64 {
                    self.load_segment((self.pos - self.mask_base) as u32);
                    continue;
                }
                self.batch_bad = 0;
                // Resume after a scalar overrun WITHOUT leaving the 64-byte
                // grid, so the precomputed next batch stays valid; stale
                // run-internal bits below `pos` are masked by `from_bit`.
                while self.scan + 64 <= self.pos {
                    self.scan += 64;
                }
                if self.scan + 64 > len {
                    // Tail: scalar to the end of the buffer.
                    self.scalar_until = usize::MAX;
                    continue;
                }
                let (usable, bad) = if self.pre_base == self.scan {
                    (self.pre_usable, self.pre_bad)
                } else {
                    S::batch_masks(bytes, self.scan)
                };
                self.mask_base = self.scan;
                self.scan += 64;
                self.batch_usable = usable;
                self.batch_bad = bad;
                // Kick off the next batch now (dirty batches too): its SIMD
                // chain overlaps this batch's pops instead of stalling the
                // next refill.
                if self.scan + 64 <= len {
                    let (u2, b2) = S::batch_masks(bytes, self.scan);
                    self.pre_base = self.scan;
                    self.pre_usable = u2;
                    self.pre_bad = b2;
                } else {
                    self.pre_base = usize::MAX;
                }
                // An overrun may have left `pos` inside this grid batch;
                // start from its bit so stale bits below never pop. The
                // no-overrun case keeps the constant argument (and its
                // folded codegen) — schemes with few bad zones take that
                // branch essentially always.
                if self.pos > self.mask_base {
                    self.load_segment((self.pos - self.mask_base) as u32);
                } else {
                    self.load_segment(0);
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                // Unreachable: scalar_until is usize::MAX on this arch.
                self.scalar_until = usize::MAX;
            }
        }
    }
}

// Two-phase chunked span fill

/// Set-bit positions of a byte, packed in 8 u16 lanes (unused lanes 0,
/// never read). 4 KB, L1-resident alongside the unicode class table.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
static BIT_POS: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut b = 1usize;
    while b < 256 {
        let mut j = 0;
        let mut w = 0;
        while j < 8 {
            if b >> j & 1 == 1 {
                t[b][w] = j as u16;
                w += 1;
            }
            j += 1;
        }
        b += 1;
    }
    t
};

/// Append the set-bit positions of `m`, offset by `rel` (wrapping), to
/// `out[0..popcount]` with no data-dependent branch: 8 fixed iterations,
/// one unconditional 8-lane store each, so 64 bits cost the same
/// regardless of population. Scribbles up to `out[popcount + 7]`
/// (`out[0..128]` on the VBMI2 tier); callers reserve the slack. Returns
/// the popcount.
///
/// # Safety
///
/// The scribble range must be writable; with `X86_TIER =
/// X86_TIER_AVX512_VBMI2` the CPU must support AVX-512 F/BW/VBMI2 (that
/// instantiation lives only inside the `_avx512_vbmi2_crc` wrapper).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn flatten_bits<const X86_TIER: u8>(m: u64, rel: u16, out: *mut u16) -> usize {
    #[cfg(target_arch = "x86_64")]
    if X86_TIER == X86_TIER_AVX512_VBMI2 {
        // SAFETY: forwarded from this fn's contract.
        return unsafe { flatten_bits_avx512(m, rel, out) };
    }
    // Per-octet popcounts (SWAR); one multiply turns them into inclusive
    // prefix sums, and a byte shift makes them exclusive write offsets —
    // the 8 stores below are mutually independent.
    let mut x = m;
    x -= (x >> 1) & 0x5555_5555_5555_5555;
    x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
    x = (x + (x >> 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    let incl = x.wrapping_mul(0x0101_0101_0101_0101);
    let excl = incl << 8;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        for j in 0..8 {
            let b = (m >> (8 * j)) as u8 as usize;
            let w = (excl >> (8 * j)) as u8 as usize;
            let v = vld1q_u16(BIT_POS[b].as_ptr());
            let v = vaddq_u16(v, vdupq_n_u16(rel.wrapping_add(8 * j as u16)));
            vst1q_u16(out.add(w), v);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        for j in 0..8 {
            let b = (m >> (8 * j)) as u8 as usize;
            let w = (excl >> (8 * j)) as u8 as usize;
            let e = &BIT_POS[b];
            let base = rel.wrapping_add(8 * j as u16);
            // Fixed 8-lane copy: autovectorizes to one 16-byte store.
            for t in 0..8 {
                out.add(w + t).write(e[t].wrapping_add(base));
            }
        }
    }
    (incl >> 56) as usize
}

/// [`flatten_bits`] via AVX-512 VBMI2: `vpcompressb` packs the set-bit
/// positions in one op, widened to u16 plus `rel`, two unconditional
/// 64-byte stores (scribbles `out[0..128]` regardless of popcount).
///
/// # Safety
///
/// The CPU must support AVX-512 F/BW/VBMI2 and `out[0..128]` must be
/// writable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
#[inline]
unsafe fn flatten_bits_avx512(m: u64, rel: u16, out: *mut u16) -> usize {
    use std::arch::x86_64::*;
    const IOTA: [u8; 64] = {
        let mut a = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = i as u8;
            i += 1;
        }
        a
    };
    unsafe {
        let iota = _mm512_loadu_si512(IOTA.as_ptr() as *const _);
        let comp = _mm512_maskz_compress_epi8(m, iota);
        let relv = _mm512_set1_epi16(rel as i16);
        let lo = _mm512_add_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(comp)), relv);
        let hi = _mm512_add_epi16(
            _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(comp)),
            relv,
        );
        _mm512_storeu_si512(out as *mut _, lo);
        _mm512_storeu_si512(out.add(32) as *mut _, hi);
    }
    m.count_ones() as usize
}

/// [`pack_mask_halves`](crate::pretokenize::pack_mask_halves) for each
/// clamped length 1..=15 as one 16-byte row: the issue-bound phase-B loop
/// loads both halves with one `ldp` (measured; the ALU form stays for the
/// latency-bound per-span paths).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
static PACK_MASK_TABLE: [[u64; 2]; 16] = {
    let mut t = [[0u64; 2]; 16];
    let mut n = 1;
    while n <= 15 {
        let (lo, hi) = crate::pretokenize::pack_mask_halves(n);
        t[n] = [lo, hi];
        n += 1;
    }
    t
};

/// Boundary scratch of one fill: PRETOKEN_CHUNK live entries, one batch of
/// overshoot, and the widest flatten scribble (128 lanes). Worst-case
/// cursor at a flatten call: 255 at batch entry + 64 in-batch boundaries
/// = 319; 319 + 128 = 447 <= 464.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const BOUND_BUF: usize = crate::pretokenize::PRETOKEN_CHUNK + 208;

/// Boundary offsets are u16-relative to the fill base; a batch is only
/// harvested while every position it can contribute (base + 63, or the
/// tail's `len`, both < base + 64) still fits.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const REL_LIMIT: isize = u16::MAX as isize - 127;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl MaskState {
    /// Two-phase `fill_spans_keyed` body: phase A harvests one chunk's
    /// boundary positions into a flat buffer (branchless [`flatten_bits`]
    /// per clean batch, the scheme's scalar `advance` through bad zones
    /// with `next_span`'s trust rules), then phase B turns consecutive
    /// boundary pairs into batch entries in a counted loop with no
    /// data-dependent branch. Boundary sets equal `next_span`'s by
    /// construction; leftover boundaries past the chunk are discarded and
    /// `scan` rewound to the grid batch containing `pos`, so iterator and
    /// chunked pulls compose in any order. Callers must ensure
    /// [`simd_scanner_available`].
    #[inline(always)]
    pub(crate) fn fill_spans_two_phase<'a, S: MaskScheme>(
        &mut self,
        bytes: &'a [u8],
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        // Tier + hash-arm dispatch once per fill, on process-immutable
        // bits (see `fill_span_hash`). Feature detection stays here:
        // `is_x86_feature_detected!` does not const-fold inside a matching
        // `#[target_feature]` fn.
        #[cfg(target_arch = "x86_64")]
        if crate::pretokenize::crc_hash_selected() {
            if avx512_fill_available() {
                // SAFETY: `avx512_fill_available` verified the AVX-512
                // scanner tier plus VBMI2; every such CPU has SSE4.2
                // (also implied by `crc_hash_selected` above).
                return unsafe {
                    self.fill_spans_two_phase_avx512_vbmi2_crc::<S>(bytes, batch, prefetch)
                };
            }
            if avx512_scanner_available() {
                // SAFETY: AVX-512 tier + SSE4.2 detected right above.
                return unsafe {
                    self.fill_spans_two_phase_avx512_crc::<S>(bytes, batch, prefetch)
                };
            }
            if avx2_scanner_available() {
                // SAFETY: AVX2 tier + SSE4.2 detected right above.
                return unsafe {
                    self.fill_spans_two_phase_avx2_crc::<S>(bytes, batch, prefetch)
                };
            }
            // SAFETY: `crc_hash_selected` verified SSE4.2 support.
            return unsafe { self.fill_spans_two_phase_crc::<S>(bytes, batch, prefetch) };
        }
        self.fill_spans_two_phase_impl::<S, false, X86_TIER_DYN>(bytes, batch, prefetch)
    }

    /// The AVX-512 + VBMI2 tier, CRC-hash monomorphization of
    /// [`Self::fill_spans_two_phase`].
    ///
    /// # Safety
    ///
    /// The CPU must support the AVX-512 scanner tier plus VBMI2
    /// ([`avx512_fill_available`]) and SSE4.2 (`crc_hash_selected`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi1,bmi2,lzcnt,popcnt,sse4.2"
    )]
    unsafe fn fill_spans_two_phase_avx512_vbmi2_crc<'a, S: MaskScheme>(
        &mut self,
        bytes: &'a [u8],
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        self.fill_spans_two_phase_impl::<S, true, X86_TIER_AVX512_VBMI2>(bytes, batch, prefetch)
    }

    /// The AVX-512-tier, CRC-hash monomorphization of
    /// [`Self::fill_spans_two_phase`] (`sse4.2` spelled out for the
    /// `X86_CRC = true` body's contract).
    ///
    /// # Safety
    ///
    /// The CPU must support the AVX-512 scanner tier
    /// ([`avx512_scanner_available`]) and SSE4.2 (`crc_hash_selected`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt,sse4.2")]
    unsafe fn fill_spans_two_phase_avx512_crc<'a, S: MaskScheme>(
        &mut self,
        bytes: &'a [u8],
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        self.fill_spans_two_phase_impl::<S, true, X86_TIER_AVX512>(bytes, batch, prefetch)
    }

    /// The AVX2-tier, CRC-hash monomorphization of
    /// [`Self::fill_spans_two_phase`].
    ///
    /// # Safety
    ///
    /// The CPU must support the AVX2 scanner tier
    /// ([`avx2_scanner_available`]) and SSE4.2 (`crc_hash_selected`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt,sse4.2")]
    unsafe fn fill_spans_two_phase_avx2_crc<'a, S: MaskScheme>(
        &mut self,
        bytes: &'a [u8],
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        self.fill_spans_two_phase_impl::<S, true, X86_TIER_AVX2>(bytes, batch, prefetch)
    }

    /// The SSE4.2-only (CRC-hash, per-batch tier dispatch)
    /// monomorphization of [`Self::fill_spans_two_phase`]: unreachable on
    /// real hardware, kept so a CPUID-masking hypervisor cannot mix hash
    /// arms in one process.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE4.2 (`crc_hash_selected`), and
    /// [`simd_scanner_available`] must hold.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    unsafe fn fill_spans_two_phase_crc<'a, S: MaskScheme>(
        &mut self,
        bytes: &'a [u8],
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        self.fill_spans_two_phase_impl::<S, true, X86_TIER_DYN>(bytes, batch, prefetch)
    }

    /// [`Self::fill_spans_two_phase`]'s body, monomorphized on the hash
    /// arm (`X86_CRC`, see `fill_span_hash`) and the x86 SIMD tier
    /// (`X86_TIER`; SIMD instantiations are reachable only through the
    /// matching `#[target_feature]` wrappers above).
    #[inline(always)]
    fn fill_spans_two_phase_impl<'a, S: MaskScheme, const X86_CRC: bool, const X86_TIER: u8>(
        &mut self,
        bytes: &'a [u8],
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        use crate::pretokenize::{
            PRETOKEN_CHUNK, fill_span_hash, pack_pretoken_key,
        };
        debug_assert!(simd_scanner_available());
        let len = bytes.len();
        let mut pending = self.pos;
        let mut scan = self.scan;
        // Rewind onto the grid batch containing `pending` after iterator
        // pops (next_span keeps consumed batches' bits in `rem`, which
        // this path recomputes). The forward direction is normalized at
        // each refill below.
        if scan > pending {
            scan -= 64 * (scan - pending).div_ceil(64);
        }
        let mut n = 0usize;
        // Opaque table base: LLVM rematerializes the static's address as
        // an adrp+add pair inside the per-span emission loop (constant
        // addresses are "free to recompute" to the register allocator);
        // pinning it here keeps the loop at one indexed ldp per span.
        let pack_masks: *const [u64; 2] = std::hint::black_box(PACK_MASK_TABLE.as_ptr());
        // Flatten scribble bound per call: 128 lanes (VBMI2) or popcount
        // + 7 (scalar); see BOUND_BUF.
        let scribble = if X86_TIER == X86_TIER_AVX512_VBMI2 { 128 } else { 72 };

        'refill: while n < PRETOKEN_CHUNK && pending < len {
            // Skip grid batches wholly behind `pending` (a direct-emitted
            // long span or a dropped overrun end can leave `scan` far
            // back); keeps `resume - base <= 63` for every batch below.
            if pending >= scan + 64 {
                scan += 64 * ((pending - scan) / 64);
            }
            let fill_base = pending;
            let needed = PRETOKEN_CHUNK - n;
            let mut buf = [std::mem::MaybeUninit::<u16>::uninit(); BOUND_BUF];
            let bufp = buf.as_mut_ptr() as *mut u16;
            let mut nb = 0usize;
            // Boundary bits at or below `resume` are settled: the pending
            // token's own start, or stale run-internal bits behind a
            // scalar overrun (see next_span's grid-keeping comment).
            let mut resume = pending;
            let mut exhausted = false;
            // A scalar end past the u16 window: dropped and re-derived
            // next fill, unless it is the fill's first boundary (emitted
            // directly below).
            let mut overflow_end: Option<usize> = None;

            // Phase A: harvest boundary positions.
            'harvest: while nb < needed {
                if scan.wrapping_sub(fill_base) as isize > REL_LIMIT {
                    break; // re-base: offsets would leave the u16 window
                }
                if scan + 64 > len {
                    // Scalar tail to end of input.
                    let mut p = if nb > 0 {
                        fill_base + unsafe { *bufp.add(nb - 1) } as usize
                    } else {
                        fill_base
                    };
                    while p < len && nb < needed {
                        p = S::advance(bytes, p);
                        // p <= len < scan + 64, within the u16 window per
                        // the REL_LIMIT check above.
                        unsafe { bufp.add(nb).write((p - fill_base) as u16) };
                        nb += 1;
                    }
                    exhausted = p >= len;
                    break;
                }
                let base = scan;
                #[cfg(target_arch = "x86_64")]
                let (usable, bad) = match X86_TIER {
                    // SAFETY: the tier wrappers instantiate these arms
                    // only after runtime tier detection (see
                    // `fill_spans_two_phase`).
                    // The VBMI2 tier runs the same AVX-512 classifiers;
                    // it only diverges in the flatten below.
                    X86_TIER_AVX512 | X86_TIER_AVX512_VBMI2 => unsafe {
                        S::batch_masks_x86::<true>(bytes, base)
                    },
                    X86_TIER_AVX2 => unsafe { S::batch_masks_x86::<false>(bytes, base) },
                    _ => S::batch_masks(bytes, base),
                };
                #[cfg(not(target_arch = "x86_64"))]
                let (usable, bad) = S::batch_masks(bytes, base);
                // At a resume point r (the pending token's start): usable
                // bits at or below r are dead (the pending start itself,
                // or stale run-internal bits behind a scalar overrun), but
                // a bad bit AT r must stay live — load_segment's
                // `live = MAX << from_bit` plus the at_start clear. A zone
                // starting exactly at r has to route r through the scalar
                // path, or the stale post-zone usable bit would be
                // trusted. Only the fill's first batch and post-overrun
                // batches have such bits.
                let (mut ulive, mut blive) = if resume >= base {
                    debug_assert!(resume - base < 64);
                    let k = resume - base;
                    ((u64::MAX << k) << 1, u64::MAX << k)
                } else {
                    (u64::MAX, u64::MAX)
                };
                let rel = base.wrapping_sub(fill_base) as u16;
                if bad & blive == 0 {
                    debug_assert!(nb + scribble <= BOUND_BUF);
                    nb += unsafe { flatten_bits::<X86_TIER>(usable & ulive, rel, bufp.add(nb)) };
                    scan = base + 64;
                    continue;
                }
                // Dirty batch: per segment, trusted prefix bits then the
                // scheme's scalar advance through the zone up to the next
                // trusted boundary — load_segment's rules, emitting into
                // the buffer.
                loop {
                    let seg_bad = bad & blive;
                    if seg_bad == 0 {
                        debug_assert!(nb + scribble <= BOUND_BUF);
                        nb += unsafe { flatten_bits::<X86_TIER>(usable & ulive, rel, bufp.add(nb)) };
                        scan = base + 64;
                        break;
                    }
                    let fb = seg_bad.trailing_zeros();
                    let prefix = usable & ulive & !(u64::MAX << fb);
                    debug_assert!(nb + scribble <= BOUND_BUF);
                    nb += unsafe { flatten_bits::<X86_TIER>(prefix, rel, bufp.add(nb)) };
                    let mut p = if nb > 0 {
                        fill_base + unsafe { *bufp.add(nb - 1) } as usize
                    } else {
                        fill_base
                    };
                    let rest = usable & (u64::MAX << fb);
                    let until = if rest != 0 {
                        base + rest.trailing_zeros() as usize
                    } else {
                        base + 64
                    };
                    // until <= base + 64 <= len, so `advance` stays in
                    // bounds; it may overrun `until` and the batch end.
                    while p < until {
                        p = S::advance(bytes, p);
                        let relp = p - fill_base;
                        if relp > u16::MAX as usize {
                            overflow_end = Some(p);
                            break 'harvest;
                        }
                        debug_assert!(nb < BOUND_BUF);
                        unsafe { bufp.add(nb).write(relp as u16) };
                        nb += 1;
                    }
                    if p >= base + 64 {
                        // Overrun past the batch: stay on the grid and
                        // resume in the batch containing p, bits at or
                        // below p masked (they can be stale run-internal
                        // bits, exactly as in next_span).
                        scan = base + 64 * ((p - base) / 64);
                        resume = p;
                        break;
                    }
                    // Resume inside the batch at p: same at-start/bad-bit
                    // split as the batch-entry masks above.
                    blive = u64::MAX << (p - base);
                    ulive = blive << 1;
                }
            }

            if nb == 0 {
                // No boundary inside the u16 window: a > 65 KB pretoken.
                // Emit it alone through the careful pack.
                debug_assert!(!exhausted);
                let end = overflow_end.unwrap_or_else(|| S::advance(bytes, fill_base));
                let span = &bytes[fill_base..end];
                let (key, h) = match pack_pretoken_key(span) {
                    Some(key) => (key, fill_span_hash::<X86_CRC>(key)),
                    None => (0, 0),
                };
                prefetch(h);
                let meta = if key != 0 { h } else { span.len() as u64 };
                batch.entries[n] = crate::pretokenize::BatchEntry {
                    key,
                    ptr: span.as_ptr(),
                    meta,
                };
                n += 1;
                pending = end;
                continue 'refill;
            }

            // Phase B: flat emission with no data-dependent branch. One
            // hoisted check proves every 16-byte key load in-bounds of the
            // input slice; only a fill reaching within 16 bytes of EOF
            // routes through the careful per-span pack.
            let emit_n = nb.min(needed);
            let last_end = unsafe { *bufp.add(emit_n - 1) } as usize;
            let entries = &mut batch.entries[n..n + emit_n];
            let base_ptr = unsafe { bytes.as_ptr().add(fill_base) };
            // `prev`/`end` in usize: the u16 domain costs masks and a
            // duplicated compare per span.
            let mut prev = 0usize;
            if fill_base + last_end + 16 <= len {
                // Every x86 tier shares this scalar key pack: an AVX-512
                // masked pack measured -36% (its kmask serializes on the
                // boundary chain). Do not re-try.
                for (i, e) in entries.iter_mut().enumerate() {
                    let end = unsafe { *bufp.add(i) } as usize;
                    let tok_len = end - prev;
                    let p = unsafe { base_ptr.add(prev) };
                    prev = end;
                    // SAFETY: p + 16 <= base_ptr + last_end + 16 <= end of
                    // the input slice (hoisted check above).
                    let raw = unsafe { (p as *const u128).read_unaligned() };
                    // Branchless pack_pretoken_key: one ldp from
                    // PACK_MASK_TABLE instead of the 7-op per-half ALU
                    // chain (see the table's docs). tok_len >= 1
                    // (boundaries are strictly increasing), so the clamped
                    // length is in the table's 1..=15 domain. Long spans
                    // take key 0 (pretoken_key_hash(0) == 0) through the
                    // `keep` AND-mask — an if/select here gets if-converted
                    // into a real branch (LLVM hoists it to skip the two
                    // loads), reintroducing the pattern-free n > 15 branch
                    // this loop exists to avoid.
                    let m = tok_len.min(15);
                    // SAFETY: m <= 15, in the 16-entry table.
                    let [mask_lo, mask_hi] = unsafe { *pack_masks.add(m) };
                    let keep = ((tok_len <= 15) as u64).wrapping_neg();
                    let klo = (raw as u64) & mask_lo & keep;
                    let khi = (((raw >> 64) as u64 & mask_hi) | ((m as u64) << 56)) & keep;
                    let key = (klo as u128) | ((khi as u128) << 64);
                    let hv = fill_span_hash::<X86_CRC>(key);
                    prefetch(hv);
                    // meta = hash for short spans, length for long ones
                    // (see `BatchEntry::meta`), in the same AND-mask style
                    // as the key routing — a select gets if-converted.
                    let meta = (hv & keep) | (tok_len as u64 & !keep);
                    e.key = key;
                    e.ptr = p;
                    e.meta = meta;
                }
            } else {
                for (i, e) in entries.iter_mut().enumerate() {
                    let end = unsafe { *bufp.add(i) } as usize;
                    let tok_len = end - prev;
                    let p = unsafe { base_ptr.add(prev) };
                    prev = end;
                    // SAFETY: as above for the span bounds.
                    let span = unsafe { std::slice::from_raw_parts(p, tok_len) };
                    let (key, hv) = match pack_pretoken_key(span) {
                        Some(key) => (key, fill_span_hash::<X86_CRC>(key)),
                        None => (0, 0),
                    };
                    prefetch(hv);
                    let meta = if key != 0 { hv } else { tok_len as u64 };
                    e.key = key;
                    e.ptr = p;
                    e.meta = meta;
                }
            }
            n += emit_n;
            pending = fill_base + prev;
            if exhausted {
                debug_assert_eq!(pending, len);
                break;
            }
        }

        // Rewind past discarded leftover boundaries and leave the state as
        // a fresh resume at `pending` for either pull style.
        if scan > pending {
            scan -= 64 * (scan - pending).div_ceil(64);
        }
        self.pos = pending;
        self.scan = scan;
        self.mask_base = scan;
        self.rem = 0;
        self.batch_usable = 0;
        self.batch_bad = 0;
        self.scalar_until = pending;
        self.pre_base = usize::MAX;
        n
    }
}
