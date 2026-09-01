//! Open-addressing cache for short (≤ 15 byte) pretoken encodings. The
//! table far exceeds L2/L3, so a probe in the Zipf tail is a DRAM access:
//!
//! - 32-byte self-contained entries in line-aligned pairs, so a probe
//!   touches exactly one line, staged by [`Self::prefetch_l2`] a chunk
//!   ahead and [`ProbeView::prefetch`] a few probes ahead.
//! - Values hold up to four tokens inline (`tiktoken::pack_val_inline`),
//!   covering ~98% of pretokens with no second access into the arena.
//! - Backing memory is 2 MiB-aligned and `MADV_HUGEPAGE`d against dTLB misses.
//!
//! Linear probing over aligned pairs (`idx` even, `idx + 1` on the same
//! line); inserts fill the first empty slot of the walk; growth doubles at
//! 3/4 load. Key 0 marks empty slots (real keys carry a nonzero length in
//! the top byte; empty pretokens route to the long map, never here).

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ptr::NonNull;

/// One slot: the packed pretoken key plus its packed encoding (`val`,
/// `ext`; see `tiktoken::pack_val_inline`). Exactly 32 bytes.
#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u128,
    val: u64,
    ext: u64,
}

const _: () = assert!(std::mem::size_of::<Entry>() == 32);

const EMPTY_KEY: u128 = 0;

/// The table's slot array: a manually managed, zeroed (== all-empty),
/// 2 MiB-aligned allocation marked `MADV_HUGEPAGE` (a `Box<[Entry]>`
/// cannot over-align).
struct Slots {
    ptr: NonNull<Entry>,
    cap: usize,
}

impl Slots {
    const HUGE_PAGE: usize = 2 * 1024 * 1024;

    fn new_zeroed(cap: usize) -> Self {
        let layout = Self::layout(cap);
        // SAFETY: layout has nonzero size (cap >= 1).
        let raw = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(raw as *mut Entry) else {
            handle_alloc_error(layout)
        };
        // madvise BEFORE the zeroing write, or the table faults in as
        // 4 KiB pages (measured: +15% cold encode from this order alone).
        super::madvise_hugepage(raw, layout.size());
        // SAFETY: raw is a live allocation of exactly layout.size() bytes.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        Self { ptr, cap }
    }

    fn layout(cap: usize) -> Layout {
        let size = cap * std::mem::size_of::<Entry>();
        // Huge-page alignment only once the table outgrows one huge page;
        // floor of 64 so an even-indexed pair always shares one cache line.
        let align = Self::HUGE_PAGE.min(size.next_power_of_two()).max(64);
        Layout::from_size_align(size, align).expect("table layout overflow")
    }

    #[inline(always)]
    unsafe fn get(&self, idx: usize) -> &Entry {
        debug_assert!(idx < self.cap);
        // SAFETY: caller guarantees idx < cap.
        unsafe { &*self.ptr.as_ptr().add(idx) }
    }

    #[inline(always)]
    unsafe fn get_mut(&mut self, idx: usize) -> &mut Entry {
        debug_assert!(idx < self.cap);
        // SAFETY: caller guarantees idx < cap.
        unsafe { &mut *self.ptr.as_ptr().add(idx) }
    }
}

impl Drop for Slots {
    fn drop(&mut self) {
        // SAFETY: allocated in `new_zeroed` with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.cap)) };
    }
}

// SAFETY: Slots owns its allocation exclusively, like Box<[Entry]>.
unsafe impl Send for Slots {}
unsafe impl Sync for Slots {}

/// Request the line holding `p` into L1 (`L1 = true`) or L2 only. No-op
/// on arches without a prefetch hint.
#[inline(always)]
fn prefetch_line<const L1: bool>(p: *const Entry) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch has no memory effects; any address is allowed.
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};
        if L1 {
            _mm_prefetch(p as *const i8, _MM_HINT_T0);
        } else {
            _mm_prefetch(p as *const i8, _MM_HINT_T1);
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prefetch has no memory effects; any address is allowed.
    unsafe {
        if L1 {
            core::arch::asm!(
                "prfm pldl1keep, [{p}]",
                p = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        } else {
            core::arch::asm!(
                "prfm pldl2keep, [{p}]",
                p = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

pub(crate) struct ShortPretokenCache {
    slots: Slots,
    /// `cap - 1` (capacity is a power of two).
    mask: usize,
    len: usize,
}

impl ShortPretokenCache {
    fn with_pow2_capacity(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two() && cap >= 2);
        Self { slots: Slots::new_zeroed(cap), mask: cap - 1, len: 0 }
    }

    /// A table holding at least `n` entries without growing (3/4 load,
    /// 2^16-slot floor), starting from at least `min_slots` slots (a
    /// worker's workload estimate; the table still grows past it).
    pub(crate) fn with_at_least(n: usize, min_slots: usize) -> Self {
        Self::with_pow2_capacity(Self::required_capacity(n, min_slots))
    }

    /// The slot count [`Self::with_at_least`] would allocate for `n`
    /// entries starting from `min_slots`, without allocating anything.
    pub(crate) fn required_capacity(n: usize, min_slots: usize) -> usize {
        let mut cap = min_slots.max(1 << 16).next_power_of_two();
        while (n + 1) * 4 > cap * 3 {
            cap *= 2;
        }
        cap
    }

    /// Zero every slot in place, keeping the allocation. Invalidates any
    /// outstanding [`ProbeView`].
    pub(crate) fn clear(&mut self) {
        // SAFETY: the allocation holds exactly `cap` entries.
        unsafe { std::ptr::write_bytes(self.slots.ptr.as_ptr(), 0, self.slots.cap) };
        self.len = 0;
    }

    /// Reset to an empty table of exactly `cap` slots (power of two,
    /// >= 2), reusing the current allocation when the capacity matches.
    pub(crate) fn reset_to_capacity(&mut self, cap: usize) {
        debug_assert!(cap.is_power_of_two() && cap >= 2);
        if cap == self.slots.cap {
            self.clear();
            return;
        }
        // Drop the old table (via a 2-slot placeholder) before building
        // the new one, so two full-size tables never coexist.
        self.slots = Slots::new_zeroed(2);
        self.slots = Slots::new_zeroed(cap);
        self.mask = cap - 1;
        self.len = 0;
    }

    /// Address of `h`'s home pair; both slots share the addressed line.
    #[inline(always)]
    fn pair_ptr(&self, h: u64) -> *const Entry {
        // SAFETY: the masked even index is <= mask - 1 < cap.
        unsafe { self.slots.ptr.as_ptr().add((h as usize) & self.mask & !1) }
    }

    /// Request the probe's cache line into L2 only: issued a full chunk
    /// before the probe (covers DRAM) without evicting the walker's L1 set.
    #[inline(always)]
    pub(crate) fn prefetch_l2(&self, h: u64) {
        prefetch_line::<false>(self.pair_ptr(h));
    }

    /// A raw, `Copy` snapshot of the probe parameters, so the emit loop
    /// keeps table base and mask in registers. Invalidated by any insert
    /// (which may grow the table): take a fresh view after one.
    pub(crate) fn probe_view(&self) -> ProbeView {
        ProbeView { base: self.slots.ptr.as_ptr(), pair_mask: self.mask & !1 }
    }

    /// Look up `key`, walking pairs from its home bucket. A miss (`Err`)
    /// reports the first empty slot of the walk — where [`Self::insert_at`]
    /// places the key, valid until the next insert or grow.
    pub(crate) fn get_or_slot(&self, key: u128, h: u64) -> Result<(u64, u64), usize> {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut idx = (h as usize) & self.mask & !1;
        loop {
            // SAFETY: idx is masked and even, so idx + 1 <= mask.
            let e0 = unsafe { self.slots.get(idx) };
            let e1 = unsafe { self.slots.get(idx + 1) };
            if e0.key == key {
                return Ok((e0.val, e0.ext));
            }
            if e1.key == key {
                return Ok((e1.val, e1.ext));
            }
            if e0.key == EMPTY_KEY {
                return Err(idx);
            }
            if e1.key == EMPTY_KEY {
                return Err(idx + 1);
            }
            idx = (idx + 2) & self.mask;
        }
    }

    /// First empty slot of `h`'s pair walk (load < 1 guarantees one).
    fn first_empty(&self, h: u64) -> usize {
        let mut idx = (h as usize) & self.mask & !1;
        loop {
            // SAFETY: idx is masked and even, so idx + 1 <= mask.
            unsafe {
                if self.slots.get(idx).key == EMPTY_KEY {
                    return idx;
                }
                if self.slots.get(idx + 1).key == EMPTY_KEY {
                    return idx + 1;
                }
            }
            idx = (idx + 2) & self.mask;
        }
    }

    /// Insert a key known to be absent (the encode loop only inserts after
    /// a [`Self::get_or_slot`] miss).
    pub(crate) fn insert(&mut self, key: u128, h: u64, val: u64, ext: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        if (self.len + 1) * 4 > self.slots.cap * 3 {
            self.grow();
        }
        let idx = self.first_empty(h);
        // SAFETY: first_empty returns an in-bounds index.
        unsafe { *self.slots.get_mut(idx) = Entry { key, val, ext } };
        self.len += 1;
    }

    /// [`Self::insert`] with the destination already known from a
    /// [`Self::get_or_slot`] miss on the same `key`/`h` (with no insert or
    /// grow in between), skipping the `first_empty` chain walk. A growth
    /// pass invalidates `slot`, so that branch recomputes it.
    pub(crate) fn insert_at(&mut self, slot: usize, key: u128, h: u64, val: u64, ext: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut slot = slot;
        if (self.len + 1) * 4 > self.slots.cap * 3 {
            self.grow();
            slot = self.first_empty(h);
        }
        debug_assert_eq!(slot, self.first_empty(h));
        // SAFETY: get_or_slot and first_empty return in-bounds indices.
        unsafe { *self.slots.get_mut(slot) = Entry { key, val, ext } };
        self.len += 1;
    }

    /// Insert `key`, overwriting its value if already present (the plain
    /// [`Self::insert`] assumes absence). Cold; used by the vocab seed.
    pub(crate) fn replace(&mut self, key: u128, h: u64, val: u64, ext: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut idx = (h as usize) & self.mask & !1;
        loop {
            // SAFETY: idx is masked and even, so idx + 1 <= mask.
            let (k0, k1) = unsafe { (self.slots.get(idx).key, self.slots.get(idx + 1).key) };
            if k0 == key {
                // SAFETY: idx is in bounds (masked above).
                unsafe { *self.slots.get_mut(idx) = Entry { key, val, ext } };
                return;
            }
            if k1 == key {
                // SAFETY: idx + 1 <= mask (masked, even idx).
                unsafe { *self.slots.get_mut(idx + 1) = Entry { key, val, ext } };
                return;
            }
            if k0 == EMPTY_KEY || k1 == EMPTY_KEY {
                // Absent: a fresh insert (with its own growth check).
                self.insert(key, h, val, ext);
                return;
            }
            idx = (idx + 2) & self.mask;
        }
    }

    #[cold]
    fn grow(&mut self) {
        let new_cap = self.slots.cap * 2;
        let old = std::mem::replace(&mut self.slots, Slots::new_zeroed(new_cap));
        self.mask = new_cap - 1;
        for i in 0..old.cap {
            // SAFETY: i < old.cap.
            let e = *unsafe { old.get(i) };
            if e.key == EMPTY_KEY {
                continue;
            }
            // Must be the same hash the inserts' `h` came from.
            let idx = self.first_empty(crate::pretokenize::pretoken_key_hash(e.key));
            // SAFETY: first_empty returns an in-bounds index.
            unsafe { *self.slots.get_mut(idx) = e };
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.cap
    }
}

/// See [`ShortPretokenCache::probe_view`]. Borrows the cache's live
/// allocation; a view taken before an insert may dangle after it.
#[derive(Clone, Copy)]
pub(crate) struct ProbeView {
    base: *const Entry,
    /// `slot mask & !1`, pre-folded off the probe-address critical path.
    pair_mask: usize,
}

impl ProbeView {
    /// Address of `h`'s home pair; both slots share the addressed line.
    #[inline(always)]
    fn pair_ptr(&self, h: u64) -> *const Entry {
        // SAFETY: the masked even index is <= pair_mask <= cap - 2.
        unsafe { self.base.add((h as usize) & self.pair_mask) }
    }

    /// Request the probe's cache line L2 -> L1, a few probes ahead of
    /// [`Self::probe_pair`].
    #[inline(always)]
    pub(crate) fn prefetch(&self, h: u64) {
        prefetch_line::<true>(self.pair_ptr(h));
    }

    /// Branchless probe of `key`'s home pair, touching exactly one cache
    /// line. On `!found` the value lanes are another entry's (displaced
    /// keys and genuine misses both come back `!found`; the slow path
    /// disambiguates). `key == 0` compares equal to empty slots, so the
    /// caller's predicate carries its own `key != 0` term.
    ///
    /// The selects run over unconditionally loaded `val`/`ext` of BOTH
    /// slots: every pure-Rust spelling canonicalizes to an address select
    /// feeding a dependent load, an extra L1 latency on the critical path,
    /// so the asm pins register-value csel/cmov.
    #[inline(always)]
    pub(crate) fn probe_pair(&self, key: u128, h: u64) -> (u64, u64, bool) {
        let p = self.pair_ptr(h);
        // SAFETY: pair_ptr's index is masked and even, so idx + 1 <= mask;
        // the base is live for the view's chunk (see type docs).
        let (e0, e1) = unsafe { (&*p, &*p.add(1)) };
        let m0 = e0.key == key;
        let m1 = e1.key == key;
        #[cfg(target_arch = "aarch64")]
        let (val, ext) = {
            let (mut val, mut ext) = (e0.val, e0.ext);
            // SAFETY: register-only conditional selects; no memory access,
            // no stack use (NZCV is clobbered, which the default options
            // already declare).
            unsafe {
                core::arch::asm!(
                    "cmp {m}, #0",
                    "csel {val}, {val}, {v1}, ne",
                    "csel {ext}, {ext}, {x1}, ne",
                    m = in(reg) m0 as u64,
                    val = inout(reg) val,
                    ext = inout(reg) ext,
                    v1 = in(reg) e1.val,
                    x1 = in(reg) e1.ext,
                    options(pure, nomem, nostack),
                );
            }
            (val, ext)
        };
        #[cfg(target_arch = "x86_64")]
        let (val, ext) = {
            let (mut val, mut ext) = (e1.val, e1.ext);
            // SAFETY: register-only test + conditional moves; no memory
            // access, no stack use.
            unsafe {
                core::arch::asm!(
                    "test {m}, {m}",
                    "cmovne {val}, {v0}",
                    "cmovne {ext}, {x0}",
                    m = in(reg) m0 as u64,
                    val = inout(reg) val,
                    ext = inout(reg) ext,
                    v0 = in(reg) e0.val,
                    x0 = in(reg) e0.ext,
                    options(pure, nomem, nostack),
                );
            }
            (val, ext)
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let (val, ext) = {
            let sel = (m0 as u64).wrapping_neg();
            (
                (e0.val & sel) | (e1.val & !sel),
                (e0.ext & sel) | (e1.ext & !sel),
            )
        };
        (val, ext, m0 | m1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::pretoken_key_hash;

    /// The miss path inserts at the slot its failed lookup reported; every
    /// entry must stay retrievable across several growth passes.
    #[test]
    fn get_or_slot_insert_at_roundtrip() {
        let mut cache = ShortPretokenCache::with_pow2_capacity(64);
        // Multiplier chosen to scatter keys; count forces multiple grows
        // from the 64-slot start (grow threshold is 3/4 load).
        let keys: Vec<u128> = (1u128..=500).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
        for (i, &key) in keys.iter().enumerate() {
            let h = pretoken_key_hash(key);
            match cache.get_or_slot(key, h) {
                Ok(_) => panic!("key {i} present before insert"),
                Err(slot) => cache.insert_at(slot, key, h, i as u64, !(i as u64)),
            }
        }
        assert_eq!(cache.len(), keys.len());
        for (i, &key) in keys.iter().enumerate() {
            let h = pretoken_key_hash(key);
            assert_eq!(cache.get_or_slot(key, h), Ok((i as u64, !(i as u64))), "key {i}");
        }
    }
}
