//! Parallel chunked batch encoding behind encode_batch and encode_files.
//! Documents are grouped into coarse chunks (an oversized document is split
//! at pretoken-safe boundaries for BPE, scanner-safe unit boundaries for
//! SentencePiece), encoded by pooled workers whose pretoken caches persist
//! across calls, and reassembled into one flat id buffer plus per-document
//! row lengths.

use crate::Tokenizer;
use crate::bpe;
use crate::bpe::madvise_hugepage;
use crate::input::DocumentIter;
use crate::input::file_source::{DocFormat, chunk_ranges};
use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, TryLockError};

/// Chunks below this size are not worth a worker handoff; an input that
/// does not fill more than one chunk is encoded serially.
const MIN_CHUNK_BYTES: usize = 1 << 20;

/// Results at least this large free their chunk buffers on a background task.
const DEFERRED_DROP_MIN_BYTES: usize = 32 << 20;

/// Target bytes per parallel chunk: ~16 chunks per thread, floored at
/// MIN_CHUNK_BYTES.
fn chunk_target_bytes(total_bytes: usize) -> usize {
    (total_bytes / (16 * rayon::current_num_threads())).max(MIN_CHUNK_BYTES)
}

/// Token output buffer reserved from a bytes-per-token estimate on the low
/// side of natural language (~4.4 on OWT/GPT-2), with huge pages requested
/// before the encode's stores fault it in.
fn ids_buf(byte_len: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::with_capacity(byte_len / 4 + 16);
    madvise_hugepage(ids.as_mut_ptr() as *mut u8, ids.capacity() * 4);
    ids
}

/// Append one document's token ids to `ids` and its row length to `lens`.
pub(crate) fn encode_into(tokenizer: &mut Tokenizer, doc: &[u8], ids: &mut Vec<u32>, lens: &mut Vec<i64>) {
    let before = ids.len();
    tokenizer.encode_with_added_tokens_flat(doc, ids);
    lens.push((ids.len() - before) as i64);
}

/// SentencePiece analog of `encode_into`. `first` is false only for a
/// non-first fragment of a split document, whose leading section continues
/// one begun earlier and is never ▁-prefixed.
fn sp_encode_into(
    encoder: &mut bpe::sentencepiece::Encoder<'_>,
    text: &str,
    first: bool,
    ids: &mut Vec<u32>,
    lens: &mut Vec<i64>,
) {
    let before = ids.len();
    encoder.encode_raw_fragment_cb(text, first, &mut |tokens| {
        ids.extend(tokens.iter().map(|&t| u32::from(t)))
    });
    lens.push((ids.len() - before) as i64);
}

/// Iterate the documents in a byte region per `format`: JSONL lines,
/// separator-delimited text, or the whole region as one document.
pub(crate) fn for_each_doc(bytes: &[u8], format: &DocFormat, mut f: impl FnMut(&[u8])) {
    use crate::input::jsonl::JsonLinesSlice;
    match format {
        DocFormat::Jsonl { field } => {
            for doc in JsonLinesSlice::new(bytes, field) {
                f(doc.as_ref());
            }
        }
        DocFormat::Text { separator: Some(sep) } if !sep.is_empty() => {
            for doc in DocumentIter::new(bytes, sep) {
                f(doc);
            }
        }
        DocFormat::Text { .. } => f(bytes),
        // Parquet rows are materialized into whole documents before encoding.
        DocFormat::Parquet { .. } => {
            unreachable!("parquet files are materialized into documents before encoding")
        }
    }
}

/// Work unit for parallel encoding.
enum EncodeChunk<'a> {
    /// A run of whole documents, one output row each.
    Docs(Vec<&'a [u8]>),
    /// A byte region holding many documents, split during encoding.
    Region { bytes: &'a [u8], format: &'a DocFormat },
    /// A pretoken-safe fragment of one oversized document; fragments of a
    /// document are consecutive chunks and `first` marks the first.
    Fragment { bytes: &'a [u8], first: bool },
}

/// Token output of one chunk. `continues` means the first length extends
/// the previous chunk's last row (a non-first fragment).
struct ChunkTokens {
    ids: Vec<u32>,
    lens: Vec<i64>,
    continues: bool,
}

fn encode_chunk(tokenizer: &mut Tokenizer, chunk: &EncodeChunk) -> ChunkTokens {
    let byte_len = match chunk {
        EncodeChunk::Docs(docs) => docs.iter().map(|d| d.len()).sum::<usize>(),
        EncodeChunk::Region { bytes, .. } | EncodeChunk::Fragment { bytes, .. } => bytes.len(),
    };
    let mut ids = ids_buf(byte_len);
    let mut lens = Vec::new();
    let mut continues = false;
    match chunk {
        EncodeChunk::Docs(docs) => {
            for doc in docs {
                encode_into(tokenizer, doc, &mut ids, &mut lens);
            }
        }
        EncodeChunk::Region { bytes, format } => {
            for_each_doc(bytes, format, |doc| encode_into(tokenizer, doc, &mut ids, &mut lens))
        }
        EncodeChunk::Fragment { bytes, first } => {
            encode_into(tokenizer, bytes, &mut ids, &mut lens);
            continues = !*first;
        }
    }
    ChunkTokens { ids, lens, continues }
}

/// LPT chunk sizing is on unless `GIGATOK_NO_LPT` is set (kept for A/B
/// measurement; output is identical either way). Read once per encode call.
fn lpt_from_env() -> bool {
    std::env::var_os("GIGATOK_NO_LPT").is_none()
}

/// Group documents into parallel chunks. With LPT: ~2x-target chunks over
/// the first ~80% of bytes, quarter-target chunks over the last ~20%, so the
/// core that draws the last chunk strands the others behind a short tail
/// (rayon hands chunks out in index order). Without LPT every chunk aims
/// for `target`. A document larger than `2 * target` is split into
/// consecutive Fragment chunks at pretoken-safe boundaries that no
/// added-token occurrence straddles.
fn build_doc_chunks<'a>(
    docs: &[&'a [u8]],
    total: usize,
    target: usize,
    added_tokens: &[(&[u8], bool)],
    lpt: bool,
) -> Vec<EncodeChunk<'a>> {
    let (head_bytes, big, tail_target) = if lpt {
        (total - total / 5, 2 * target, (target / 4).max(MIN_CHUNK_BYTES))
    } else {
        (0, target, target)
    };
    let mut chunks = Vec::new();
    let mut group: Vec<&[u8]> = Vec::new();
    let mut emitted = 0usize;
    let mut acc = 0usize;
    for &doc in docs {
        if doc.len() > 2 * target {
            if !group.is_empty() {
                chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
                emitted += acc;
                acc = 0;
            }
            let head_len = if lpt { head_bytes.saturating_sub(emitted) } else { usize::MAX };
            push_fragment_chunks(&mut chunks, doc, head_len, big, tail_target, added_tokens);
            emitted += doc.len();
            continue;
        }
        group.push(doc);
        acc += doc.len();
        let group_target = if emitted < head_bytes { big } else { tail_target };
        if acc >= group_target {
            chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
            emitted += acc;
            acc = 0;
        }
    }
    if !group.is_empty() {
        chunks.push(EncodeChunk::Docs(group));
    }
    chunks
}

/// Split one oversized document into Fragment chunks: `big`-sized over the
/// first `head_len` bytes, `tail_target`-sized after. Sub-splitting a tail
/// fragment stays boundary-safe: the pretoken cut check is local, and an
/// added token is far shorter than any sub-cut's distance from the
/// fragment's already-safe edges.
fn push_fragment_chunks<'a>(
    chunks: &mut Vec<EncodeChunk<'a>>,
    doc: &'a [u8],
    head_len: usize,
    big: usize,
    tail_target: usize,
    added_tokens: &[(&[u8], bool)],
) {
    let mut first = true;
    let mut push = |bytes: &'a [u8]| {
        chunks.push(EncodeChunk::Fragment { bytes, first: std::mem::take(&mut first) })
    };
    for r in crate::pretokenize::safe_split_ranges(doc, big, added_tokens) {
        if r.start < head_len || r.len() <= tail_target {
            push(&doc[r]);
        } else {
            for sub in crate::pretokenize::safe_split_ranges(&doc[r.clone()], tail_target, added_tokens) {
                push(&doc[r.start + sub.start..r.start + sub.end]);
            }
        }
    }
}

/// Map items serially when there is at most one, in parallel otherwise.
fn map_maybe_par<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use rayon::prelude::*;
    if items.len() <= 1 {
        items.iter().map(&f).collect()
    } else {
        items.par_iter().map(&f).collect()
    }
}

/// Concatenate per-chunk row lengths into per-document row counts, merging
/// `continues` fragments into the previous document's row.
fn row_counts(chunks: &[ChunkTokens]) -> Vec<i64> {
    let mut counts: Vec<i64> = Vec::new();
    for chunk in chunks {
        let mut lens = chunk.lens.iter().copied();
        if chunk.continues
            && let Some(l) = lens.next()
        {
            *counts.last_mut().expect("continuation fragment before any document") += l;
        }
        counts.extend(lens);
    }
    counts
}

/// Free spent chunk buffers off the caller's critical path: their munmap
/// teardown (address-space write lock) would otherwise convoy the gather
/// copy's page faults. Small results just drop inline.
fn defer_drop(chunks: Vec<ChunkTokens>) {
    let total: usize = chunks.iter().map(|c| c.ids.len()).sum();
    if total * std::mem::size_of::<u32>() >= DEFERRED_DROP_MIN_BYTES {
        rayon::spawn(move || drop(chunks));
    }
}

/// Overlapped gather: a flat id buffer reserved at an upper bound before
/// chunk sizes are known, plus a cursor over the longest fully-encoded
/// prefix of the chunk sequence. Chunk completion is near-sequential
/// (in-order handout, descending sizes), so a worker that finishes a chunk
/// commits the ready prefix while the tail still encodes, hiding the copy
/// and its page faults inside the encode phase.
///
/// The bound: a token consumes at least one input byte, so `total_bytes`
/// tokens bounds the output. If the reservation fails or a chunk overflows
/// the bound (only possible under NFC expansion), the caller falls back to
/// the collect-then-gather path.
struct Committer {
    /// Owns the reservation. `UnsafeCell` so each `advance` derives the
    /// destination pointer fresh under the cursor lock instead of capturing
    /// one across the struct's construction-time moves.
    flat: UnsafeCell<Vec<u32>>,
    /// Reserved capacity in tokens; commits never write at or past it.
    cap: usize,
    cursor: Mutex<CommitCursor>,
}

struct CommitCursor {
    /// Index of the first uncommitted chunk.
    next: usize,
    /// Tokens committed so far == sum of ids.len() over chunks[..next].
    offset: usize,
    /// A chunk did not fit under `cap`; committing has stopped for good.
    overflowed: bool,
}

// SAFETY: the heap buffer behind `flat` is never reallocated while shared
// (the Vec is resized only in `finish`, after all shared use has ended), and
// every shared-phase write lands in a disjoint, in-bounds range under the
// cursor lock.
unsafe impl Send for Committer {}
unsafe impl Sync for Committer {}

impl Committer {
    /// Chunks committed per `advance` call, bounding how long one worker is
    /// away from encoding.
    const MAX_DRAIN: usize = 8;

    /// Reserve `cap` tokens up front, or None if the allocator refuses.
    fn try_new(cap: usize) -> Option<Self> {
        let mut flat: Vec<u32> = Vec::new();
        if cap == 0 || flat.try_reserve_exact(cap).is_err() {
            return None;
        }
        madvise_hugepage(flat.as_mut_ptr() as *mut u8, cap * std::mem::size_of::<u32>());
        Some(Self {
            flat: UnsafeCell::new(flat),
            cap,
            cursor: Mutex::new(CommitCursor { next: 0, offset: 0, overflowed: false }),
        })
    }

    /// Copy any freshly completed prefix chunks into the flat buffer.
    /// Non-blocking: if another worker is mid-commit, return to encoding;
    /// the holder, a later completion, or `finish` picks the chunk up.
    fn advance(&self, outs: &[OnceLock<ChunkTokens>]) {
        let Ok(mut cur) = self.cursor.try_lock() else {
            return;
        };
        // SAFETY: the cursor lock is held and the Vec is not mutated during
        // the shared phase, so this only reads the buffer pointer.
        let base = unsafe { (*self.flat.get()).as_mut_ptr() };
        for _ in 0..Self::MAX_DRAIN {
            if cur.overflowed {
                return;
            }
            let Some(chunk) = outs.get(cur.next).and_then(OnceLock::get) else {
                return;
            };
            let len = chunk.ids.len();
            if self.cap - cur.offset < len {
                cur.overflowed = true;
                return;
            }
            // SAFETY: holding `cursor`; [offset, offset+len) is within the
            // reservation and disjoint from every earlier commit.
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.ids.as_ptr(), base.add(cur.offset), len);
            }
            cur.offset += len;
            cur.next += 1;
        }
    }

    /// After all chunks are encoded: copy the uncommitted suffix in
    /// parallel, size the buffer to `total` and trim the reservation. None
    /// means the bound was overrun; the caller falls back to the classic
    /// gather from the (intact) chunk buffers.
    fn finish(self, chunks: &[ChunkTokens], total: usize) -> Option<Vec<u32>> {
        use rayon::prelude::*;
        struct SyncPtr(*mut u32);
        // SAFETY: only used for the disjoint in-bounds writes below.
        unsafe impl Send for SyncPtr {}
        unsafe impl Sync for SyncPtr {}
        impl SyncPtr {
            /// SAFETY: `off` must be within the reservation.
            unsafe fn at(&self, off: usize) -> *mut u32 {
                unsafe { self.0.add(off) }
            }
        }

        let Committer { flat, cap, cursor } = self;
        let mut flat = flat.into_inner();
        let cur = cursor.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
        if cur.overflowed || total > cap {
            return None;
        }
        let rest = &chunks[cur.next..];
        let mut offsets = Vec::with_capacity(rest.len());
        let mut offset = cur.offset;
        for chunk in rest {
            offsets.push(offset);
            offset += chunk.ids.len();
        }
        debug_assert_eq!(offset, total);
        // Derived after the Vec moved out of the cell (a move retags its
        // pointer under strict aliasing models).
        let base = SyncPtr(flat.as_mut_ptr());
        // `with_max_len(1)` keeps the multi-MB copies stealable one by one.
        rest.par_iter().zip(offsets).with_max_len(1).for_each(|(chunk, off)| {
            // SAFETY: workers are joined; suffix ranges are disjoint from each
            // other and from the committed prefix, and end at total <= cap.
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.ids.as_ptr(), base.at(off), chunk.ids.len());
            }
        });
        // SAFETY: capacity >= total and [0, total) is fully initialized.
        unsafe {
            flat.set_len(total);
        }
        // Mainstream allocators trim large allocations in place; the
        // untouched tail pages were never faulted.
        flat.shrink_to_fit();
        Some(flat)
    }
}

/// Encode all chunks with pooled workers and gather them into one flat id
/// buffer plus per-document row counts. Chunks are handed out in strict
/// index order through an atomic counter (one pulling task per rayon
/// thread) rather than `par_iter`, which could leave a thread starting a
/// big early chunk after everyone else reached the small tail; in-order
/// handout also makes completion near-sequential, which is what lets the
/// gather overlap the encode (see `Committer`).
fn encode_chunks_gathered(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
    total_bytes: usize,
) -> (Vec<u32>, Vec<i64>) {
    encode_chunks_gathered_with_cap(workers, proto, chunks, total_bytes, total_bytes)
}

/// `encode_chunks_gathered` with the committer's reservation bound passed
/// explicitly, so tests can force the overflow fallback.
fn encode_chunks_gathered_with_cap(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
    total_bytes: usize,
    cap_tokens: usize,
) -> (Vec<u32>, Vec<i64>) {
    let share = total_bytes / rayon::current_num_threads().max(1);
    let encode = |c: &EncodeChunk| workers.with_worker(proto, share, |tok| encode_chunk(tok, c));
    if chunks.len() <= 1 {
        // A lone chunk's id buffer is the flat result, no gather copy at all.
        return match chunks.first() {
            Some(chunk) => {
                let out = encode(chunk);
                let counts = row_counts(std::slice::from_ref(&out));
                (out.ids, counts)
            }
            None => (Vec::new(), Vec::new()),
        };
    }
    let next = AtomicUsize::new(0);
    let outs: Vec<OnceLock<ChunkTokens>> = (0..chunks.len()).map(|_| OnceLock::new()).collect();
    let committer = Committer::try_new(cap_tokens);
    let tasks = rayon::current_num_threads().min(chunks.len());
    rayon::scope(|s| {
        for _ in 0..tasks {
            s.spawn(|_| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(i) else {
                        // One last drain: this worker's final chunk may have
                        // been skipped while another held the commit lock.
                        if let Some(c) = &committer {
                            c.advance(&outs);
                        }
                        break;
                    };
                    let _ = outs[i].set(encode(chunk));
                    if let Some(c) = &committer {
                        c.advance(&outs);
                    }
                }
            });
        }
    });
    let outs: Vec<ChunkTokens> = outs
        .into_iter()
        .map(|slot| slot.into_inner().expect("every claimed chunk was encoded"))
        .collect();
    let counts = row_counts(&outs);
    let total: usize = outs.iter().map(|c| c.ids.len()).sum();
    match committer.and_then(|c| c.finish(&outs, total)) {
        Some(flat) => {
            defer_drop(outs);
            (flat, counts)
        }
        None => (gather_flat(outs), counts),
    }
}

/// Merge per-chunk outputs into one flat id buffer and per-document row
/// counts with a parallel copy (the SentencePiece paths gather this way;
/// the BPE path falls back to it when the overlapped gather is refused).
fn assemble_ragged(chunks: Vec<ChunkTokens>) -> (Vec<u32>, Vec<i64>) {
    let counts = row_counts(&chunks);
    (gather_flat(chunks), counts)
}

fn gather_flat(chunks: Vec<ChunkTokens>) -> Vec<u32> {
    use rayon::prelude::*;
    let total: usize = chunks.iter().map(|c| c.ids.len()).sum();
    let mut flat = vec![0u32; total];
    madvise_hugepage(flat.as_mut_ptr() as *mut u8, total * std::mem::size_of::<u32>());
    let mut rest: &mut [u32] = &mut flat;
    let mut slices = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let (head, tail) = rest.split_at_mut(chunk.ids.len());
        slices.push(head);
        rest = tail;
    }
    slices
        .into_par_iter()
        .zip(chunks.par_iter())
        .with_max_len(1)
        .for_each(|(dst, chunk)| dst.copy_from_slice(&chunk.ids));
    defer_drop(chunks);
    flat
}

/// Pool of forked tokenizer workers, one slot per rayon thread plus a
/// dedicated serial worker, forked lazily and retained for the tokenizer's
/// lifetime so pretoken caches stay warm across calls.
///
/// Invariant: the prototype must not be mutated between encodes that share
/// a pool. Workers are never refreshed, so a later mutation would leave
/// already-forked slots on the old state. The Python bindings expose no
/// mutator after construction.
pub struct WorkerPool {
    slots: OnceLock<Vec<Mutex<Option<Tokenizer>>>>,
    /// Worker for the `parallel=false` paths, kept apart from `slots` so a
    /// sequential call never touches (or even sizes) the rayon pool.
    serial: Mutex<Option<Tokenizer>>,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerPool {
    pub fn new() -> Self {
        Self { slots: OnceLock::new(), serial: Mutex::new(None) }
    }

    /// Run `f` with exclusive access to a pooled worker, forking one sized
    /// for `expected_bytes` if the slot is empty. Rayon never runs more
    /// tasks than threads and there is one slot per thread, so a free slot
    /// always exists; the yield loop only spins when non-rayon threads
    /// encode concurrently.
    fn with_worker<R>(
        &self,
        proto: &Tokenizer,
        expected_bytes: usize,
        f: impl FnOnce(&mut Tokenizer) -> R,
    ) -> R {
        let slots = self
            .slots
            .get_or_init(|| (0..rayon::current_num_threads()).map(|_| Mutex::new(None)).collect());
        loop {
            for slot in slots {
                match slot.try_lock() {
                    Ok(mut guard) => {
                        return f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)));
                    }
                    Err(TryLockError::Poisoned(poisoned)) => {
                        // A worker panicked mid-encode; rebuild it.
                        let mut guard = poisoned.into_inner();
                        *guard = None;
                        return f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)));
                    }
                    Err(TryLockError::WouldBlock) => {}
                }
            }
            std::thread::yield_now();
        }
    }

    /// `with_worker` for the sequential paths: the dedicated serial worker,
    /// never touching the rayon-sized slots.
    fn with_serial_worker<R>(
        &self,
        proto: &Tokenizer,
        expected_bytes: usize,
        f: impl FnOnce(&mut Tokenizer) -> R,
    ) -> R {
        let mut guard = self.serial.lock().unwrap_or_else(|poisoned| {
            let mut guard = poisoned.into_inner();
            *guard = None;
            guard
        });
        f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)))
    }
}

/// Shared core of encode_batch / encode_files for pre-resolved document
/// slices. Public so Rust benches exercise the same parallel path as the
/// Python bindings. `GIGATOK_NO_LPT` in the environment disables LPT chunk
/// sizing (see `lpt_from_env`).
pub fn encode_docs_ragged(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
) -> (Vec<u32>, Vec<i64>) {
    encode_docs_ragged_with(workers, proto, docs, lpt_from_env())
}

/// `encode_docs_ragged` with the LPT switch passed explicitly.
pub(crate) fn encode_docs_ragged_with(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
    lpt: bool,
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = docs.iter().map(|d| d.len()).sum();
    let added = proto.added_token_split_blockers();
    let chunks = build_doc_chunks(docs, total, chunk_target_bytes(total), &added, lpt);
    encode_chunks_gathered(workers, proto, &chunks, total)
}

/// Work unit for parallel SentencePiece encoding, mirroring `EncodeChunk`.
enum SpChunk<'a> {
    Docs(Vec<&'a str>),
    /// A scanner-safe fragment of one oversized document (see
    /// `SentencePieceBPE::safe_fragment_ranges`).
    Fragment { text: &'a str, first: bool },
}

/// Group documents into parallel chunks of roughly `target` bytes. A
/// document larger than `2 * target` is split into Fragment chunks at unit
/// boundaries the scanner proves safe, except on models without the raw
/// fast path, where it stays one chunk.
fn sp_build_chunks<'a>(
    tokenizer: &bpe::SentencePieceBPE,
    texts: &[&'a str],
    target: usize,
) -> Vec<SpChunk<'a>> {
    let can_split = tokenizer.supports_fragment_split();
    let mut chunks = Vec::new();
    let mut group: Vec<&str> = Vec::new();
    let mut acc = 0usize;
    for &text in texts {
        if can_split && text.len() > 2 * target {
            if !group.is_empty() {
                chunks.push(SpChunk::Docs(std::mem::take(&mut group)));
                acc = 0;
            }
            let mut first = true;
            for r in tokenizer.safe_fragment_ranges(text, target) {
                chunks.push(SpChunk::Fragment { text: &text[r], first: std::mem::take(&mut first) });
            }
            continue;
        }
        group.push(text);
        acc += text.len();
        if acc >= target {
            chunks.push(SpChunk::Docs(std::mem::take(&mut group)));
            acc = 0;
        }
    }
    if !group.is_empty() {
        chunks.push(SpChunk::Docs(group));
    }
    chunks
}

/// Encode SentencePiece chunks with a per-chunk Encoder and gather them.
fn sp_encode_chunks(tokenizer: &bpe::SentencePieceBPE, chunks: &[SpChunk]) -> (Vec<u32>, Vec<i64>) {
    let outs = map_maybe_par(chunks, |chunk| {
        let byte_len = match chunk {
            SpChunk::Docs(group) => group.iter().map(|t| t.len()).sum::<usize>(),
            SpChunk::Fragment { text, .. } => text.len(),
        };
        let mut ids = ids_buf(byte_len);
        let mut encoder = tokenizer.encoder();
        let mut lens: Vec<i64> = Vec::new();
        let mut continues = false;
        match chunk {
            SpChunk::Docs(group) => {
                for text in group {
                    sp_encode_into(&mut encoder, text, true, &mut ids, &mut lens);
                }
            }
            SpChunk::Fragment { text, first } => {
                sp_encode_into(&mut encoder, text, *first, &mut ids, &mut lens);
                continues = !*first;
            }
        }
        ChunkTokens { ids, lens, continues }
    });
    assemble_ragged(outs)
}

/// SentencePiece analog of `encode_docs_ragged`.
pub fn sp_encode_docs_ragged(tokenizer: &bpe::SentencePieceBPE, texts: &[&str]) -> (Vec<u32>, Vec<i64>) {
    let total: usize = texts.iter().map(|t| t.len()).sum();
    let chunks = sp_build_chunks(tokenizer, texts, chunk_target_bytes(total));
    sp_encode_chunks(tokenizer, &chunks)
}

/// Sequential `sp_encode_docs_ragged`: one Encoder over all documents.
#[cfg(test)]
fn sp_encode_docs_ragged_serial(
    tokenizer: &bpe::SentencePieceBPE,
    texts: &[&str],
) -> (Vec<u32>, Vec<i64>) {
    let mut encoder = tokenizer.encoder();
    let mut ids: Vec<u32> = Vec::new();
    let mut lens: Vec<i64> = Vec::with_capacity(texts.len());
    for &text in texts {
        sp_encode_into(&mut encoder, text, true, &mut ids, &mut lens);
    }
    (ids, lens)
}

/// encode_files core for the BPE backend. With no separator each file is
/// one document; otherwise each file is cut into byte regions at document
/// boundaries and documents are extracted while encoding.
pub(crate) fn encode_files_docs(
    workers: &WorkerPool,
    proto: &Tokenizer,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    if matches!(format, DocFormat::Text { separator: None }) {
        return encode_docs_ragged(workers, proto, files);
    }
    let total: usize = files.iter().map(|f| f.len()).sum();
    let target = chunk_target_bytes(total);
    let chunks: Vec<EncodeChunk> = files
        .iter()
        .flat_map(|&bytes| {
            chunk_ranges(bytes, format, target)
                .into_iter()
                .map(move |r| EncodeChunk::Region { bytes: &bytes[r], format })
        })
        .collect();
    encode_chunks_gathered(workers, proto, &chunks, total)
}

/// Sequential `encode_files_docs`: every document in file order on the
/// calling thread with the pool's serial worker, never touching rayon.
/// Token- and order-identical to the parallel path.
pub(crate) fn encode_files_docs_serial(
    workers: &WorkerPool,
    proto: &Tokenizer,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = files.iter().map(|f| f.len()).sum();
    workers.with_serial_worker(proto, total, |tok| {
        let mut ids = ids_buf(total);
        let mut lens = Vec::new();
        for &bytes in files {
            for_each_doc(bytes, format, |doc| encode_into(tok, doc, &mut ids, &mut lens));
        }
        (ids, lens)
    })
}

/// encode_files core for the SentencePiece backend; documents are trusted
/// to be valid UTF-8.
pub(crate) fn sp_encode_files_docs(
    tokenizer: &bpe::SentencePieceBPE,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    if matches!(format, DocFormat::Text { separator: None }) {
        // SAFETY: file contents are trusted valid UTF-8 (encode_files' contract).
        let texts: Vec<&str> = files.iter().map(|&b| unsafe { std::str::from_utf8_unchecked(b) }).collect();
        return sp_encode_docs_ragged(tokenizer, &texts);
    }
    let total: usize = files.iter().map(|f| f.len()).sum();
    let target = chunk_target_bytes(total);
    let chunks: Vec<(usize, Range<usize>)> = files
        .iter()
        .enumerate()
        .flat_map(|(i, &bytes)| chunk_ranges(bytes, format, target).into_iter().map(move |r| (i, r)))
        .collect();
    let outs = map_maybe_par(&chunks, |(file, range)| {
        let mut encoder = tokenizer.encoder();
        let mut ids: Vec<u32> = Vec::new();
        let mut lens: Vec<i64> = Vec::new();
        for_each_doc(&files[*file][range.clone()], format, |doc| {
            let text = unsafe { std::str::from_utf8_unchecked(doc) };
            sp_encode_into(&mut encoder, text, true, &mut ids, &mut lens);
        });
        ChunkTokens { ids, lens, continues: false }
    });
    assemble_ragged(outs)
}

/// Sequential `sp_encode_files_docs`: one Encoder over every file's
/// documents in order, never touching rayon.
pub(crate) fn sp_encode_files_docs_serial(
    tokenizer: &bpe::SentencePieceBPE,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    let mut encoder = tokenizer.encoder();
    let mut ids: Vec<u32> = Vec::new();
    let mut lens: Vec<i64> = Vec::new();
    for &bytes in files {
        for_each_doc(bytes, format, |doc| {
            let text = unsafe { std::str::from_utf8_unchecked(doc) };
            sp_encode_into(&mut encoder, text, true, &mut ids, &mut lens);
        });
    }
    (ids, lens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Byte-level vocab: one token per byte, so any misordered or dropped
    /// chunk is visible in the flat buffer.
    fn byte_proto() -> Tokenizer {
        let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = (0..=u8::MAX).map(|b| vec![b]).collect();
        Tokenizer::new(merges, vocab, None)
    }

    /// Deterministic pseudo-text with plenty of alnum-space-alpha cut points
    /// for safe_split_ranges.
    fn lcg_text(seed: u64) -> impl FnMut(usize) -> Vec<u8> {
        let mut state = seed;
        move |len| {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    b"abcdefghijklmnopqrstuvwxyz0123456789    "[((state >> 33) % 40) as usize]
                })
                .collect()
        }
    }

    /// The parallel chunked path (LPT sizes, pooled workers, overlapped
    /// gather) must be token- and order-identical to a serial per-document
    /// encode, with LPT both on and off.
    #[test]
    fn parallel_ragged_matches_serial() {
        let proto = byte_proto();
        let mut text = lcg_text(0x9E3779B97F4A7C15);
        // Mid-size docs that group, one oversized doc that fragments across
        // the head/tail boundary, then small docs so continuation rows land
        // mid-output.
        let mut owned: Vec<Vec<u8>> = Vec::new();
        for _ in 0..30 {
            owned.push(text(300 << 10));
        }
        owned.push(text(12 << 20));
        for _ in 0..30 {
            owned.push(text(100 << 10));
        }
        let docs: Vec<&[u8]> = owned.iter().map(|d| d.as_slice()).collect();

        let mut ids_ref: Vec<u32> = Vec::new();
        let mut lens_ref: Vec<i64> = Vec::new();
        let mut serial = proto.fork();
        for doc in &docs {
            encode_into(&mut serial, doc, &mut ids_ref, &mut lens_ref);
        }

        for lpt in [true, false] {
            let workers = WorkerPool::new();
            let (flat, lens) = encode_docs_ragged_with(&workers, &proto, &docs, lpt);
            assert_eq!(lens, lens_ref, "lens mismatch (lpt={lpt})");
            assert_eq!(flat, ids_ref, "ids mismatch (lpt={lpt})");
        }

        // The bindings' parallel=false path must match too.
        let workers = WorkerPool::new();
        let whole = DocFormat::Text { separator: None };
        let (flat, lens) = encode_files_docs_serial(&workers, &proto, &docs, &whole);
        assert_eq!(lens, lens_ref, "lens mismatch (serial)");
        assert_eq!(flat, ids_ref, "ids mismatch (serial)");
    }

    /// Assert token identity, reporting the first divergence and a short
    /// window of both streams instead of millions of ids.
    #[track_caller]
    fn assert_ids_match(tag: &str, ids: &[u32], ids_ref: &[u32]) {
        if ids == ids_ref {
            return;
        }
        let i = ids_ref
            .iter()
            .zip(ids)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| ids_ref.len().min(ids.len()));
        panic!(
            "{tag}: ids mismatch at token {i}: serial[{i}..] = {:?}, fragmented[{i}..] = {:?}",
            &ids_ref[i..(i + 8).min(ids_ref.len())],
            &ids[i..(i + 8).min(ids.len())],
        );
    }

    /// SentencePiece parallel encode with fragmented oversized documents
    /// must match the serial one-encoder path. The cached models cover the
    /// raw fast-path shapes (TinyLlama: unguarded prepend; gemma-2b:
    /// crossing pieces; gemma-3: guarded prepend, many added tokens) on
    /// boundary-hostile text with a small target forcing many cuts.
    #[test]
    fn sp_parallel_fragmented_matches_serial() {
        let models = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "unsloth/gemma-2b",
            "google/gemma-3-4b-it",
        ];
        let block = concat!(
            "The  quick   brown fox \u{2014} jumps; over, the: lazy. dog!  \n\n",
            "\t\tindent()\twide  spacing      here \r\nCRLF\r\n",
            "<s>raw<s> tokens </s> spaced <unk> out <bos> and <eos>\n",
            "<start_of_turn>user hello<end_of_turn> <pad>\n",
            "html-ish <b> </b> crossing > </ pieces >\u{2581}</ literal mark \u{2581}word\n",
            "日本語のテキストと émojis 🎉🚀, combining a\u{301}e\u{308}o\u{302}, nbsp\u{a0}here ½ ﬁ ㎒\n",
            "   leading and trailing   \n",
        );
        let mut big = String::new();
        while big.len() < (6 << 20) {
            big.push_str(block);
        }
        let texts: Vec<&str> = vec![
            "short doc <s> one",
            block,
            &big,
            "",
            "tail doc </s>",
            block,
        ];
        for repo in models {
            let Some(path) = crate::test_hub::hf_tokenizer_json(repo) else {
                eprintln!("Skipping {repo}: tokenizer.json not in the HF cache");
                continue;
            };
            let tok = crate::load_tokenizer::hf::load_hf_sentencepiece(&path).unwrap();
            assert!(
                tok.supports_fragment_split(),
                "{repo} should take the raw fast path (fragment splitting)"
            );
            let (ids_ref, lens_ref) = sp_encode_docs_ragged_serial(&tok, &texts);

            // The public parallel path (default target sizing).
            let (ids, lens) = sp_encode_docs_ragged(&tok, &texts);
            assert_eq!(lens, lens_ref, "{repo}: lens mismatch (default target)");
            assert_ids_match(&format!("{repo} (default target)"), &ids, &ids_ref);

            // A small target forces ~a hundred fragment boundaries inside
            // the hostile content.
            let chunks = sp_build_chunks(&tok, &texts, 64 << 10);
            let fragments = chunks
                .iter()
                .filter(|c| matches!(c, SpChunk::Fragment { .. }))
                .count();
            assert!(
                fragments > 50,
                "{repo}: expected many fragments, got {fragments}"
            );
            let (ids, lens) = sp_encode_chunks(&tok, &chunks);
            assert_eq!(lens, lens_ref, "{repo}: lens mismatch (small target)");
            assert_ids_match(&format!("{repo} (small target)"), &ids, &ids_ref);
        }
    }

    /// Fragment cuts vs added-token edge cases on synthetic models covering
    /// every raw-prepend shape, with lstrip/rstrip tokens, a space-carrying
    /// token and a self-overlapping one; a tiny target tries a cut every
    /// few bytes.
    #[test]
    fn sp_fragment_cuts_respect_added_tokens() {
        use crate::load_tokenizer::hf::load_hf_slice;
        // Char vocab + byte fallback, no merges, so boundary divergence is
        // fully visible in char-level ids.
        let mut vocab_entries: Vec<String> = (0u16..=255)
            .map(|b| format!("\"<0x{b:02X}>\": {b}"))
            .collect();
        let mut next_id = 256u32;
        vocab_entries.push(format!("\"\u{2581}\": {next_id}"));
        next_id += 1;
        for c in "abcdefghijklmnopqrstuvwxyz.,!?".chars() {
            vocab_entries.push(format!("\"{c}\": {next_id}"));
            next_id += 1;
        }
        let added = [
            ("<p>", false, false),
            ("<l>", true, false),
            ("<r>", false, true),
            ("w w", false, false), // can straddle a space cut
            ("aa", false, false),  // self-overlapping occurrences
        ];
        let added_json: Vec<String> = added
            .iter()
            .map(|(content, lstrip, rstrip)| {
                let id = next_id;
                next_id += 1;
                format!(
                    "{{\"id\": {id}, \"content\": \"{content}\", \"lstrip\": {lstrip}, \
                     \"rstrip\": {rstrip}, \"normalized\": false, \"special\": true}}"
                )
            })
            .collect();
        // Every raw-prepend shape a cut interacts with.
        use crate::bpe::sentencepiece::{RawPrepend, WordSplit};
        let pipelines = [
            (
                "{\"type\": \"Sequence\", \"normalizers\": [\
                   {\"type\": \"Prepend\", \"prepend\": \"\u{2581}\"}, \
                   {\"type\": \"Replace\", \"pattern\": {\"String\": \" \"}, \"content\": \"\u{2581}\"}]}",
                "null",
                RawPrepend::Unguarded,
                WordSplit::SpaceRuns,
            ),
            (
                "null",
                "{\"type\": \"Metaspace\", \"replacement\": \"\u{2581}\", \
                  \"prepend_scheme\": \"always\", \"split\": true}",
                RawPrepend::GuardedAlways,
                WordSplit::EveryMark,
            ),
            (
                "null",
                "{\"type\": \"Metaspace\", \"replacement\": \"\u{2581}\", \
                  \"prepend_scheme\": \"first\", \"split\": false}",
                RawPrepend::GuardedFirst,
                WordSplit::SpaceRuns,
            ),
        ];

        let block = concat!(
            "plain words here <p> and <p><p> doubled\n",
            "lstrip near ws   <l> and far<l>tight\n",
            "rstrip eats ws <r>   after and<r>tight\n",
            "aaa aaaa a aa overlapping aa\n",
            "w w w w w straddling spaces\n",
            "punct, split. here! ok? more,words.now\n",
            "   runs\t\tof   whitespace \n\n",
            "<l>   <r>   <l><r> adjacent tokens\n",
        );
        let mut text = String::new();
        while text.len() < 100_000 {
            text.push_str(block);
        }
        let texts = [text.as_str()];

        for (normalizer, pre_tokenizer, prepend, word_split) in pipelines {
            let json = format!(
                "{{\"added_tokens\": [{}], \
                  \"normalizer\": {normalizer}, \
                  \"pre_tokenizer\": {pre_tokenizer}, \
                  \"model\": {{\"type\": \"BPE\", \"byte_fallback\": true, \
                    \"vocab\": {{{}}}, \"merges\": []}}}}",
                added_json.join(", "),
                vocab_entries.join(", "),
            );
            let tok = match load_hf_slice(json.as_bytes()).unwrap() {
                crate::load_tokenizer::hf::HfTokenizer::SentencePiece(tok) => tok,
                _ => panic!("synthetic model should load as SentencePiece"),
            };
            // Pin the shape under test.
            assert_eq!(tok.raw_prepend, Some(prepend), "unexpected raw prepend");
            assert_eq!(tok.word_split, word_split, "unexpected word split");

            let (ids_ref, lens_ref) = sp_encode_docs_ragged_serial(&tok, &texts);
            let chunks = sp_build_chunks(&tok, &texts, 48);
            assert!(
                chunks.len() > 1000,
                "expected a cut attempt every few bytes, got {} chunks",
                chunks.len()
            );
            let (ids, lens) = sp_encode_chunks(&tok, &chunks);
            assert_eq!(lens, lens_ref, "{prepend:?}: lens mismatch");
            assert_ids_match(&format!("{prepend:?}"), &ids, &ids_ref);
        }
    }

    /// SentencePiece parallel-vs-serial on ~290 MB of OWT (owt_valid).
    /// `cargo test --release verify_sp_parallel_matches_serial_owt -- --ignored --nocapture`
    #[test]
    #[ignore = "reads ~290 MB of OWT; run explicitly in release mode"]
    fn verify_sp_parallel_matches_serial_owt() {
        let path = std::env::home_dir().unwrap().join("data/owt_valid.txt");
        let input = std::fs::read(&path).expect("read ~/data/owt_valid.txt");
        let text = match std::str::from_utf8(&input) {
            Ok(t) => t,
            Err(e) => std::str::from_utf8(&input[..e.valid_up_to()]).unwrap(),
        };
        // One huge doc plus a few multi-MB slices, so grouped-doc and
        // fragment chunks both appear.
        let mut texts: Vec<&str> = vec![text];
        let mut off = 0usize;
        for mb in [3, 7, 12] {
            let mut end = (off + (mb << 20)).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            texts.push(&text[off..end]);
            off = end;
        }
        for repo in [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "unsloth/gemma-2b",
            "google/gemma-3-4b-it",
        ] {
            let Some(path) = crate::test_hub::hf_tokenizer_json(repo) else {
                eprintln!("Skipping {repo}: tokenizer.json not in the HF cache");
                continue;
            };
            let tok = crate::load_tokenizer::hf::load_hf_sentencepiece(&path).unwrap();
            let t0 = std::time::Instant::now();
            let (ids_ref, lens_ref) = sp_encode_docs_ragged_serial(&tok, &texts);
            let t_serial = t0.elapsed();
            let t0 = std::time::Instant::now();
            let (ids, lens) = sp_encode_docs_ragged(&tok, &texts);
            let t_par = t0.elapsed();
            eprintln!(
                "{repo}: {} tokens; serial {:.2?}, parallel {:.2?}",
                ids_ref.len(),
                t_serial,
                t_par
            );
            assert_eq!(lens, lens_ref, "{repo}: lens mismatch");
            assert_ids_match(repo, &ids, &ids_ref);
        }
    }

    /// Parallel-vs-serial on ~1 GB of OWT with GPT-2: mixed doc sizes,
    /// oversized docs that fragment, `<|endoftext|>` injected mid-doc and
    /// doc-final, LPT on and off.
    /// `cargo test --release verify_parallel_ragged_matches_serial_owt_gpt2_1g -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 1 GB of OWT; run explicitly in release mode"]
    fn verify_parallel_ragged_matches_serial_owt_gpt2_1g() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        use std::io::Read;
        let tokenizer_path = crate::test_hub::gpt2_tokenizer_json();
        let proto = load_hf_bpe(&tokenizer_path).expect("load GPT-2 tokenizer");
        let added = proto.added_token_split_blockers();
        let sep: Vec<u8> = added.first().expect("GPT-2 has an added token").0.to_vec();

        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let f = std::fs::File::open(&path).expect("open ~/data/owt_train.txt");
        let mut input = Vec::new();
        f.take(1_000_000_000).read_to_end(&mut input).unwrap();
        while !input.is_empty() && std::str::from_utf8(&input).is_err() {
            input.pop();
        }
        assert!(input.len() > 900_000_000, "corpus too small: {}", input.len());

        // Doc size pattern: small (group), mid, large; every 20th doc is
        // oversized (24 MB) so it splits into Fragment chunks.
        let sizes = [64 << 10, 300 << 10, 1 << 20, 100 << 10, 3 << 20];
        let mut owned: Vec<Vec<u8>> = Vec::new(); // docs with injected added tokens
        let mut ranges: Vec<(usize, usize, bool)> = Vec::new(); // (start, end, inject)
        let mut pos = 0usize;
        let mut i = 0usize;
        while pos < input.len() {
            let want = if i % 20 == 19 { 24 << 20 } else { sizes[i % sizes.len()] };
            let end = (pos + want).min(input.len());
            // Inject the added token into every 7th doc (mid + tail).
            ranges.push((pos, end, i % 7 == 3));
            pos = end;
            i += 1;
        }
        for &(s, e, inject) in &ranges {
            if inject {
                let piece = &input[s..e];
                let mid = piece.len() / 2;
                let mut doc = Vec::with_capacity(piece.len() + 2 * sep.len());
                doc.extend_from_slice(&piece[..mid]);
                doc.extend_from_slice(&sep);
                doc.extend_from_slice(&piece[mid..]);
                doc.extend_from_slice(&sep);
                owned.push(doc);
            }
        }
        let mut docs: Vec<&[u8]> = Vec::with_capacity(ranges.len());
        let mut oi = 0usize;
        for &(s, e, inject) in &ranges {
            if inject {
                docs.push(&owned[oi]);
                oi += 1;
            } else {
                docs.push(&input[s..e]);
            }
        }
        eprintln!(
            "{} docs ({} with injected {:?}), {} bytes total",
            docs.len(),
            owned.len(),
            String::from_utf8_lossy(&sep),
            docs.iter().map(|d| d.len()).sum::<usize>()
        );

        let mut ids_ref: Vec<u32> = Vec::new();
        let mut lens_ref: Vec<i64> = Vec::new();
        let mut serial = proto.fork();
        for doc in &docs {
            encode_into(&mut serial, doc, &mut ids_ref, &mut lens_ref);
        }
        drop(serial);
        eprintln!("serial reference: {} tokens", ids_ref.len());

        for lpt in [true, false] {
            let workers = WorkerPool::new();
            let (flat, lens) = encode_docs_ragged_with(&workers, &proto, &docs, lpt);
            assert_eq!(lens, lens_ref, "lens mismatch (lpt={lpt})");
            assert_ids_match(&format!("lpt={lpt}"), &flat, &ids_ref);
        }
    }

    /// The overlapped gather's escape hatches must match the committed
    /// path: cap 0 is a refused reservation, cap 1 overflows on the first
    /// commit, and a mid-range cap overflows after a real prefix has been
    /// committed.
    #[test]
    fn gather_fallbacks_match() {
        let proto = byte_proto();
        let mut text = lcg_text(0xD1B54A32D192ED03);
        let owned: Vec<Vec<u8>> = (0..20).map(|_| text(1 << 20)).collect();
        let docs: Vec<&[u8]> = owned.iter().map(|d| d.as_slice()).collect();
        let total: usize = docs.iter().map(|d| d.len()).sum();
        let added = proto.added_token_split_blockers();
        let chunks = build_doc_chunks(&docs, total, chunk_target_bytes(total), &added, true);
        assert!(chunks.len() > 1, "test must exercise the parallel path");

        let workers = WorkerPool::new();
        let (flat_ref, lens_ref) = encode_chunks_gathered(&workers, &proto, &chunks, total);
        for cap in [0, 1, total / 3] {
            let workers = WorkerPool::new();
            let (flat, lens) =
                encode_chunks_gathered_with_cap(&workers, &proto, &chunks, total, cap);
            assert_eq!(lens, lens_ref, "lens mismatch (cap={cap})");
            assert_eq!(flat, flat_ref, "ids mismatch (cap={cap})");
        }
    }
}
