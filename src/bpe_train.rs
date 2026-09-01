use dashmap::DashMap;
use indicatif::ProgressBar;
use itertools::Itertools;
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;

struct Word {
    symbols: Vec<u32>,
    word_count: isize,
}

type Pair = (u32, u32);

fn count_pairs(words: &[Word]) -> HashMap<Pair, isize> {
    let mut symbol_counts: HashMap<Pair, isize> = HashMap::new();
    for word in words.iter() {
        for w in word.symbols.windows(2) {
            *symbol_counts.entry((w[0], w[1])).or_insert(0) += word.word_count;
        }
    }
    symbol_counts
}

fn update_word(
    w: &mut Word,
    pair: Pair,
    new_symbol: u32,
    mut record_changes: impl FnMut(Pair, isize),
) {
    let mut i = 0;
    while i + 1 < w.symbols.len() {
        if w.symbols[i] == pair.0 && w.symbols[i + 1] == pair.1 {
            // Perform the merge
            if i >= 1 {
                record_changes((w.symbols[i - 1], pair.0), -w.word_count);
                record_changes((w.symbols[i - 1], new_symbol), w.word_count);
            }
            if w.symbols.len() >= 3 && i <= w.symbols.len() - 3 {
                record_changes((pair.1, w.symbols[i + 2]), -w.word_count);
                record_changes((new_symbol, w.symbols[i + 2]), w.word_count);
            }
            w.symbols[i] = new_symbol;
            w.symbols.remove(i + 1);
        }
        i += 1;
    }
}

/// Raw pointer into the words array so parallel workers can each mutate
/// their own words; sound only while every index is visited by one worker.
struct SendPtr(*mut Word);

unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

impl SendPtr {
    /// SAFETY: the caller must be the only one touching word `i`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn word(&self, i: u32) -> &mut Word {
        unsafe { &mut *self.0.add(i as usize) }
    }
}

/// Merge `pair` into a new symbol in every word containing it. Adds the new
/// pairs to `contained_in_words` (stale entries are left in place) and
/// returns pair -> count change for the priority queue.
fn update_words(
    words: &mut [Word],
    contained_in_words: &mut HashMap<Pair, BTreeSet<u32>>,
    pair: Pair,
    new_symbol: u32,
) -> DashMap<Pair, isize, FxBuildHasher> {
    let count_changes: DashMap<Pair, isize, FxBuildHasher> = DashMap::default();

    let n_threads = rayon::current_num_threads();

    // Iterate through all words containing first or second
    let word_idcs = &contained_in_words[&(pair.0, pair.1)];
    let words_ptr = SendPtr(words.as_mut_ptr());

    // Pair -> words the pair was added to (contended early in merging, when
    // updated pairs overlap a lot).
    let contained_updates: DashMap<Pair, BTreeSet<u32>, FxBuildHasher> = DashMap::default();

    let process = |i: u32| {
        // SAFETY: word_idcs is a set of unique indices, so only this call touches word `i`.
        let word = unsafe { words_ptr.word(i) };
        update_word(word, pair, new_symbol, |pair, change| {
            if change > 0 {
                // Added to the word: track immediately, other threads might subtract.
                contained_updates.entry(pair).or_default().insert(i);
            }
            *count_changes.entry(pair).or_default() += change;
        });
    };
    if word_idcs.len() > 2 * n_threads {
        word_idcs
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .par_chunks(word_idcs.len().div_ceil(n_threads))
            .for_each(|idcs_chunk| idcs_chunk.iter().copied().for_each(&process));
    } else {
        word_idcs.iter().copied().for_each(&process);
    }

    for (pair, mut word_idcs) in contained_updates.into_iter() {
        let set = contained_in_words.entry(pair).or_default();
        set.append(&mut word_idcs);
    }

    count_changes
}

pub struct BPEResult {
    /// Token bytes by ID (dense: `0..vocab_size`).
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

/// How to break ties when multiple pairs have the same frequency count.
#[derive(Clone, Copy, Debug, Default)]
pub enum TieBreaking {
    /// Compare by token IDs remapped to match HuggingFace tokenizers' BpeTrainer
    /// initial vocabulary ordering (ByteLevel unicode codepoint order for bytes 0-255).
    #[default]
    HuggingFace,
    /// Compare by raw (u32, u32) token IDs (byte value = token ID).
    RawTokenIds,
    /// Compare each token's bytes lexicographically.
    AssembledBytes,
}

/// Build a mapping from byte value (0-255) to the rank it would receive in
/// HuggingFace's BpeTrainer initial vocabulary. HF sorts the ByteLevel alphabet
/// by unicode codepoint: printable ASCII/Latin-1 bytes keep their codepoint,
/// while the remaining 68 bytes are remapped to U+0100..U+0143.
fn build_byte_to_hf_rank() -> [u32; 256] {
    let mut byte_to_cp = [0u32; 256];
    let mut n = 0u32;
    for b in 0..=255u8 {
        let is_allowed = matches!(b, 33..=126 | 161..=172 | 174..=255);
        if is_allowed {
            byte_to_cp[b as usize] = b as u32;
        } else {
            byte_to_cp[b as usize] = 256 + n;
            n += 1;
        }
    }

    // Sort bytes by their unicode codepoint, then record position as rank
    let mut bytes_sorted: Vec<u8> = (0..=255).collect();
    bytes_sorted.sort_by_key(|&b| byte_to_cp[b as usize]);

    let mut rank = [0u32; 256];
    for (i, &b) in bytes_sorted.iter().enumerate() {
        rank[b as usize] = i as u32;
    }
    rank
}

pub fn train_bpe<K: AsRef<[u8]> + Eq + Hash>(
    counts: HashMap<K, usize, FxBuildHasher>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    tie_breaking: TieBreaking,
) -> BPEResult {
    // Indicates which word indices contain a given symbol
    let mut contained_in_words: HashMap<Pair, BTreeSet<u32>> = HashMap::new();
    let mut contained_in_words_arr = vec![vec![vec![]; 256]; 256];
    let mut words: Vec<Word> = counts
        .into_iter()
        .enumerate()
        .map(|(word_i, (word, count))| {
            // At first we have only bytes, so we won't need to hash the u32 pairs
            let word_symbols: Vec<u32> = word.as_ref().iter().map(|&b| b as u32).collect();
            for c in word_symbols.iter().copied().tuple_windows::<Pair>() {
                contained_in_words_arr[c.0 as usize][c.1 as usize].push(word_i as u32);
            }
            Word {
                symbols: word_symbols,
                word_count: count as isize,
            }
        })
        .collect();

    for (i, j) in (0..256).cartesian_product(0..256) {
        if !contained_in_words_arr[i][j].is_empty() {
            contained_in_words.insert(
                (i as u32, j as u32),
                BTreeSet::from_iter(contained_in_words_arr[i][j].iter().copied()),
            );
        }
    }
    drop(contained_in_words_arr);

    let symbol_counts = count_pairs(&words);

    // Symbols 0 through 255 are unicode characters
    let mut symbols: Vec<Vec<u8>> = (0..=255).map(|x| vec![x]).collect();
    symbols.extend(
        special_tokens
            .into_iter()
            .map(|x| x.bytes().collect::<Vec<u8>>()),
    );

    // Build HF rank table for tie-breaking (only used in HuggingFace mode)
    let hf_rank = build_byte_to_hf_rank();
    let remap = |id: u32| if id < 256 { hf_rank[id as usize] } else { id };

    let mut pq = PriorityQueue::new();
    symbol_counts.into_iter().for_each(|(pair, count)| {
        pq.push(pair, count);
    });

    let mut merges = vec![];

    let bar = ProgressBar::new(vocab_size as u64).with_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    while !pq.is_empty() && symbols.len() < vocab_size {
        bar.set_position(symbols.len() as u64);
        let pair = {
            let (first_pair, first_count) = pq.pop().unwrap();
            let mut tied_pairs = vec![first_pair];
            while let Some((_next_pair, &next_count)) = pq.peek() {
                if next_count != first_count {
                    break;
                }
                tied_pairs.push(pq.pop().unwrap().0);
            }
            // The smallest pair under the chosen tie-breaking rule (first on
            // equal keys, like the original scan).
            let tied = tied_pairs.iter().copied();
            let smallest_pair = match tie_breaking {
                TieBreaking::HuggingFace => tied.min_by_key(|&p| (remap(p.0), remap(p.1))),
                TieBreaking::RawTokenIds => tied.min(),
                TieBreaking::AssembledBytes => {
                    tied.min_by_key(|&p| (&symbols[p.0 as usize], &symbols[p.1 as usize]))
                }
            }
            .unwrap();

            for pair in tied_pairs {
                if pair != smallest_pair {
                    pq.push(pair, first_count);
                }
            }

            smallest_pair
        };

        // Merge the pair
        let new_symbol: Vec<u8> = [&symbols[pair.0 as usize], &symbols[pair.1 as usize]]
            .into_iter()
            .flatten()
            .copied()
            .collect();

        merges.push((
            symbols[pair.0 as usize].clone(),
            symbols[pair.1 as usize].clone(),
        ));

        symbols.push(new_symbol);

        let count_changes = update_words(
            &mut words,
            &mut contained_in_words,
            pair,
            symbols.len() as u32 - 1,
        );

        for (pair, change) in count_changes.into_iter() {
            let found_item = pq.change_priority_by(&pair, |p| *p += change);
            if !found_item {
                pq.push(pair, change);
            }
        }
    }
    bar.finish();

    BPEResult { vocab: symbols, merges }
}
