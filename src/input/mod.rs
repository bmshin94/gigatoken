//! Byte-level input: memory-mapped files, splitting a buffer into documents
//! on a separator, and separator-aligned chunking for parallel work.

use memchr::memmem;
use memmap2::Mmap;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::path::Path;

pub(crate) mod decompress;
pub mod file_source;
pub mod jsonl;
pub mod parquet;

pub(crate) type PretokenCounts = HashMap<Vec<u8>, usize, FxBuildHasher>;

/// Add every pretoken of `doc` to `counts`.
pub(crate) fn count_pretokens(counts: &mut PretokenCounts, doc: &[u8]) {
    for pretoken in crate::pretokenize::pretokenize_as_iter(doc) {
        *counts.entry(pretoken.as_ref().to_vec()).or_default() += 1;
    }
}

/// Fold `counts` into `acc` (the reduce step of parallel counting).
pub(crate) fn merge_counts(mut acc: PretokenCounts, counts: PretokenCounts) -> PretokenCounts {
    if acc.is_empty() {
        return counts;
    }
    for (k, v) in counts {
        *acc.entry(k).or_default() += v;
    }
    acc
}

/// Owns a memory-mapped file.
pub struct MmappedFile {
    mmap: Mmap,
}

impl MmappedFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

/// Zero-copy iterator over the documents of a byte slice split on a
/// separator; empty documents are skipped, and an empty separator means the
/// whole slice is one document.
pub struct DocumentIter<'a> {
    bytes: &'a [u8],
    separator: &'a [u8],
    finder: memmem::Finder<'a>,
    position: usize,
    end: usize,
}

impl<'a> DocumentIter<'a> {
    pub fn new(bytes: &'a [u8], separator: &'a [u8]) -> Self {
        Self::new_range(bytes, separator, 0, bytes.len())
    }

    fn new_range(bytes: &'a [u8], separator: &'a [u8], start: usize, end: usize) -> Self {
        Self { bytes, separator, finder: memmem::Finder::new(separator), position: start, end }
    }
}

impl<'a> Iterator for DocumentIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        while self.position < self.end {
            let rest = &self.bytes[self.position..self.end];
            let found = if self.separator.is_empty() { None } else { self.finder.find(rest) };
            let doc = match found {
                Some(off) => {
                    self.position += off + self.separator.len();
                    &rest[..off]
                }
                None => {
                    self.position = self.end;
                    rest
                }
            };
            if !doc.is_empty() {
                return Some(doc);
            }
        }
        None
    }
}

/// Boundaries cutting `bytes` into up to `n` ranges, each ending right after
/// a separator occurrence, as `[0, b1, .., len]`.
pub(crate) fn chunk_boundaries(bytes: &[u8], separator: &[u8], n: usize) -> Vec<usize> {
    let mut boundaries = vec![0usize];
    if n > 1 && !bytes.is_empty() && !separator.is_empty() {
        let chunk_size = bytes.len() / n;
        let finder = memmem::Finder::new(separator);
        for i in 1..n {
            let target = i * chunk_size;
            match finder.find(&bytes[target..]) {
                Some(offset) => boundaries.push(target + offset + separator.len()),
                None => break,
            }
        }
    }
    boundaries.push(bytes.len());
    boundaries.dedup();
    boundaries
}

/// Up to `n` document iterators over disjoint, separator-aligned ranges.
pub(crate) fn par_document_chunks<'a>(bytes: &'a [u8], separator: &'a [u8], n: usize) -> Vec<DocumentIter<'a>> {
    chunk_boundaries(bytes, separator, n)
        .windows(2)
        .map(|w| DocumentIter::new_range(bytes, separator, w[0], w[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs<'a>(data: &'a [u8], sep: &'a [u8]) -> Vec<&'a [u8]> {
        DocumentIter::new(data, sep).collect()
    }

    #[test]
    fn test_document_iter() {
        assert_eq!(docs(b"hello<|endoftext|>world<|endoftext|>foo", b"<|endoftext|>"), [b"hello".as_slice(), b"world", b"foo"]);
        assert_eq!(docs(b"hello world", b"<|endoftext|>"), [b"hello world".as_slice()]);
        assert_eq!(docs(b"hello world", b""), [b"hello world".as_slice()]);
        assert_eq!(docs(b"a<SEP><SEP>b", b"<SEP>"), [b"a".as_slice(), b"b"]);
        assert_eq!(docs(b"<SEP>hello<SEP>", b"<SEP>"), [b"hello".as_slice()]);
        assert!(docs(b"", b"<SEP>").is_empty());
    }

    #[test]
    fn test_par_document_chunks() {
        let sep = b"<|endoftext|>";
        let parts: Vec<&str> = (0..100).map(|i| if i % 10 == 9 { "doc" } else { "word " }).collect();
        let data = parts.join(std::str::from_utf8(sep).unwrap());
        let bytes = data.as_bytes();
        let all_docs: Vec<&[u8]> = par_document_chunks(bytes, sep, 4).into_iter().flatten().collect();
        let single_docs: Vec<&[u8]> = DocumentIter::new(bytes, sep).collect();
        assert_eq!(all_docs, single_docs);

        let chunks = par_document_chunks(b"a<SEP>b<SEP>c", b"<SEP>", 1);
        assert_eq!(chunks.len(), 1);
        let docs: Vec<&[u8]> = chunks.into_iter().flatten().collect();
        assert_eq!(docs, [b"a".as_slice(), b"b", b"c"]);
    }
}
