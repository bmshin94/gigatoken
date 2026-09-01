use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::input::jsonl::{JsonLinesReader, JsonLinesSlice};
use crate::input::{MmappedFile, PretokenCounts, chunk_boundaries, count_pretokens, decompress, merge_counts};
use crate::pretokenize::pretokenize_par_bytes;

#[derive(Clone, Copy)]
pub(crate) enum Compression {
    None,
    Gzip,
    Zstd,
}

/// Strip a compression extension: (stem, compression).
fn detect_compression(name: &str) -> (&str, Compression) {
    if let Some(stem) = name.strip_suffix(".zst").or_else(|| name.strip_suffix(".zstd")) {
        (stem, Compression::Zstd)
    } else if let Some(stem) = name.strip_suffix(".gz") {
        (stem, Compression::Gzip)
    } else {
        (name, Compression::None)
    }
}

fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("")
}

fn compression_of(path: &Path) -> Compression {
    detect_compression(file_name(path)).1
}

/// How a file's bytes split into documents. Compression (.gz/.zst) is
/// orthogonal and always detected from the file extension.
#[derive(Debug, Clone)]
pub enum DocFormat {
    /// Plain text: the pieces between separator occurrences, or the whole
    /// file as one document without a separator.
    Text { separator: Option<Vec<u8>> },
    /// JSON Lines: one document per line, text taken from `field`.
    Jsonl { field: String },
    /// Parquet: one document per row, text taken from `column` (null rows
    /// become empty documents). Rows are materialized up front (see
    /// `input::parquet`) and never reach the byte-region paths.
    Parquet { column: String },
}

impl DocFormat {
    /// Separator to split text documents on; empty means one document.
    fn separator(&self) -> &[u8] {
        match self {
            DocFormat::Text { separator } => separator.as_deref().unwrap_or(b""),
            DocFormat::Jsonl { .. } | DocFormat::Parquet { .. } => b"",
        }
    }
}

/// Default format for a bare path: JSONL (field "text") for .jsonl, parquet
/// (column "text") for .parquet, otherwise the whole file is one document.
pub fn detect_default_format(path: &Path) -> DocFormat {
    let (stem, _) = detect_compression(file_name(path));
    if stem.ends_with(".jsonl") {
        DocFormat::Jsonl { field: "text".to_string() }
    } else if stem.ends_with(".parquet") {
        DocFormat::Parquet { column: "text".to_string() }
    } else {
        DocFormat::Text { separator: None }
    }
}

/// A file's full contents: mmapped when stored uncompressed, otherwise
/// decompressed into memory (parallel chunking needs random access).
pub enum LoadedFile {
    Mmapped(MmappedFile),
    Owned(Vec<u8>),
}

impl LoadedFile {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            LoadedFile::Mmapped(m) => m.as_bytes(),
            LoadedFile::Owned(v) => v,
        }
    }
}

/// Open a file for encoding: mmap if uncompressed, else decompress fully.
pub fn load_file(path: &Path) -> Result<LoadedFile, std::io::Error> {
    use std::io::Read;
    let mut buf = Vec::new();
    match compression_of(path) {
        Compression::None => return Ok(LoadedFile::Mmapped(MmappedFile::open(path)?)),
        Compression::Gzip => decompress::open_gzip(path)?.read_to_end(&mut buf)?,
        Compression::Zstd => decompress::open_zstd(path)?.read_to_end(&mut buf)?,
    };
    Ok(LoadedFile::Owned(buf))
}

/// Cut `bytes` into ranges of roughly `target` bytes, each ending on a
/// document boundary. Plain text without a separator is one document and
/// stays one range.
pub fn chunk_ranges(bytes: &[u8], format: &DocFormat, target: usize) -> Vec<std::ops::Range<usize>> {
    let len = bytes.len();
    // `next_boundary(probe)` finds the first document boundary at or after
    // `probe` and returns (chunk_end, next_chunk_start).
    let cut = |next_boundary: &dyn Fn(usize) -> Option<(usize, usize)>| {
        let mut out = Vec::new();
        let mut start = 0;
        while start < len {
            let probe = start + target;
            match (probe < len).then(|| next_boundary(probe)).flatten() {
                Some((end, next_start)) => {
                    out.push(start..end);
                    start = next_start;
                }
                None => {
                    out.push(start..len);
                    break;
                }
            }
        }
        if out.is_empty() {
            out.push(0..0); // empty file: one empty chunk, so files stay 1:1
        }
        out
    };
    match format {
        DocFormat::Jsonl { .. } => cut(&|probe| {
            memchr::memchr(b'\n', &bytes[probe..]).map(|off| (probe + off + 1, probe + off + 1))
        }),
        DocFormat::Text { separator: Some(sep) } if !sep.is_empty() => {
            let finder = memchr::memmem::Finder::new(sep);
            cut(&|probe| finder.find(&bytes[probe..]).map(|off| (probe + off, probe + off + sep.len())))
        }
        DocFormat::Text { .. } => vec![0..len],
        DocFormat::Parquet { .. } => {
            unreachable!("parquet files are materialized into documents before chunking")
        }
    }
}

/// Parallel JSONL pretokenization of an mmapped file: newline-aligned
/// chunks, one rayon task each.
fn pretokenize_jsonl_par(bytes: &[u8], field: &str) -> PretokenCounts {
    chunk_boundaries(bytes, b"\n", rayon::current_num_threads())
        .par_windows(2)
        .map(|w| {
            let mut counts = PretokenCounts::default();
            for doc in JsonLinesSlice::new(&bytes[w[0]..w[1]], field) {
                count_pretokens(&mut counts, &doc);
            }
            counts
        })
        .reduce(PretokenCounts::default, merge_counts)
}

/// Pretokenize documents from a streaming reader without buffering the
/// whole (decompressed) file.
fn pretokenize_streaming(reader: impl std::io::BufRead, format: &DocFormat) -> PretokenCounts {
    let mut counts = PretokenCounts::default();
    match format {
        DocFormat::Jsonl { field } => {
            for doc in JsonLinesReader::new(reader, field) {
                count_pretokens(&mut counts, &doc);
            }
        }
        DocFormat::Text { .. } => {
            for doc in SeparatorReader::new(reader, format.separator()) {
                count_pretokens(&mut counts, &doc);
            }
        }
        DocFormat::Parquet { .. } => {
            unreachable!("parquet files are pretokenized by input::parquet, not streamed")
        }
    }
    counts
}

fn pretokenize_file(path: &Path, format: &DocFormat) -> Result<PretokenCounts, std::io::Error> {
    if let DocFormat::Parquet { column } = format {
        return crate::input::parquet::pretokenize_par(path, column);
    }
    let with_path = |e: std::io::Error| std::io::Error::new(e.kind(), format!("{}: {e}", path.display()));
    Ok(match compression_of(path) {
        Compression::None => {
            let resource = MmappedFile::open(path).map_err(with_path)?;
            match format {
                DocFormat::Jsonl { field } => pretokenize_jsonl_par(resource.as_bytes(), field),
                _ => pretokenize_par_bytes(resource.as_bytes(), format.separator())
                    .into_iter()
                    .map(|(k, v)| (k.as_ref().to_vec(), v))
                    .collect(),
            }
        }
        Compression::Gzip => pretokenize_streaming(decompress::open_gzip(path).map_err(with_path)?, format),
        Compression::Zstd => pretokenize_streaming(decompress::open_zstd(path).map_err(with_path)?, format),
    })
}

/// Yields the documents of a `BufRead` split on a byte separator, buffering
/// only the current document.
struct SeparatorReader<R> {
    reader: R,
    separator: Vec<u8>,
    finder: memchr::memmem::Finder<'static>,
    buf: Vec<u8>,
    /// Where the next search starts; everything before it has been searched.
    search_from: usize,
    finished: bool,
}

impl<R: std::io::BufRead> SeparatorReader<R> {
    fn new(reader: R, separator: &[u8]) -> Self {
        Self {
            reader,
            separator: separator.to_vec(),
            finder: memchr::memmem::Finder::new(separator).into_owned(),
            buf: Vec::with_capacity(4096),
            search_from: 0,
            finished: false,
        }
    }
}

impl<R: std::io::BufRead> Iterator for SeparatorReader<R> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        if self.finished {
            return None;
        }
        if self.separator.is_empty() {
            self.finished = true;
            let mut all = Vec::new();
            self.reader.read_to_end(&mut all).ok()?;
            return if all.is_empty() { None } else { Some(all) };
        }
        loop {
            if let Some(pos) = self.finder.find(&self.buf[self.search_from..]) {
                let sep_start = self.search_from + pos;
                let doc = self.buf[..sep_start].to_vec();
                self.buf.drain(..sep_start + self.separator.len());
                self.search_from = 0;
                if !doc.is_empty() {
                    return Some(doc);
                }
                continue;
            }
            // A separator may straddle the searched tail and the next read.
            self.search_from = self.buf.len().saturating_sub(self.separator.len() - 1);
            let available = match self.reader.fill_buf() {
                Ok([]) => {
                    self.finished = true;
                    return if self.buf.is_empty() { None } else { Some(std::mem::take(&mut self.buf)) };
                }
                Ok(buf) => buf,
                Err(_) => {
                    self.finished = true;
                    return None;
                }
            };
            self.buf.extend_from_slice(available);
            let consumed = available.len();
            self.reader.consume(consumed);
        }
    }
}

/// Multi-file parallel pretokenization for BPE training.
pub struct FileSourceSpec {
    pub paths: Vec<PathBuf>,
    pub format: DocFormat,
}

impl FileSourceSpec {
    pub fn pretokenize(&self) -> Result<PretokenCounts, std::io::Error> {
        self.paths
            .par_iter()
            .map(|path| pretokenize_file(path, &self.format))
            .try_reduce(PretokenCounts::default, |acc, counts| Ok(merge_counts(acc, counts)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_compression() {
        assert!(matches!(detect_compression("data.jsonl.zst"), ("data.jsonl", Compression::Zstd)));
        assert!(matches!(detect_compression("data.jsonl.zstd"), ("data.jsonl", Compression::Zstd)));
        assert!(matches!(detect_compression("data.txt.gz"), ("data.txt", Compression::Gzip)));
        assert!(matches!(detect_compression("data.jsonl"), ("data.jsonl", Compression::None)));
        assert!(matches!(detect_compression("data.txt"), ("data.txt", Compression::None)));
    }

    #[test]
    fn test_detect_default_format() {
        for name in ["data.jsonl.zst", "data.jsonl.gz", "data.jsonl"] {
            assert!(matches!(detect_default_format(Path::new(name)), DocFormat::Jsonl { .. }), "{name}");
        }
        assert!(matches!(detect_default_format(Path::new("rows.parquet")), DocFormat::Parquet { .. }));
        for name in ["data.txt.zst", "data.txt.gz", "data.txt", "data.zst", "data.gz", "data.csv"] {
            assert!(matches!(detect_default_format(Path::new(name)), DocFormat::Text { separator: None }), "{name}");
        }
        assert!(matches!(compression_of(Path::new("data.zst")), Compression::Zstd));
        assert!(matches!(compression_of(Path::new("data.csv")), Compression::None));
    }

    #[test]
    fn test_separator_reader() {
        let cases: [(&[u8], &[&[u8]]); 5] = [
            (b"aaa<SEP>bbb<SEP><SEP>ccc", &[b"aaa", b"bbb", b"ccc"]),
            (b"<SEP>hello<SEP>", &[b"hello"]),
            (b"<SEP><SEP>", &[]),
            (b"a<SEP>b<SEP>c", &[b"a", b"b", b"c"]),
            (b"", &[]),
        ];
        // Small capacities force separators across reads; the default
        // delivers several per read.
        for cap in [1, 2, 3, 7, 8192] {
            for (data, want) in cases {
                let reader = std::io::BufReader::with_capacity(cap, data);
                let docs: Vec<Vec<u8>> = SeparatorReader::new(reader, b"<SEP>").collect();
                assert_eq!(docs, want, "cap {cap}: {:?}", String::from_utf8_lossy(data));
            }
        }
        let reader = std::io::BufReader::new(&b"one document"[..]);
        let docs: Vec<Vec<u8>> = SeparatorReader::new(reader, b"").collect();
        assert_eq!(docs, [b"one document".to_vec()]);
    }
}
