//! JSON Lines documents: one per line, text taken from a named field.

use sonic_rs::JsonValueTrait;
use std::io::BufRead;

/// The `field` string of one JSONL line as bytes, or None when the line
/// does not parse or the field is not a string.
fn field_text(line: &[u8], field: &str) -> Option<Vec<u8>> {
    let value = sonic_rs::get_from_slice(line, &[field]).ok()?;
    Some(value.as_str()?.as_bytes().to_vec())
}

/// JSONL iterator over a byte slice (e.g. an mmap).
pub struct JsonLinesSlice<'a> {
    slice: &'a [u8],
    position: usize,
    field: &'a str,
}

impl<'a> JsonLinesSlice<'a> {
    pub fn new(slice: &'a [u8], field: &'a str) -> Self {
        Self { slice, position: 0, field }
    }
}

impl Iterator for JsonLinesSlice<'_> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        while self.slice.get(self.position) == Some(&b'\n') {
            self.position += 1;
        }
        if self.position >= self.slice.len() {
            return None;
        }
        let line_end = memchr::memchr(b'\n', &self.slice[self.position..])
            .map_or(self.slice.len(), |i| self.position + i);
        let line = &self.slice[self.position..line_end];
        self.position = line_end + 1;
        field_text(line, self.field)
    }
}

/// Streaming JSONL iterator over a `BufRead`, one line in memory at a time.
pub(crate) struct JsonLinesReader<R> {
    reader: R,
    field: String,
    line_buf: Vec<u8>,
}

impl<R: BufRead> JsonLinesReader<R> {
    pub(crate) fn new(reader: R, field: &str) -> Self {
        Self { reader, field: field.to_string(), line_buf: Vec::with_capacity(4096) }
    }
}

impl<R: BufRead> Iterator for JsonLinesReader<R> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        loop {
            self.line_buf.clear();
            if self.reader.read_until(b'\n', &mut self.line_buf).ok()? == 0 {
                return None;
            }
            if !self.line_buf.iter().all(|&b| b == b'\n' || b == b'\r') {
                return field_text(&self.line_buf, &self.field);
            }
        }
    }
}
