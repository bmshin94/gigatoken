#![feature(portable_simd)]

pub(crate) mod batch;
pub(crate) mod bindings;
pub(crate) mod bpe;
pub(crate) mod bpe_train;
pub(crate) mod input;
pub mod pretokenize;
#[cfg(test)]
pub(crate) mod test_hub;
pub(crate) mod token;
pub use crate::batch::{WorkerPool, encode_docs_ragged, sp_encode_docs_ragged};
pub use crate::bpe::Tokenizer;
pub use crate::bpe::sentencepiece::EncodeState;
pub mod load_tokenizer;

use crate::batch::{
    encode_files_docs, encode_files_docs_serial, encode_into, sp_encode_files_docs,
    sp_encode_files_docs_serial,
};
use crate::bindings::bridge::{
    EncodeInput, encode_batch_pylist, encode_batch_ragged, extract_doc, extract_token_ids,
    merges_to_pylist, vocab_to_pydict,
};
use crate::bindings::matcher::{SpecialTokenFound, SubstringMatcher};
use crate::bindings::padding;
use crate::bindings::pretokenize::{
    PretokenizerIter, pretokenized_counts, pretokenizer, pretokenizer_scheme,
};
use crate::bindings::sources::{
    BytesSource, FileSource, JsonlFileSource, ParquetFileSource, TextFileSource,
    encode_files_ragged,
};
use crate::bindings::train::train_bpe;
use crate::input::file_source::DocFormat;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;

#[pyclass]
struct BPETokenizer {
    tokenizer: Tokenizer,
    workers: WorkerPool,
}

impl BPETokenizer {
    /// Encode byte regions that split into documents per `format` (one
    /// document per region for plain batches, separator pieces for a
    /// BytesSource). Call with the GIL released.
    fn encode_regions_ragged(
        &self,
        regions: &[&[u8]],
        format: &DocFormat,
        parallel: bool,
    ) -> (Vec<u32>, Vec<i64>) {
        if parallel {
            encode_files_docs(&self.workers, &self.tokenizer, regions, format)
        } else {
            encode_files_docs_serial(&self.workers, &self.tokenizer, regions, format)
        }
    }
}

#[pymethods]
impl BPETokenizer {
    #[staticmethod]
    #[pyo3(signature = (path, pretokenizer, special_tokens = None))]
    fn from_tiktoken(
        path: PathBuf,
        pretokenizer: &str,
        special_tokens: Option<HashMap<String, u32>>,
    ) -> PyResult<Self> {
        let scheme = pretokenizer_scheme(pretokenizer)?;
        let special_tokens = special_tokens.unwrap_or_default().into_iter().collect();
        Ok(Self {
            tokenizer: bindings::cache::apply_max_cache_bytes(
                load_tokenizer::tiktoken::load_tiktoken(&path, scheme, special_tokens)?,
            ),
            workers: WorkerPool::new(),
        })
    }

    #[staticmethod]
    fn from_hf(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: bindings::cache::apply_max_cache_bytes(load_tokenizer::hf::load_hf_bpe(
                &path,
            )?),
            workers: WorkerPool::new(),
        })
    }

    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let input = extract_doc(&input)?;
        let (mut ids, mut lens) = (Vec::new(), Vec::new());
        encode_into(&mut self.tokenizer, input.as_bytes(), &mut ids, &mut lens);
        Ok(ids.into_pyarray(py))
    }

    #[pyo3(signature = (inputs, *, parallel = true))]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_batch_ragged(py, &inputs, |docs, format| {
            Ok(self.encode_regions_ragged(docs, format, parallel))
        })
    }

    #[pyo3(signature = (inputs, *, parallel = true))]
    fn encode_batch_list<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        encode_batch_pylist(py, &inputs, None, parallel, |docs, format| {
            Ok(self.encode_regions_ragged(docs, format, parallel))
        })
    }

    #[pyo3(signature = (inputs, options, *, parallel = true))]
    fn _encode_batch_list_compat<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        options: Py<bindings::bridge::WrapTruncate>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        encode_batch_pylist(py, &inputs, Some(options.get()), parallel, |docs, format| {
            Ok(self.encode_regions_ragged(docs, format, parallel))
        })
    }

    #[pyo3(signature = (inputs, options, *, parallel = true))]
    fn encode_batch_padded<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        options: padding::PadTruncate,
        parallel: bool,
    ) -> PyResult<padding::PaddedMatrix<'py>> {
        padding::encode_batch_matrix(py, &inputs, options, parallel, |docs, format| {
            Ok(self.encode_regions_ragged(docs, format, parallel))
        })
    }

    #[pyo3(signature = (source, *, parallel = true))]
    fn encode_files<'py>(
        &self,
        py: Python<'py>,
        source: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_files_ragged(py, &source, parallel, |files, format| {
            Ok(self.encode_regions_ragged(files, format, parallel))
        })
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    #[getter]
    fn vocab<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        vocab_to_pydict(py, self.tokenizer.vocab_entries())
    }

    #[getter]
    fn merges<'py>(&self, py: Python<'py>) -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        merges_to_pylist(py, self.tokenizer.merge_entries())
    }

    fn decode(&self, tokens: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let ids = extract_token_ids(&tokens)?;
        Ok(self.tokenizer.decode(ids.as_slice()?).collect())
    }

    fn cache_entries(&self) -> usize {
        self.tokenizer.cache_entries()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.tokenizer))
    }
}

#[pyclass]
struct SentencePieceTokenizer {
    tokenizer: bpe::SentencePieceBPE,
    /// Pretoken cache + scratch for single-document `encode`.
    state: bpe::sentencepiece::EncodeState,
}

impl SentencePieceTokenizer {
    /// BPETokenizer::encode_regions_ragged for SentencePiece. Regions are
    /// trusted to be valid UTF-8 (the documented input contract, never
    /// validated), which makes the unchecked str conversions sound.
    fn encode_regions_ragged(
        &self,
        regions: &[&[u8]],
        format: &DocFormat,
        parallel: bool,
    ) -> PyResult<(Vec<u32>, Vec<i64>)> {
        debug_assert!(regions.iter().all(|d| std::str::from_utf8(d).is_ok()));
        // Only a UTF-8 separator is guaranteed to cut valid UTF-8 at char
        // boundaries, which the unchecked conversions rely on.
        if let DocFormat::Text { separator: Some(sep) } = format
            && std::str::from_utf8(sep).is_err()
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "the SentencePiece backend requires a separator that is valid UTF-8",
            ));
        }
        Ok(if parallel {
            sp_encode_files_docs(&self.tokenizer, regions, format)
        } else {
            sp_encode_files_docs_serial(&self.tokenizer, regions, format)
        })
    }

    fn with_model(model: bpe::SentencePieceBPE) -> Self {
        let tokenizer = bindings::cache::apply_max_cache_bytes_sp(model);
        let state =
            bpe::sentencepiece::EncodeState::with_budget(tokenizer.max_cache_bytes());
        Self { tokenizer, state }
    }
}

#[pymethods]
impl SentencePieceTokenizer {
    #[staticmethod]
    fn from_hf(path: PathBuf) -> PyResult<Self> {
        Ok(Self::with_model(load_tokenizer::hf::load_hf_sentencepiece(&path)?))
    }

    #[pyo3(signature = (inputs, *, parallel = true))]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_batch_ragged(py, &inputs, |docs, format| {
            self.encode_regions_ragged(docs, format, parallel)
        })
    }

    #[pyo3(signature = (inputs, *, parallel = true))]
    fn encode_batch_list<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        encode_batch_pylist(py, &inputs, None, parallel, |docs, format| {
            self.encode_regions_ragged(docs, format, parallel)
        })
    }

    #[pyo3(signature = (inputs, options, *, parallel = true))]
    fn _encode_batch_list_compat<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        options: Py<bindings::bridge::WrapTruncate>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        encode_batch_pylist(py, &inputs, Some(options.get()), parallel, |docs, format| {
            self.encode_regions_ragged(docs, format, parallel)
        })
    }

    #[pyo3(signature = (inputs, options, *, parallel = true))]
    fn encode_batch_padded<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        options: padding::PadTruncate,
        parallel: bool,
    ) -> PyResult<padding::PaddedMatrix<'py>> {
        padding::encode_batch_matrix(py, &inputs, options, parallel, |docs, format| {
            self.encode_regions_ragged(docs, format, parallel)
        })
    }

    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let input = extract_doc(&input)?;
        let text: &str = match &input {
            EncodeInput::Text(s) => s,
            EncodeInput::Bytes(b) => {
                debug_assert!(std::str::from_utf8(b).is_ok());
                // SAFETY: valid UTF-8 by the documented input contract.
                unsafe { std::str::from_utf8_unchecked(b) }
            }
        };
        let mut ids: Vec<u32> = Vec::new();
        self.tokenizer.encode_raw_cb(&mut self.state, text, &mut |tokens| {
            ids.extend(tokens.iter().map(|&t| u32::from(t)))
        });
        Ok(ids.into_pyarray(py))
    }

    #[pyo3(signature = (source, *, parallel = true))]
    fn encode_files<'py>(
        &self,
        py: Python<'py>,
        source: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_files_ragged(py, &source, parallel, |files, format| {
            self.encode_regions_ragged(files, format, parallel)
        })
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    #[getter]
    fn vocab<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        vocab_to_pydict(py, self.tokenizer.vocab_entries())
    }

    #[getter]
    fn merges<'py>(&self, py: Python<'py>) -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        merges_to_pylist(py, self.tokenizer.merge_entries())
    }

    fn encode_no_normalize<'py>(
        &mut self,
        py: Python<'py>,
        input: &str,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let mut ids: Vec<u32> = Vec::new();
        self.tokenizer.encode_normalized_cb(&mut self.state, input, &mut |tokens| {
            ids.extend(tokens.iter().map(|&t| u32::from(t)))
        });
        Ok(ids.into_pyarray(py))
    }

    fn decode(&self, tokens: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let ids = extract_token_ids(&tokens)?;
        Ok(self.tokenizer.decode(ids.as_slice()?))
    }

    fn cache_entries(&self) -> usize {
        self.state.cache_size()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.tokenizer))
    }
}

/// Load in-memory tokenizer.json contents (str or bytes) as a
/// SentencePieceTokenizer (byte_fallback) or a BPETokenizer.
#[pyfunction]
fn load_hf_json(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let backed_str;
    let backed_bytes;
    let bytes: &[u8] = if let Ok(s) = data.extract::<PyBackedStr>() {
        backed_str = s;
        backed_str.as_bytes()
    } else if let Ok(b) = data.extract::<PyBackedBytes>() {
        backed_bytes = b;
        &backed_bytes
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "expected tokenizer.json contents as str or bytes, got {}",
            data.get_type()
        )));
    };
    match load_tokenizer::hf::load_hf_slice(bytes)? {
        load_tokenizer::hf::HfTokenizer::Bpe(tokenizer) => Ok(Py::new(
            py,
            BPETokenizer {
                tokenizer: bindings::cache::apply_max_cache_bytes(tokenizer),
                workers: WorkerPool::new(),
            },
        )?
        .into_any()),
        load_tokenizer::hf::HfTokenizer::SentencePiece(tokenizer) => {
            Ok(Py::new(py, SentencePieceTokenizer::with_model(tokenizer))?.into_any())
        }
    }
}

#[pymodule]
fn gigatoken_rs<'py>(py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add("SpecialTokenFound", py.get_type::<SpecialTokenFound>())?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_class::<FileSource>()?;
    m.add_class::<TextFileSource>()?;
    m.add_class::<JsonlFileSource>()?;
    m.add_class::<ParquetFileSource>()?;
    m.add_class::<BytesSource>()?;
    m.add_class::<PretokenizerIter>()?;
    m.add_class::<SubstringMatcher>()?;
    m.add_class::<bindings::bridge::WrapTruncate>()?;
    m.add_class::<padding::PadTruncate>()?;
    m.add_class::<BPETokenizer>()?;
    m.add_class::<SentencePieceTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenized_counts, m)?)?;
    m.add_function(wrap_pyfunction!(load_hf_json, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::hub::hub_file, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::hub::looks_like_repo_id, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::hub::get_hf_token, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::cache::set_max_cache_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::cache::get_max_cache_bytes, m)?)?;
    Ok(())
}
