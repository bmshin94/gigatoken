//! The FileSource/BytesSource Python classes and the helpers that turn an
//! encode_files or train_bpe argument into loaded, format-tagged file
//! contents.

use crate::input::file_source::{DocFormat, LoadedFile, detect_default_format, load_file};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use std::path::PathBuf;

/// Base class for file sources; constructed through the subclasses.
#[pyclass(subclass, from_py_object)]
#[derive(Clone)]
pub(crate) struct FileSource {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) format: DocFormat,
}

#[pymethods]
impl FileSource {
    fn __repr__(&self) -> String {
        let n = self.paths.len();
        match &self.format {
            DocFormat::Jsonl { field } => {
                format!("JsonlFileSource(paths=[{n} files], field={field:?})")
            }
            DocFormat::Text {
                separator: Some(sep),
            } => format!(
                "TextFileSource(paths=[{n} files], separator={:?})",
                String::from_utf8_lossy(sep)
            ),
            DocFormat::Text { separator: None } => {
                format!("TextFileSource(paths=[{n} files])")
            }
            DocFormat::Parquet { column } => {
                format!("ParquetFileSource(paths=[{n} files], column={column:?})")
            }
        }
    }
}

/// A `separator` argument: bytes used as-is, or str encoded to its UTF-8
/// bytes.
#[derive(FromPyObject)]
pub(crate) enum Separator {
    Bytes(Vec<u8>),
    Str(String),
}

impl Separator {
    fn into_bytes(self) -> Vec<u8> {
        match self {
            Separator::Bytes(b) => b,
            Separator::Str(s) => s.into_bytes(),
        }
    }
}

/// Plain-text files. With `separator` (str or bytes), documents are the
/// pieces between separator occurrences (the separator itself belongs to no
/// document); without one, each file is a single document.
#[pyclass(extends = FileSource)]
pub(crate) struct TextFileSource;

#[pymethods]
impl TextFileSource {
    #[new]
    #[pyo3(signature = (paths, separator = None))]
    fn new(paths: Vec<PathBuf>, separator: Option<Separator>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(FileSource {
            paths,
            format: DocFormat::Text {
                separator: separator.map(Separator::into_bytes),
            },
        })
        .add_subclass(Self)
    }
}

/// JSON Lines files: one document per line, text taken from `field`.
#[pyclass(extends = FileSource)]
pub(crate) struct JsonlFileSource;

#[pymethods]
impl JsonlFileSource {
    #[new]
    #[pyo3(signature = (paths, field = "text"))]
    fn new(paths: Vec<PathBuf>, field: &str) -> PyClassInitializer<Self> {
        PyClassInitializer::from(FileSource {
            paths,
            format: DocFormat::Jsonl {
                field: field.to_string(),
            },
        })
        .add_subclass(Self)
    }
}

/// Parquet files: one document per row, text taken from `column` (a string
/// or binary column; null rows become empty documents, so results stay
/// row-aligned with the table). Parquet handles compression internally, so
/// no .gz/.zst detection applies.
#[pyclass(extends = FileSource)]
pub(crate) struct ParquetFileSource;

#[pymethods]
impl ParquetFileSource {
    #[new]
    #[pyo3(signature = (paths, column = "text"))]
    fn new(paths: Vec<PathBuf>, column: &str) -> PyClassInitializer<Self> {
        PyClassInitializer::from(FileSource {
            paths,
            format: DocFormat::Parquet {
                column: column.to_string(),
            },
        })
        .add_subclass(Self)
    }
}

/// In-memory bytes for encode_batch, the buffer analog of TextFileSource.
/// The buffers are borrowed, not copied, and split during the encode.
#[pyclass(frozen)]
pub(crate) struct BytesSource {
    pub(crate) buffers: Vec<PyBackedBytes>,
    pub(crate) format: DocFormat,
}

#[pymethods]
impl BytesSource {
    #[new]
    #[pyo3(signature = (data, separator = None))]
    fn new(data: &Bound<'_, PyAny>, separator: Option<Separator>) -> PyResult<Self> {
        let buffers = if let Ok(buffer) = data.extract::<PyBackedBytes>() {
            vec![buffer]
        } else {
            data.extract::<Vec<PyBackedBytes>>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "expected bytes or a list of bytes, got {}",
                    data.get_type()
                ))
            })?
        };
        Ok(Self {
            buffers,
            format: DocFormat::Text {
                separator: separator.map(Separator::into_bytes),
            },
        })
    }

    fn __repr__(&self) -> String {
        let n = self.buffers.len();
        let total: usize = self.buffers.iter().map(|b| b.len()).sum();
        match &self.format {
            DocFormat::Text {
                separator: Some(sep),
            } => format!(
                "BytesSource(data=[{n} buffers, {total} bytes], separator={:?})",
                String::from_utf8_lossy(sep)
            ),
            _ => format!("BytesSource(data=[{n} buffers, {total} bytes])"),
        }
    }
}

/// Resolve an encode_files argument: a FileSource, a path, or a list of
/// paths (format detected from the first path's extension).
pub(crate) fn resolve_files_source(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<PathBuf>, DocFormat)> {
    if let Ok(fs) = obj.extract::<FileSource>() {
        return Ok((fs.paths, fs.format));
    }
    if let Ok(path) = obj.extract::<PathBuf>() {
        let format = detect_default_format(&path);
        return Ok((vec![path], format));
    }
    if let Ok(paths) = obj.extract::<Vec<PathBuf>>() {
        let format = paths
            .first()
            .map(|p| detect_default_format(p))
            .unwrap_or(DocFormat::Text { separator: None });
        return Ok((paths, format));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "expected a TextFileSource/JsonlFileSource/ParquetFileSource, a path, or a list of paths, got {}",
        obj.get_type()
    )))
}

/// Shared scaffold of the encode_files pymethods: resolve the source, load
/// the files with the GIL released, run `encode` (still detached), and
/// return the ragged result as an awkward Array.
pub(crate) fn encode_files_ragged<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    parallel: bool,
    encode: impl FnOnce(&[&[u8]], &DocFormat) -> PyResult<(Vec<u32>, Vec<i64>)> + Send,
) -> PyResult<Bound<'py, PyAny>> {
    let (paths, format) = resolve_files_source(source)?;
    let (flat, counts) = py.detach(|| -> PyResult<_> {
        // Parquet rows are materialized as owned documents and encoded
        // through the whole-document path.
        if let DocFormat::Parquet { column } = &format {
            let docs = load_parquet_docs(&paths, column, parallel)?;
            let bytes: Vec<&[u8]> = docs.iter().map(|d| d.as_slice()).collect();
            return encode(&bytes, &DocFormat::Text { separator: None });
        }
        let files = load_files(&paths, parallel)?;
        let bytes: Vec<&[u8]> = files.iter().map(|f| f.as_bytes()).collect();
        encode(&bytes, &format)
    })?;
    super::bridge::ragged_to_python(py, flat, counts)
}

/// `column` of every parquet file as one owned document per row, files in
/// argument order. Serial on the calling thread when `parallel` is false.
fn load_parquet_docs(
    paths: &[PathBuf],
    column: &str,
    parallel: bool,
) -> PyResult<Vec<Vec<u8>>> {
    use crate::input::parquet::read_docs;
    use rayon::prelude::*;
    // input::parquet errors already carry the offending path.
    let to_pyerr = |e: std::io::Error| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string());
    let per_file: Vec<Vec<Vec<u8>>> = if parallel {
        paths
            .par_iter()
            .map(|p| read_docs(p, column, true).map_err(to_pyerr))
            .collect::<PyResult<_>>()?
    } else {
        paths
            .iter()
            .map(|p| read_docs(p, column, false).map_err(to_pyerr))
            .collect::<PyResult<_>>()?
    };
    Ok(per_file.into_iter().flatten().collect())
}

/// Load all files (see `load_file`), serially when `parallel` is false.
pub(crate) fn load_files(paths: &[PathBuf], parallel: bool) -> PyResult<Vec<LoadedFile>> {
    use rayon::prelude::*;
    let load = |p: &PathBuf| {
        load_file(p).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}: {e}", p.display()))
        })
    };
    if parallel {
        paths.par_iter().map(load).collect()
    } else {
        paths.iter().map(load).collect()
    }
}
