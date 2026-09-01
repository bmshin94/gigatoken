//! Load HuggingFace tokenizer.json files.
//!
//! Supports two styles:
//! - SentencePiece BPE with `byte_fallback=true` (e.g. Llama) → [`load_hf_sentencepiece`]
//! - ByteLevel BPE without byte_fallback (e.g. GPT-2) → [`load_hf_bpe`]

// The tokenizer variants differ greatly in size
#![allow(clippy::large_enum_variant)]

use crate::bpe::sentencepiece::{AddedTokenSpec, Metaspace, NormOp, PrependScheme};
use crate::bpe::{self, SentencePieceBPE};
use crate::token::TokenId;
use eyre::{Context, Result, ensure};
use rustc_hash::FxBuildHasher;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

#[derive(Deserialize)]
struct TokenizerJson {
    model: Model,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<PreTokenizerJson>,
    #[serde(default)]
    normalizer: Option<NormalizerJson>,
}

#[derive(Deserialize)]
struct NormalizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    normalizers: Vec<NormalizerJson>,
    /// `Prepend` normalizer: the prefix (Llama 2's "▁").
    #[serde(default)]
    prepend: Option<String>,
    /// `Replace` normalizer: pattern and replacement content.
    #[serde(default)]
    pattern: Option<PatternJson>,
    #[serde(default)]
    content: Option<String>,
    /// `Strip` normalizer sides.
    #[serde(default)]
    strip_left: Option<bool>,
    #[serde(default)]
    strip_right: Option<bool>,
    /// `Precompiled` normalizer: base64-encoded sentencepiece charsmap.
    #[serde(default)]
    precompiled_charsmap: Option<String>,
}

#[derive(Deserialize)]
struct PreTokenizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    pretokenizers: Vec<PreTokenizerJson>,
    #[serde(default)]
    pattern: Option<PatternJson>,
    /// `Metaspace` fields. `add_prefix_space` is the pre-0.15 spelling of
    /// `prepend_scheme`.
    #[serde(default)]
    replacement: Option<String>,
    #[serde(default)]
    prepend_scheme: Option<String>,
    #[serde(default)]
    add_prefix_space: Option<bool>,
    #[serde(default)]
    split: Option<bool>,
    /// `Split` field (e.g. "MergedWithPrevious" for the gemma-3/4 no-op
    /// space Split).
    #[serde(default)]
    behavior: Option<String>,
}

#[derive(Deserialize)]
struct PatternJson {
    #[serde(rename = "Regex", default)]
    regex: Option<String>,
    #[serde(rename = "String", default)]
    literal: Option<String>,
}

#[derive(Deserialize)]
struct Model {
    /// tokenizer.json files written before tokenizers 0.9 (e.g. the original
    /// GPT-2 upload) omit `model.type`; those are always BPE.
    #[serde(rename = "type", default = "legacy_bpe_type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    #[serde(deserialize_with = "deserialize_merges")]
    merges: Vec<[String; 2]>,
    #[serde(default)]
    byte_fallback: bool,
    /// HF BPE `ignore_merges`: a pretoken whose whole byte string is a vocab
    /// entry encodes as that single ID, skipping the merge loop (GLM-5.2,
    /// DeepSeek V3, Llama 3).
    #[serde(default)]
    ignore_merges: bool,
}

fn legacy_bpe_type() -> String {
    "BPE".to_string()
}

/// Merges appear as `["a", "b"]` arrays in current tokenizer.json files and
/// as `"a b"` strings in older ones; accept both.
fn deserialize_merges<'de, D>(deserializer: D) -> Result<Vec<[String; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Merge {
        Pair([String; 2]),
        Legacy(String),
    }
    let raw = Vec::<Merge>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|m| match m {
            Merge::Pair(pair) => Ok(pair),
            Merge::Legacy(s) => {
                let (a, b) = s.split_once(' ').ok_or_else(|| {
                    serde::de::Error::custom(format!("invalid merge entry: {s:?}"))
                })?;
                Ok([a.to_string(), b.to_string()])
            }
        })
        .collect()
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
}

/// Parse a byte-fallback token string `<0xHH>` into its byte.
fn parse_byte_fallback(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

/// A SentencePiece vocab string as bytes: `<0xHH>` is the byte, anything
/// else its UTF-8 (▁ kept as-is).
fn token_str_to_bytes(s: &str) -> Vec<u8> {
    match parse_byte_fallback(s) {
        Some(byte) => vec![byte],
        None => s.as_bytes().to_vec(),
    }
}

type VocabInv = HashMap<Arc<[u8]>, TokenId, FxBuildHasher>;

/// Vocab by ID plus its inverse from `(bytes, id, weak)` entries; a weak
/// entry (SentencePiece `<0xHH>` byte tokens) never displaces another
/// token with the same bytes in the inverse. Added tokens may live outside
/// model.vocab (Qwen2's <|endoftext|>); the vocab is extended so their IDs
/// decode to the literal content.
fn build_vocab(
    entries: impl Iterator<Item = (Vec<u8>, u32, bool)>,
    added_tokens: &[AddedToken],
) -> (Vec<Arc<[u8]>>, VocabInv) {
    let entries: Vec<(Arc<[u8]>, u32, bool)> = entries.map(|(b, id, weak)| (b.into(), id, weak)).collect();
    let max_id = entries.iter().map(|e| e.1).max().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    let mut vocab_inv = VocabInv::with_capacity_and_hasher(entries.len(), FxBuildHasher);
    for (bytes, id, weak) in entries {
        vocab[id as usize] = bytes.clone();
        if weak {
            vocab_inv.entry(bytes).or_insert(TokenId::from(id));
        } else {
            vocab_inv.insert(bytes, TokenId::from(id));
        }
    }
    for t in added_tokens {
        let id = t.id as usize;
        if id >= vocab.len() {
            vocab.resize(id + 1, Arc::from(Vec::new().as_slice()));
        }
        if vocab[id].is_empty() {
            vocab[id] = t.content.as_bytes().into();
        }
    }
    (vocab, vocab_inv)
}

/// A tokenizer loaded from HuggingFace `tokenizer.json` data: the model's
/// `byte_fallback` flag decides which of the two supported styles applies.
pub enum HfTokenizer {
    Bpe(bpe::tiktoken::Tokenizer),
    SentencePiece(SentencePieceBPE),
}

/// Probes `model.type` alone so an unsupported family (WordPiece, Unigram)
/// is refused by name instead of with a misleading shape error from the
/// full BPE schema.
#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model: Option<ModelTypeOnly>,
}

#[derive(Deserialize)]
struct ModelTypeOnly {
    #[serde(rename = "type")]
    model_type: Option<String>,
    /// Family markers for untyped legacy files: `unk_id` only exists on
    /// Unigram, `max_input_chars_per_word` only on WordPiece.
    unk_id: Option<u64>,
    max_input_chars_per_word: Option<u64>,
}

fn parse_tokenizer_json(data: &[u8]) -> Result<TokenizerJson> {
    if let Ok(ModelTypeProbe { model: Some(m) }) = sonic_rs::from_slice::<ModelTypeProbe>(data) {
        let family = match m.model_type.as_deref() {
            Some("BPE") => None,
            Some(other) => Some(other.to_string()),
            None if m.unk_id.is_some() => Some("Unigram (untyped legacy file)".to_string()),
            None if m.max_input_chars_per_word.is_some() => {
                Some("WordPiece (untyped legacy file)".to_string())
            }
            None => None, // untyped BPE (pre-0.9 GPT-2-style files)
        };
        if let Some(family) = family {
            return Err(eyre::eyre!(
                "Unsupported model type \"{family}\": gigatoken supports BPE tokenizers \
                 (byte-level, or SentencePiece-style with byte_fallback)"
            ));
        }
    }
    sonic_rs::from_slice(data).map_err(|e| eyre::eyre!("Failed to parse tokenizer JSON: {e}"))
}

fn read_tokenizer_json(path: impl AsRef<Path>) -> Result<TokenizerJson> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    parse_tokenizer_json(&data).with_context(|| format!("Failed to parse {}", path.display()))
}

/// Load a tokenizer from in-memory `tokenizer.json` contents, choosing the
/// SentencePiece or ByteLevel BPE style from the model's `byte_fallback` flag.
pub fn load_hf_slice(data: &[u8]) -> Result<HfTokenizer> {
    let tj = parse_tokenizer_json(data)?;
    if tj.model.byte_fallback {
        Ok(HfTokenizer::SentencePiece(build_sentencepiece(&tj)?))
    } else {
        Ok(HfTokenizer::Bpe(build_bpe(&tj)?))
    }
}

/// Load a SentencePiece-style BPE tokenizer.json (byte_fallback, e.g. Llama).
pub fn load_hf_sentencepiece(path: impl AsRef<Path>) -> Result<SentencePieceBPE> {
    build_sentencepiece(&read_tokenizer_json(path)?)
}

fn build_sentencepiece(tj: &TokenizerJson) -> Result<SentencePieceBPE> {
    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        tj.model.byte_fallback,
        "Only byte_fallback tokenizers are supported"
    );

    let hf_vocab = &tj.model.vocab;
    let (vocab, vocab_inv) = build_vocab(
        hf_vocab.iter().map(|(s, &id)| (token_str_to_bytes(s), id, parse_byte_fallback(s).is_some())),
        &tj.added_tokens,
    );

    // Some vocabs omit byte tokens they never need (Gemma has literal `\t`
    // pieces instead of `<0x09>`); those stay `None`.
    let mut byte_fallback_ids = [None; 256];
    for byte_val in 0u16..=255 {
        let key = format!("<0x{:02X}>", byte_val);
        byte_fallback_ids[byte_val as usize] = hf_vocab.get(&key).map(|&id| TokenId::from(id));
    }
    ensure!(
        byte_fallback_ids.iter().any(|id| id.is_some()),
        "byte_fallback is set but the vocab has no <0xHH> byte tokens"
    );

    // Merges keep their explicit list rank. The merged piece is looked up
    // as the concatenated *string* (HF semantics), not concatenated bytes.
    let mut merges: HashMap<u64, (TokenId, u32), FxBuildHasher> =
        HashMap::with_capacity_and_hasher(tj.model.merges.len(), FxBuildHasher);
    let hf_str_to_id = |s: &str| vocab_inv.get(token_str_to_bytes(s).as_slice()).copied();
    for (rank, [str_a, str_b]) in tj.model.merges.iter().enumerate() {
        let (Some(id_a), Some(id_b), Some(id_merged)) = (
            hf_str_to_id(str_a),
            hf_str_to_id(str_b),
            hf_str_to_id(&format!("{str_a}{str_b}")),
        ) else {
            continue;
        };
        merges
            .entry(crate::bpe::ranked_merge_key(id_a, id_b))
            .or_insert((id_merged, rank as u32));
    }

    let mut norm_ops = Vec::new();
    if let Some(n) = &tj.normalizer {
        parse_sp_normalizer(n, &mut norm_ops)?;
    }
    let metaspace = parse_sp_metaspace(&tj.pre_tokenizer, &norm_ops)?;

    // All added tokens are matched atomically, like HF's AddedVocabulary;
    // `normalized: true` ones match against normalizer output.
    let mut added_tokens = Vec::new();
    let mut norm_added_tokens = Vec::new();
    for t in &tj.added_tokens {
        let spec = AddedTokenSpec {
            content: t.content.clone(),
            id: TokenId::from(t.id),
            lstrip: t.lstrip,
            rstrip: t.rstrip,
        };
        if t.normalized {
            norm_added_tokens.push(spec);
        } else {
            added_tokens.push(spec);
        }
    }

    Ok(SentencePieceBPE::new(
        merges,
        vocab,
        vocab_inv,
        byte_fallback_ids,
        added_tokens,
        norm_added_tokens,
        norm_ops,
        metaspace,
    ))
}

/// Translate a tokenizer.json `normalizer` into [`NormOp`]s; anything
/// unsupported is an error, since skipping it would diverge from HF.
fn parse_sp_normalizer(n: &NormalizerJson, out: &mut Vec<NormOp>) -> Result<()> {
    match n.kind.as_str() {
        "Sequence" => {
            for child in &n.normalizers {
                parse_sp_normalizer(child, out)?;
            }
        }
        "Prepend" => {
            let prefix = n
                .prepend
                .clone()
                .ok_or_else(|| eyre::eyre!("Prepend normalizer without a `prepend` string"))?;
            out.push(NormOp::Prepend(prefix));
        }
        "Replace" => {
            let content = n
                .content
                .clone()
                .ok_or_else(|| eyre::eyre!("Replace normalizer without a `content` string"))?;
            match &n.pattern {
                Some(PatternJson {
                    literal: Some(pattern),
                    ..
                }) => out.push(NormOp::Replace {
                    pattern: pattern.clone(),
                    content,
                }),
                // SpmConverter's spelling of `remove_extra_whitespaces`.
                Some(PatternJson {
                    regex: Some(re), ..
                }) if re == " {2,}" => out.push(NormOp::CollapseSpaces { content }),
                Some(PatternJson {
                    regex: Some(re), ..
                }) => {
                    return Err(eyre::eyre!(
                        "Unsupported Replace normalizer regex: {re:?} (only \" {{2,}}\" is supported)"
                    ));
                }
                _ => return Err(eyre::eyre!("Replace normalizer without a pattern")),
            }
        }
        "Strip" => out.push(NormOp::Strip {
            left: n.strip_left.unwrap_or(true),
            right: n.strip_right.unwrap_or(true),
        }),
        "Precompiled" => {
            use base64::Engine;
            let b64 = n.precompiled_charsmap.as_deref().ok_or_else(|| {
                eyre::eyre!("Precompiled normalizer without a `precompiled_charsmap`")
            })?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .context("Failed to base64-decode precompiled_charsmap")?;
            let precompiled = spm_precompiled::Precompiled::from(&bytes)
                .map_err(|e| eyre::eyre!("Failed to parse precompiled_charsmap: {e}"))?;
            out.push(NormOp::Precompiled(
                crate::bpe::sentencepiece::PrecompiledCharsmap::new(precompiled),
            ));
        }
        other => {
            return Err(eyre::eyre!(
                "Unsupported normalizer type for SentencePiece tokenizers: {other}"
            ));
        }
    }
    Ok(())
}

/// Translate a tokenizer.json `pre_tokenizer` into a [`Metaspace`] config;
/// `None` (e.g. Llama 2) lets merges cross word boundaries. `norm_ops`
/// proves that a `Split` on a literal space is a no-op (gemma-3/4).
fn parse_sp_metaspace(
    pre_tokenizer: &Option<PreTokenizerJson>,
    norm_ops: &[NormOp],
) -> Result<Option<Metaspace>> {
    fn from_metaspace(pt: &PreTokenizerJson) -> Result<Metaspace> {
        ensure!(
            pt.replacement.as_deref().unwrap_or("\u{2581}") == "\u{2581}",
            "Unsupported Metaspace replacement: {:?} (expected \"▁\")",
            pt.replacement
        );
        let prepend = match (&pt.prepend_scheme, pt.add_prefix_space) {
            (Some(scheme), _) => match scheme.as_str() {
                "never" => PrependScheme::Never,
                "always" => PrependScheme::Always,
                "first" => PrependScheme::First,
                other => {
                    return Err(eyre::eyre!("Unsupported Metaspace prepend_scheme: {other}"));
                }
            },
            (None, Some(false)) => PrependScheme::Never,
            (None, _) => PrependScheme::Always,
        };
        Ok(Metaspace {
            prepend,
            split: pt.split.unwrap_or(true),
        })
    }

    let Some(pt) = pre_tokenizer else {
        return Ok(None);
    };
    match pt.kind.as_str() {
        "Metaspace" => Ok(Some(from_metaspace(pt)?)),
        "Sequence"
            if pt.pretokenizers.len() == 1 && pt.pretokenizers[0].kind == "Metaspace" =>
        {
            Ok(Some(from_metaspace(&pt.pretokenizers[0])?))
        }
        // gemma-3/4: a Split on " " after the normalizer replaced every
        // space is a no-op.
        "Split"
            if matches!(
                &pt.pattern,
                Some(PatternJson { literal: Some(l), .. }) if l == " "
            ) && pt.behavior.as_deref() == Some("MergedWithPrevious")
                && norm_ops.iter().any(|op| matches!(
                    op,
                    NormOp::Replace { pattern, content }
                        if pattern == " " && !content.contains(' ')
                )) =>
        {
            Ok(None)
        }
        other => Err(eyre::eyre!(
            "Unsupported pre_tokenizer type for SentencePiece tokenizers: {other}"
        )),
    }
}

/// Whether the normalizer is NFC (the only kind supported for ByteLevel
/// BPE); anything else is an error.
fn detect_nfc_normalizer(normalizer: &Option<NormalizerJson>) -> Result<bool> {
    fn is_nfc(n: &NormalizerJson) -> Result<bool> {
        match n.kind.as_str() {
            "NFC" => Ok(true),
            "Sequence" => n
                .normalizers
                .iter()
                .try_fold(false, |acc, c| Ok(acc | is_nfc(c)?)),
            other => Err(eyre::eyre!("Unsupported normalizer type: {other}")),
        }
    }
    normalizer.as_ref().map_or(Ok(false), is_nfc)
}

/// The pretokenization scheme of a `pre_tokenizer`: a bare `ByteLevel` is
/// GPT-2, otherwise the `Split` regexes (in order) must form a known scheme.
fn detect_pretokenizer_type(
    pre_tokenizer: &Option<PreTokenizerJson>,
) -> Result<crate::pretokenize::PretokenizerType> {
    use crate::pretokenize::PretokenizerType;

    fn collect_split_regexes<'a>(pt: &'a PreTokenizerJson, out: &mut Vec<&'a str>) {
        if pt.kind == "Split"
            && let Some(PatternJson { regex: Some(re), .. }) = &pt.pattern
        {
            out.push(re);
        }
        for child in &pt.pretokenizers {
            collect_split_regexes(child, out);
        }
    }

    let Some(pt) = pre_tokenizer else {
        return Ok(PretokenizerType::GPT2);
    };
    let mut regexes = Vec::new();
    collect_split_regexes(pt, &mut regexes);
    if regexes.is_empty() {
        if pt.kind == "ByteLevel" {
            return Ok(PretokenizerType::GPT2);
        }
        return Err(eyre::eyre!(
            "Unsupported pre_tokenizer type: {} (no Split regex found)",
            pt.kind
        ));
    }
    PretokenizerType::from_split_regexes(&regexes).ok_or_else(|| {
        eyre::eyre!("Unknown pre_tokenizer Split regexes, no fast pretokenizer for: {regexes:?}")
    })
}

/// Whether a `ByteLevel` pre-tokenizer in the chain sets `add_prefix_space`.
fn detect_add_prefix_space(pre_tokenizer: &Option<PreTokenizerJson>) -> bool {
    fn walk(pt: &PreTokenizerJson) -> bool {
        (pt.kind == "ByteLevel" && pt.add_prefix_space == Some(true))
            || pt.pretokenizers.iter().any(walk)
    }
    pre_tokenizer.as_ref().is_some_and(walk)
}

/// The GPT-2 byte-to-unicode table and its inverse.
fn build_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let allowed: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut b2u = ['\0'; 256];
    for &b in &allowed {
        b2u[b as usize] = b as char;
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if b2u[b as usize] == '\0' {
            b2u[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let u2b: HashMap<char, u8> = b2u.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
    (b2u, u2b)
}

/// A ByteLevel vocab string as raw bytes; a string with any char outside
/// the table (DeepSeek V4 keeps specials unencoded) is taken as literal UTF-8.
fn unicode_to_bytes(s: &str, u2b: &HashMap<char, u8>) -> Vec<u8> {
    if s.chars().all(|c| u2b.contains_key(&c)) {
        s.chars().map(|c| u2b[&c]).collect()
    } else {
        s.as_bytes().to_vec()
    }
}

/// Load a ByteLevel BPE tokenizer.json (no byte_fallback, e.g. GPT-2).
pub fn load_hf_bpe(path: impl AsRef<Path>) -> Result<bpe::tiktoken::Tokenizer> {
    build_bpe(&read_tokenizer_json(path)?)
}

fn build_bpe(tj: &TokenizerJson) -> Result<bpe::tiktoken::Tokenizer> {
    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        !tj.model.byte_fallback,
        "byte_fallback tokenizers should use load_hf_sentencepiece instead"
    );

    let (_b2u, u2b) = build_byte_unicode_tables();
    let (vocab, vocab_inv) = build_vocab(
        tj.model.vocab.iter().map(|(s, &id)| (unicode_to_bytes(s, &u2b), id, false)),
        &tj.added_tokens,
    );

    // (a, b, merged) ids per merge; merges naming unknown pieces are skipped.
    let mut entries: Vec<(TokenId, TokenId, TokenId)> = Vec::with_capacity(tj.model.merges.len());
    for [str_a, str_b] in &tj.model.merges {
        let bytes_a = unicode_to_bytes(str_a, &u2b);
        let bytes_b = unicode_to_bytes(str_b, &u2b);
        let merged_bytes = [bytes_a.as_slice(), bytes_b.as_slice()].concat();
        let (Some(&id_a), Some(&id_b), Some(&id_merged)) = (
            vocab_inv.get(bytes_a.as_slice()),
            vocab_inv.get(bytes_b.as_slice()),
            vocab_inv.get(merged_bytes.as_slice()),
        ) else {
            continue;
        };
        entries.push((id_a, id_b, id_merged));
    }

    let byte_remapping = bpe::ByteRemapping::from_byte_vocab(&vocab)?;
    let vocab: Vec<Vec<u8>> = vocab.into_iter().map(|a| a.to_vec()).collect();

    // The fast merge loops use the merged ID as the priority, which is right
    // when the merge list produces IDs in rank order (every tiktoken-style
    // vocab). Fairseq-heritage vocabs (RoBERTa/OPT) carry explicit ranks.
    let id_order_ok = entries.is_sorted_by_key(|&(_, _, merged)| merged);
    let mut tokenizer = if id_order_ok {
        let mut merges: HashMap<(TokenId, TokenId), TokenId, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for (id_a, id_b, id_merged) in entries {
            merges.entry((id_a, id_b)).or_insert(id_merged);
        }
        bpe::tiktoken::Tokenizer::new(merges, vocab, byte_remapping)
    } else {
        let mut merges: bpe::RankedMerges =
            HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for (rank, (id_a, id_b, id_merged)) in entries.into_iter().enumerate() {
            merges
                .entry(bpe::ranked_merge_key(id_a, id_b))
                .or_insert((id_merged, rank as u32));
        }
        bpe::tiktoken::Tokenizer::new_ranked(merges, vocab, byte_remapping)
    };
    tokenizer.set_pretokenizer_type(detect_pretokenizer_type(&tj.pre_tokenizer)?);
    tokenizer.set_normalize_nfc(detect_nfc_normalizer(&tj.normalizer)?);
    tokenizer.set_add_prefix_space(detect_add_prefix_space(&tj.pre_tokenizer));
    tokenizer.set_ignore_merges(tj.model.ignore_merges);
    tokenizer.set_added_tokens(
        tj.added_tokens
            .iter()
            .map(|t| bpe::tiktoken::AddedTokenDef {
                content: t.content.as_bytes().into(),
                id: TokenId::from(t.id),
                lstrip: t.lstrip,
                rstrip: t.rstrip,
            })
            .collect(),
    );
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_str_to_bytes() {
        assert_eq!(token_str_to_bytes("<0x00>"), vec![0x00]);
        assert_eq!(token_str_to_bytes("<0xFF>"), vec![0xFF]);
        assert_eq!(token_str_to_bytes("<0x0A>"), vec![0x0A]);
        assert_eq!(token_str_to_bytes("hello"), b"hello".to_vec());
        assert_eq!(token_str_to_bytes("▁the"), "▁the".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("▁"), "▁".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("<unk>"), b"<unk>".to_vec());
        assert_eq!(token_str_to_bytes("<s>"), b"<s>".to_vec());
    }

    #[test]
    fn test_parse_legacy_model_without_type() {
        // Pre-tokenizers-0.9 files have no `model.type`; they must parse as BPE.
        let json = br#"{"model": {"vocab": {"a": 0}, "merges": []}}"#;
        let tj = parse_tokenizer_json(json).unwrap();
        assert_eq!(tj.model.model_type, "BPE");
    }

    #[test]
    fn test_unsupported_model_type_named_in_error() {
        let wordpiece = br#"{"model": {"type": "WordPiece", "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "hello": 1}}}"#;
        let unigram = br#"{"model": {"type": "Unigram", "unk_id": 0,
            "vocab": [["<unk>", 0.0], ["hello", -3.1]]}}"#;
        let untyped_unigram = br#"{"model": {"unk_id": 0, "vocab": [["<unk>", 0.0]]}}"#;
        let untyped_wordpiece: &[u8] = b"{\"model\": {\"unk_token\": \"[UNK]\",
            \"continuing_subword_prefix\": \"##\", \"max_input_chars_per_word\": 100,
            \"vocab\": {\"[UNK]\": 0}}}";
        for (json, name) in [
            (&wordpiece[..], "WordPiece"),
            (&unigram[..], "Unigram"),
            (&untyped_unigram[..], "Unigram (untyped legacy file)"),
            (&untyped_wordpiece[..], "WordPiece (untyped legacy file)"),
        ] {
            let err = match parse_tokenizer_json(json) {
                Ok(_) => panic!("expected {name} to be refused"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains(&format!("Unsupported model type \"{name}\"")),
                "unhelpful {name} error: {err}"
            );
            assert!(!err.contains("missing field"), "shape error leaked: {err}");
        }
    }

    #[test]
    fn test_parse_error_names_the_field() {
        let json = br#"{"model": {"type": "BPE", "merges": []}}"#;
        let err = match parse_tokenizer_json(json) {
            Ok(_) => panic!("expected a parse error"),
            Err(e) => e,
        };
        let first_line = err.to_string();
        assert!(first_line.contains("vocab"), "unhelpful error: {first_line}");
    }

    fn tinyllama_path() -> Option<std::path::PathBuf> {
        let path = crate::test_hub::hf_tokenizer_json("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
        if path.is_none() {
            eprintln!("Skipping: TinyLlama tokenizer.json not in the HF cache");
        }
        path
    }

    #[test]
    fn test_encode_hello_sentencepiece() {
        let Some(path) = tinyllama_path() else { return };
        let tokenizer = load_hf_sentencepiece(path).unwrap();
        let ids = tokenizer.encode_raw("Hello world");
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, b"Hello world");
    }

    /// A minimal ByteLevel tokenizer.json: the 256 byte tokens (ID == byte),
    /// plus `extra_vocab` and `merges` given as raw text.
    fn byte_level_json(
        extra_vocab: &[(&str, u32)],
        merges: &[(&str, &str)],
        added_tokens_json: &str,
    ) -> Vec<u8> {
        byte_level_json_with_pretok(extra_vocab, merges, added_tokens_json, r#"{"type": "ByteLevel"}"#)
    }

    fn byte_level_json_with_pretok(
        extra_vocab: &[(&str, u32)],
        merges: &[(&str, &str)],
        added_tokens_json: &str,
        pre_tokenizer_json: &str,
    ) -> Vec<u8> {
        let (b2u, _) = build_byte_unicode_tables();
        let esc = |s: String| -> String { s.replace('\\', "\\\\").replace('"', "\\\"") };
        let enc = |s: &str| -> String { esc(s.bytes().map(|b| b2u[b as usize]).collect()) };
        let mut vocab_entries: Vec<String> = (0u16..=255)
            .map(|b| format!("\"{}\": {}", esc(b2u[b as usize].to_string()), b))
            .collect();
        for (text, id) in extra_vocab {
            vocab_entries.push(format!("\"{}\": {}", enc(text), id));
        }
        let merges_entries: Vec<String> = merges
            .iter()
            .map(|(a, b)| format!("[\"{}\", \"{}\"]", enc(a), enc(b)))
            .collect();
        format!(
            "{{\"added_tokens\": [{}], \"pre_tokenizer\": {}, \
             \"model\": {{\"type\": \"BPE\", \"vocab\": {{{}}}, \"merges\": [{}]}}}}",
            added_tokens_json,
            pre_tokenizer_json,
            vocab_entries.join(", "),
            merges_entries.join(", ")
        )
        .into_bytes()
    }

    fn encode_bpe(tok: &mut bpe::tiktoken::Tokenizer, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        tok.encode_with_added_tokens_flat(text.as_bytes(), &mut out);
        out
    }

    /// Merge priority must follow the merge list's order even when the
    /// merged token IDs do not (fairseq-heritage vocabs).
    #[test]
    fn test_rank_mapped_merges_follow_list_order() {
        // Rank 0 produces ID 350, rank 1 produces ID 300.
        let json = byte_level_json(
            &[("bc", 350), ("ab", 300)],
            &[("b", "c"), ("a", "b")],
            "",
        );
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        assert_eq!(encode_bpe(&mut tok, "abc"), vec![97, 350]);
        assert_eq!(encode_bpe(&mut tok, "ab"), vec![300]);
        assert_eq!(encode_bpe(&mut tok, "bc"), vec![350]);

        // Same merges with IDs in rank order take the id-as-rank path.
        let json = byte_level_json(
            &[("bc", 300), ("ab", 350)],
            &[("b", "c"), ("a", "b")],
            "",
        );
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        assert_eq!(encode_bpe(&mut tok, "abc"), vec![97, 300]);
    }

    #[test]
    fn test_added_token_lstrip_rstrip() {
        let added = r#"{"id": 400, "content": "<m>", "lstrip": true, "special": true},
                     {"id": 401, "content": "<r>", "rstrip": true, "special": true}"#;
        let json = byte_level_json(&[], &[], added);
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        assert_eq!(encode_bpe(&mut tok, "a <m> b"), vec![97, 400, 32, 98]);
        assert_eq!(encode_bpe(&mut tok, "a \t\n<m>"), vec![97, 400]);
        assert_eq!(
            encode_bpe(&mut tok, "a\u{a0}<m>"),
            vec![97, 400],
            "U+00A0 is `\\s` whitespace and must be absorbed"
        );
        assert_eq!(encode_bpe(&mut tok, "a <r> b"), vec![97, 32, 401, 98]);
        assert_eq!(encode_bpe(&mut tok, "<r>\n\n\nb"), vec![401, 98]);
        assert_eq!(encode_bpe(&mut tok, "<r> <m>"), vec![401, 400]);
        assert_eq!(encode_bpe(&mut tok, "a b"), vec![97, 32, 98]);
    }

    /// `ByteLevel(add_prefix_space=true)`: every non-empty added-token-split
    /// segment gets a leading space; empty segments do not.
    #[test]
    fn test_byte_level_add_prefix_space() {
        let added = r#"{"id": 400, "content": "<m>", "lstrip": true, "special": true}"#;
        let json = byte_level_json_with_pretok(
            &[],
            &[],
            added,
            r#"{"type": "ByteLevel", "add_prefix_space": true}"#,
        );
        let HfTokenizer::Bpe(mut tok) = load_hf_slice(&json).unwrap() else {
            panic!("expected ByteLevel BPE");
        };
        assert_eq!(encode_bpe(&mut tok, "ab"), vec![32, 97, 98]);
        assert_eq!(encode_bpe(&mut tok, " ab"), vec![32, 97, 98]);
        assert_eq!(encode_bpe(&mut tok, "a<m>b"), vec![32, 97, 400, 32, 98]);
        assert_eq!(encode_bpe(&mut tok, "<m><m>"), vec![400, 400]);
        assert_eq!(encode_bpe(&mut tok, "x <m> y"), vec![32, 120, 400, 32, 121]);
    }

    #[test]
    fn test_load_gpt2_from_hf() {
        let path = crate::test_hub::gpt2_tokenizer_json();
        let mut tokenizer = load_hf_bpe(&path).unwrap();
        let text = b"Hello, world! This is a test.";
        let pretokens = crate::pretokenize::pretokenize_as_iter(text);
        let mut token_ids: Vec<TokenId> = Vec::new();
        tokenizer.memoized_encode(pretokens, |tokens| {
            token_ids.extend_from_slice(tokens);
        });
        let decoded: Vec<u8> = tokenizer.decode(&token_ids).collect();
        assert_eq!(decoded, text);
    }
}
