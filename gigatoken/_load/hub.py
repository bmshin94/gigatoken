"""HuggingFace Hub file fetch: thin forwards to `src/load_tokenizer/hub.rs`,
which mirrors `huggingface_hub.hf_hub_download` (same URL layout, token
discovery and cache layout) without requiring huggingface_hub."""

from __future__ import annotations

from gigatoken.gigatoken_rs import get_hf_token, hub_file, looks_like_repo_id

# A name ending in one of these is a local tokenizer file, never a Hub repo
# id. Keep in sync with TOKENIZER_FILE_SUFFIXES in `src/load_tokenizer/hub.rs`.
TOKENIZER_FILE_SUFFIXES = (".json", ".model")

__all__ = [
    "TOKENIZER_FILE_SUFFIXES",
    "download_hub_file",
    "get_hf_token",
    "hub_file",
    "looks_like_repo_id",
]


def download_hub_file(repo_id: str, filename: str = "tokenizer.json", *, revision: str = "main") -> bytes:
    """Contents of `filename` from Hub repo `repo_id` at `revision`, served
    from the standard HF cache, downloading into it first when absent."""
    return hub_file(repo_id, filename, revision=revision).read_bytes()
