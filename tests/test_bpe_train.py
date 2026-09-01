from pathlib import Path

import pytest

from gigatoken import train_bpe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CORPUS = (
    b"low lower lowest newer newest wider widest " * 50
    + b"the cat sat on the mat. the dog ate the cat's hat!\n" * 30
)


@pytest.mark.parametrize("tie_breaking", ["huggingface", "raw_token_ids", "assembled_bytes"])
def test_tie_breaking_modes(tie_breaking):
    """Every tie-breaking mode yields a dense vocab whose merged tokens are the
    concatenation of their merge pair, with special tokens right after the bytes."""
    vocab, merges = train_bpe(CORPUS, 300, ["<|endoftext|>"], tie_breaking=tie_breaking)
    assert sorted(vocab) == list(range(300))
    assert [vocab[b] for b in range(256)] == [bytes([b]) for b in range(256)]
    assert vocab[256] == b"<|endoftext|>"
    assert len(merges) == 300 - 257
    for i, (a, b) in enumerate(merges):
        assert vocab[257 + i] == a + b


if __name__ == "__main__":
    train_bpe(DATA_DIR / "TinyStoriesV2-GPT4-train.txt", 10_000, [])
