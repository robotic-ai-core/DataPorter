"""Composable transform pipeline for the ParquetPrefetcher.

Transforms are callables that take a raw HF document dict and return
a list of rows (each a list[int]) to write to Parquet, or None to skip.

The ``compose`` function chains multiple transforms into a single callable.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# Type alias for prefetcher transforms
# Input: raw HF doc dict -> Output: list of rows (list[int]) or None
PrefetchTransform = Callable[[dict[str, Any]], list[list[int]] | None]


def compose(*transforms: PrefetchTransform) -> PrefetchTransform:
    """Compose multiple transforms into a single pipeline.

    Each transform receives the output of the previous one. The first
    transform receives the raw HF document dict. Subsequent transforms
    receive the list of rows from the previous transform.

    If any transform returns None, the entire pipeline returns None.
    """
    if not transforms:
        raise ValueError("At least one transform is required")
    if len(transforms) == 1:
        return transforms[0]

    def composed(doc: dict[str, Any]) -> list[list[int]] | None:
        result = transforms[0](doc)
        if result is None:
            return None
        for t in transforms[1:]:
            # Subsequent transforms get called per-row
            new_result = []
            for row in result:
                out = t({"input_ids": row})
                if out is not None:
                    new_result.extend(out)
            if not new_result:
                return None
            result = new_result
        return result

    return composed


def tokenize_transform(
    tokenizer_name: str,
    text_field: str = "text",
) -> PrefetchTransform:
    """Create a transform that tokenizes text documents.

    Returns a transform that extracts the text field, tokenizes it,
    and returns token IDs as a single row.
    """

    def transform(doc: dict[str, Any]) -> list[list[int]] | None:
        text = doc.get(text_field, "")
        if not text or not text.strip():
            return None
        tok = _get_tokenizer(tokenizer_name)
        ids = tok.encode(text)
        if not ids:
            return None
        return [ids]

    return transform


def chunk_transform(
    seq_len: int,
    eot_token_id: int = 0,
) -> PrefetchTransform:
    """Create a transform that chunks token sequences into fixed lengths.

    Takes a doc with an "input_ids" field (list[int]) and splits it
    into seq_len chunks, inserting EOT tokens between documents.
    Returns uint16-compatible token lists.
    """
    buffer: list[int] = []

    def transform(doc: dict[str, Any]) -> list[list[int]] | None:
        token_ids = doc.get("input_ids", [])
        if not token_ids:
            return None

        buffer.extend(token_ids)
        buffer.append(eot_token_id)

        chunks = []
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            del buffer[:seq_len]
            chunks.append(chunk)

        return chunks if chunks else None

    return transform


def tokenize_and_chunk(
    tokenizer_name: str,
    seq_len: int,
    eot_token_id: int | None = None,
    text_field: str = "text",
    tokenizer_batch_size: int = 1,
) -> PrefetchTransform:
    """Create a combined tokenize + chunk transform.

    This is more efficient than composing separate tokenize and chunk
    transforms because it avoids intermediate allocations and handles
    the cross-document chunking buffer internally.

    Args:
        tokenizer_name: HF tokenizer name or local path.
        seq_len: Fixed sequence length for output chunks.
        eot_token_id: End-of-text token ID. Auto-detected if None.
        text_field: Name of the text column in HF docs.
        tokenizer_batch_size: Not used in single-doc mode (kept for API compat).
    """
    buffer: list[int] = []
    resolved_eot: list[int | None] = [eot_token_id]

    def transform(doc: dict[str, Any]) -> list[list[int]] | None:
        text = doc.get(text_field, "")
        if not text or not text.strip():
            return None

        tok = _get_tokenizer(tokenizer_name)
        ids = tok.encode(text)
        if not ids:
            return None

        # Resolve EOT on first call
        if resolved_eot[0] is None:
            if hasattr(tok, "token_to_id"):
                resolved_eot[0] = tok.token_to_id("<|endoftext|>") or 0
            elif hasattr(tok, "eos_token_id"):
                resolved_eot[0] = tok.eos_token_id or 0
            else:
                resolved_eot[0] = 0

        buffer.extend(ids)
        buffer.append(resolved_eot[0])

        chunks = []
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            del buffer[:seq_len]
            chunks.append(chunk)

        return chunks if chunks else None

    return transform


# Tokenizer cache (avoid repeated loads)
_tokenizer_cache: dict[str, Any] = {}


class _RawTokenizerWrapper:
    """Wraps raw ``tokenizers.Tokenizer`` so ``.encode()`` returns ``list[int]``."""

    def __init__(self, tok):
        self._tok = tok

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)


def _get_tokenizer(name: str) -> Any:
    """Load and cache a tokenizer.

    Raw ``tokenizers.Tokenizer`` is wrapped so ``.encode()`` returns
    ``list[int]`` instead of ``Encoding`` objects.
    """
    if name not in _tokenizer_cache:
        from pathlib import Path

        path = Path(name) / "tokenizer.json"
        if path.exists():
            from tokenizers import Tokenizer

            _tokenizer_cache[name] = _RawTokenizerWrapper(
                Tokenizer.from_file(str(path))
            )
        else:
            from transformers import AutoTokenizer

            _tokenizer_cache[name] = AutoTokenizer.from_pretrained(name)
    return _tokenizer_cache[name]
