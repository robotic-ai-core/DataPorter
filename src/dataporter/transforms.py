"""Composable transform utilities and tokenizer cache.

The ``compose`` function chains callables. The ``_get_tokenizer`` cache
avoids repeated loads across DataLoader workers.
"""

from __future__ import annotations

from typing import Any, Callable


def compose(*transforms: Callable) -> Callable:
    """Compose multiple callables into a single pipeline.

    Each transform receives the output of the previous one.
    If any transform returns None, the entire pipeline returns None.
    """
    if not transforms:
        raise ValueError("At least one transform is required")
    if len(transforms) == 1:
        return transforms[0]

    def composed(x: Any) -> Any:
        result = transforms[0](x)
        for t in transforms[1:]:
            if result is None:
                return None
            result = t(result)
        return result

    return composed


# ---------------------------------------------------------------------------
# Tokenizer cache (shared across DataLoader workers within a process)
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[str, Any] = {}


class _RawTokenizerWrapper:
    """Wraps raw ``tokenizers.Tokenizer`` so ``.encode()`` returns ``list[int]``."""

    def __init__(self, tok: Any):
        self._tok = tok

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)


def get_tokenizer(name: str) -> Any:
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
