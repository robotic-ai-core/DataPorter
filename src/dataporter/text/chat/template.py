"""Chat template formatting and tokenization.

Pure functions for converting raw instruction examples to tokenized
sequences with loss masking.  No dataset or framework dependency —
these are composable building blocks usable from any data pipeline.
"""

from __future__ import annotations

import numpy as np


def apply_chat_template(
    example: dict[str, str],
    tokenizer,
    roles: dict[str, str],
    field_map: dict[str, str],
    seq_len: int,
) -> dict[str, np.ndarray] | None:
    """Tokenize a raw example with chat template and loss masking.

    Formats an instruction example as::

        {query_role} {query_text} {response_role} {response_text} {end_role}

    Then tokenizes and creates a ``loss_mask`` that is True only on
    response tokens (and the end token).

    Args:
        example: Raw example dict with text fields.
        tokenizer: HuggingFace tokenizer, already extended with any
            special tokens the caller needs.
        roles: Maps semantic role names to special token strings.
            Required keys: ``"query"``, ``"response"``, ``"end"``.
        field_map: Maps role names (``"query"``, ``"response"``) to
            keys in *example*.
        seq_len: Fixed output length.  Examples that tokenize to more
            than *seq_len* tokens are **dropped** (returns ``None``).

    Returns:
        Dict with numpy arrays, all of shape ``[seq_len]``:

        - ``input_ids``  (uint16): token IDs
        - ``labels``     (uint16): same as input_ids (shift handled in loss)
        - ``loss_mask``  (bool):   True where loss should be computed

        Returns ``None`` if the tokenized example exceeds *seq_len*.
    """
    query_text = example[field_map["query"]]
    response_text = example[field_map["response"]]

    query_role = roles["query"]
    response_role = roles["response"]
    end_role = roles["end"]

    # Tokenize the query prefix (everything the model should read but
    # NOT be trained to generate).
    query_part = f"{query_role} {query_text} {response_role}"
    query_tokens = tokenizer.encode(query_part, add_special_tokens=False)

    # Tokenize the response (the part the model IS trained to generate).
    response_part = f" {response_text} {end_role}"
    response_tokens = tokenizer.encode(response_part, add_special_tokens=False)

    all_tokens = query_tokens + response_tokens
    n_tokens = len(all_tokens)

    if n_tokens > seq_len:
        return None

    # Pad to seq_len
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    n_pad = seq_len - n_tokens

    input_ids = np.array(all_tokens + [pad_id] * n_pad, dtype=np.uint16)
    labels = input_ids.copy()

    # loss_mask: True only on response tokens (not query, not padding)
    loss_mask = np.zeros(seq_len, dtype=np.bool_)
    loss_mask[len(query_tokens):n_tokens] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
    }
