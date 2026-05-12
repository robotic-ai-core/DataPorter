"""Document tokenization and fixed-length chunking with EOT boundaries.

Accumulates tokens from documents, inserts EOT tokens at document boundaries,
and splits into fixed-length chunks for training.
"""

import numpy as np


class TokenChunker:
    """Accumulates tokenized documents and yields fixed-length chunks.

    Documents are concatenated with EOT tokens between them, then split
    into fixed-length chunks. Partial chunks at the end of a document
    are carried over to the next document.

    Args:
        seq_len: Fixed length of output chunks.
        eot_token_id: End-of-text token ID inserted between documents.
    """

    def __init__(self, seq_len: int, eot_token_id: int):
        self._seq_len = seq_len
        self._eot_token_id = eot_token_id
        self._buffer: list[int] = []

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def buffer_size(self) -> int:
        """Number of tokens currently in the buffer."""
        return len(self._buffer)

    def add_document(self, token_ids: list[int]) -> list[np.ndarray]:
        """Add a tokenized document and return any complete chunks.

        Appends the document's tokens followed by an EOT token to the
        internal buffer, then extracts all complete seq_len chunks.

        Args:
            token_ids: Token IDs for one document (no EOT — added automatically).

        Returns:
            List of complete chunks as uint16 numpy arrays, each of shape [seq_len].
        """
        if not token_ids:
            return []

        self._buffer.extend(token_ids)
        self._buffer.append(self._eot_token_id)

        chunks = []
        while len(self._buffer) >= self._seq_len:
            chunk = np.array(self._buffer[: self._seq_len], dtype=np.uint16)
            chunks.append(chunk)
            self._buffer = self._buffer[self._seq_len :]

        return chunks

    def flush(self) -> list[np.ndarray]:
        """Flush remaining buffer as a padded chunk (if non-empty).

        Pads the remaining tokens with EOT tokens to reach seq_len.
        Call this after all documents have been added.

        Returns:
            List containing at most one padded chunk, or empty if buffer is empty.
        """
        if not self._buffer:
            return []

        # Pad with EOT to fill the chunk
        padded = self._buffer + [self._eot_token_id] * (self._seq_len - len(self._buffer))
        chunk = np.array(padded[: self._seq_len], dtype=np.uint16)
        self._buffer = []
        return [chunk]

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buffer = []
