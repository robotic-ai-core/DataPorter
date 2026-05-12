"""Chat/instruction datasets and templates."""

from .template import apply_chat_template
from .dataset import ChatDataset, ChatStreamDataset

__all__ = ["apply_chat_template", "ChatDataset", "ChatStreamDataset"]
