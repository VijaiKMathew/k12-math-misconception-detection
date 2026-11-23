"""
Grandmaster NLP & Transformer Modeling Primitives
"""

from src.qaggle.nlp.losses import (
    InfoNCELoss,
    RootVsSecondaryMarginLoss,
    TraceGroupedMarginLoss,
)
from src.qaggle.nlp.padding import DynamicPaddingCollator
from src.qaggle.nlp.pooling import GeMPooling, MeanPooling, MultiHeadAttentionPooling

__all__ = [
    "DynamicPaddingCollator",
    "GeMPooling",
    "InfoNCELoss",
    "MeanPooling",
    "MultiHeadAttentionPooling",
    "RootVsSecondaryMarginLoss",
    "TraceGroupedMarginLoss",
]
