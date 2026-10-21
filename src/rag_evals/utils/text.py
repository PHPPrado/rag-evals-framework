"""Normalização e tokenização determinísticas para evaluators offline."""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Stopwords mínimas PT-BR + EN usadas para "content tokens" na checagem de fidelidade.
STOPWORDS: frozenset[str] = frozenset(
    {
        # PT-BR
        "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
        "no", "na", "nos", "nas", "pelo", "pela", "pelos", "pelas", "que", "e", "ou",
        "em", "com", "por", "para", "sem", "sob", "sobre", "entre", "ate", "é", "era",
        "era", "sao", "ser", "esta", "estao", "foi", "foram", "tem", "tinha", "muito",
        "mais", "menos", "tambem", "ja", "nao", "sim", "se", "lhe", "ele", "ela", "eles",
        "elas", "eu", "voce", "voce", "nos", "isso", "isto", "aquilo", "esse", "essa",
        "este", "esta", "aquele", "aquela",
        # EN
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
        "are", "was", "were", "be", "been", "it", "this", "that", "as", "at", "by",
        "from", "we", "you", "they", "he", "she",
    }
)


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos, sem pontuação, whitespace colapsado.

    Determinístico: usado pelo ExactMatchEvaluator para comparação justa.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokens normalizados (minúsculos, sem acentos/pontuação)."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def content_tokens(text: str) -> list[str]:
    """Tokens com conteúdo (sem stopwords) — úteis para checagem de alucinação."""
    return [t for t in tokenize(text) if t not in STOPWORDS]
