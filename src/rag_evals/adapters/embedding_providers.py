"""Provedores de embeddings: contrato + implementação local determinística + OpenAI.

O `LocalDeterministicEmbedder` usa a "hashing trick" sobre tokens (bag-of-words
com hash estável via blake2b) para gerar vetores esparsos densificados e
L2-normalizados. Isso permite similaridade de cosseno determinística e offline,
sem custo de API — ideal para testes de regressão em CI.

Para produção/qualidade semântica real, use `OpenAIEmbeddingProvider`.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter

from rag_evals.utils.text import tokenize
from rag_evals.utils.vectors import l2_normalize

DEFAULT_DIM = 512


class EmbeddingProvider(ABC):
    """Contrato de provedor de embeddings."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Retorna o embedding L2-normalizado do texto."""
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embedding de vários textos; base envolve em paralelo via gather."""
        import asyncio

        return list(await asyncio.gather(*(self.embed(t) for t in texts)))


class LocalDeterministicEmbedder(EmbeddingProvider):
    """Embedding determinístico baseado em hash de tokens (offline, sem custo).

    Vetor = soma de features hash(token) com peso TF, L2-normalizado.
    Textos com vocabulário compartilhado ficam mais próximos no espaço,
    aproximando "similaridade semântica lexical" de forma reprodutível.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim deve ser positivo")
        self.dim = dim

    def _token_feature(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        # Sinal estável por token para reduzir colisões de hash (hashing trick com sinal).
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        counts = Counter(tokenize(text))
        for token, tf in counts.items():
            index, sign = self._token_feature(token)
            vec[index] += sign * (1.0 + float(tf))
        return l2_normalize(vec)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Provedor OpenAI (text-embedding-3-small por padrão).

    Requer `pip install openai` e `OPENAI_API_KEY` no ambiente.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependência opcional
            raise ImportError(
                "OpenAIEmbeddingProvider requer `pip install openai`."
            ) from exc
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    async def embed(self, text: str) -> list[float]:  # pragma: no cover - requer rede/chave
        response = await self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


def build_embedding_provider(name: str, **kwargs: object) -> EmbeddingProvider:
    """Factory a partir do settings (`local` ou `openai`)."""
    if name == "local":
        return LocalDeterministicEmbedder(**kwargs)  # type: ignore[arg-type]
    if name == "openai":
        return OpenAIEmbeddingProvider(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Provedor de embeddings desconhecido: {name!r} (use 'local' ou 'openai')")
