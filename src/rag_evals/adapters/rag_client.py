"""Interface do pipeline RAG sob teste + mocks para execução offline.

O benchmark nunca acoplа a um RAG específico: ele consome o contrato `RAGClient`.
Em produção, implemente `RAGClient` apontando para o seu serviço real (HTTP,
grafo LangChain/LlamaIndex, etc.). Em CI offline, use `MockRAGClient`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from rag_evals.domain.models import RAGResponse


class RAGClient(ABC):
    """Contrato mínimo de um pipeline RAG a ser avaliado."""

    @abstractmethod
    async def answer(self, question: str) -> RAGResponse:
        """Recebe a pergunta e devolve a resposta (+ contextos recuperados)."""
        raise NotImplementedError


class MockRAGClient(RAGClient):
    """Mock determinístico que responde com um "golden answer" por pergunta.

    Qualquer pergunta fora do mapa recebe uma resposta padrão de baixa qualidade
    (simulando uma falha de recuperação), o que permite testar o gate de regressão.
    """

    DEFAULT_ANSWER = "Desculpe, não encontrei informação suficiente para responder."

    def __init__(
        self,
        golden_answers: dict[str, str] | None = None,
        contexts: dict[str, list[str]] | None = None,
        latency_ms: float = 5.0,
        fail_rate: float = 0.0,
    ) -> None:
        # Chaves normalizadas por convenção do chamador (aqui normalizamos por segurança).
        self._golden = {k.strip().lower(): v for k, v in (golden_answers or {}).items()}
        self._contexts = {k.strip().lower(): v for k, v in (contexts or {}).items()}
        self._latency_ms = latency_ms
        self._fail_rate = fail_rate

    @classmethod
    def from_test_cases(cls, cases: list[Any], latency_ms: float = 5.0) -> "MockRAGClient":
        """Constrói o mock a partir dos TestCases (simula um RAG 'perfeito' no dataset)."""
        golden = {c.question: c.expected_answer for c in cases}
        contexts = {c.question: [c.reference_context] for c in cases}
        return cls(golden_answers=golden, contexts=contexts, latency_ms=latency_ms)

    async def answer(self, question: str) -> RAGResponse:
        await asyncio.sleep(self._latency_ms / 1000.0)  # simula latência de rede/modelo
        key = question.strip().lower()
        if self._fail_rate > 0.0 and (hash(key) % 100) < int(self._fail_rate * 100):
            return RAGResponse(answer=self.DEFAULT_ANSWER, contexts=[], metadata={"mock": "miss"})
        answer = self._golden.get(key, self.DEFAULT_ANSWER)
        return RAGResponse(
            answer=answer,
            contexts=self._contexts.get(key, []),
            metadata={"mock": "hit" if key in self._golden else "default"},
        )


class HttpRAGClient(RAGClient):
    """Cliente HTTP genérico para um RAG real exposto via REST.

    Espera um endpoint que aceite `POST {"question": "..."}` e responda
    `{"answer": "...", "contexts": ["..."]}`.
    """

    def __init__(self, endpoint_url: str, timeout_s: float = 30.0, headers: dict[str, str] | None = None) -> None:
        import httpx

        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            headers=headers or {},
        )
        self._endpoint_url = endpoint_url

    async def answer(self, question: str) -> RAGResponse:  # pragma: no cover - requer serviço externo
        response = await self._client.post(self._endpoint_url, json={"question": question})
        response.raise_for_status()
        data = response.json()
        return RAGResponse(
            answer=data.get("answer", ""),
            contexts=list(data.get("contexts", [])),
            metadata={"transport": "http"},
        )

    async def aclose(self) -> None:  # pragma: no cover
        await self._client.aclose()
