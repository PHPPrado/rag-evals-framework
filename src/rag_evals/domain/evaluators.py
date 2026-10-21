"""Contrato abstrato (ABC) que todo evaluator deve implementar."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_evals.domain.models import EvalScore


class BaseEvaluator(ABC):
    """Interface única de evaluators.

    Cada evaluator recebe a pergunta, a resposta gerada pelo RAG, a resposta
    esperada (ground truth) e o contexto de referência, e devolve um `EvalScore`
    normalizado em [0, 1].

    Implementações DEVEM ser determinísticas (mesmo input → mesmo output) para
    que o benchmark sirva como gate confiável de regressão.
    """

    #: Nome estável usado em relatórios e configurações.
    name: str = "base"

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str,
        context: str,
    ) -> EvalScore:
        """Avalia uma resposta gerada contra o ground truth e retorna um EvalScore."""
        raise NotImplementedError

    def _build_score(self, score: float, *, threshold: float = 0.5, **details: object) -> EvalScore:
        """Helper para subclasses construírem EvalScore com clamp em [0, 1]."""
        clamped = max(0.0, min(1.0, score))
        return EvalScore(evaluator=self.name, score=clamped, passed=clamped >= threshold, details=details)
