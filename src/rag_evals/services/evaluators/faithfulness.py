"""Evaluator de fidelidade (groundedness) — detector estrutural de alucinação.

Estratégia: os "content tokens" (sem stopwords) da resposta gerada devem estar
presentes (suporte léxico-normalizado) no contexto recuperado. Respostas que
introduzem termos ausentes do contexto recebem score reduzido proporcionalmente
ao vocabulário não suportado.

É uma heurística estrutural barata e determinística. O contrato permite trocar
por um LLM-as-judge no futuro (basta implementar `BaseEvaluator`).
"""

from __future__ import annotations

from rag_evals.domain.evaluators import BaseEvaluator
from rag_evals.domain.models import EvalScore
from rag_evals.utils.text import content_tokens, normalize_text


class FaithfulnessEvaluator(BaseEvaluator):
    """Mede quanto da resposta gerada é sustentado pelo contexto recuperado."""

    name = "faithfulness"

    def __init__(self, pass_threshold: float = 0.70) -> None:
        self.pass_threshold = pass_threshold

    async def evaluate(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str,
        context: str,
    ) -> EvalScore:
        answer_tokens = content_tokens(generated_answer)
        if not answer_tokens:
            return self._build_score(
                0.0,
                threshold=self.pass_threshold,
                reason="resposta_vazia_ou_só_stopwords",
                unsupported_tokens=[],
            )

        context_norm = normalize_text(context)
        unsupported = [t for t in set(answer_tokens) if t not in context_norm]
        supported_ratio = 1.0 - (len(unsupported) / len(set(answer_tokens)))
        score = supported_ratio

        return self._build_score(
            score,
            threshold=self.pass_threshold,
            content_token_count=len(answer_tokens),
            unsupported_tokens=sorted(unsupported)[:20],
            supported_ratio=round(supported_ratio, 4),
        )
