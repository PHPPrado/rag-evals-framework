"""Evaluator de correspondência exata com normalização + similaridade de unigramas.

Implementa duas visões complementares (determinísticas e baratas):
1. **Exact match normalizado**: 1.0 se as strings normalizadas (minúsculas,
   sem acentos/pontuação) forem idênticas.
2. **Unigram overlap (BLEU-1 / ROUGE-1 simplificados)**: precisão, recall e F1
   sobre unigramas — cobre paráfrases leves sem depender de modelos externos.

O score final é `max(exact_match, unigram_f1)`.
"""

from __future__ import annotations

from collections import Counter

from rag_evals.domain.evaluators import BaseEvaluator
from rag_evals.domain.models import EvalScore
from rag_evals.utils.text import normalize_text, tokenize


class ExactMatchEvaluator(BaseEvaluator):
    """Comparação direta/normalizada + overlap de unigramas (BLEU/ROUGE simplificado)."""

    name = "exact_match"

    def __init__(self, pass_threshold: float = 0.5) -> None:
        self.pass_threshold = pass_threshold

    async def evaluate(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str,
        context: str,
    ) -> EvalScore:
        norm_gen = normalize_text(generated_answer)
        norm_exp = normalize_text(expected_answer)
        exact = 1.0 if norm_gen == norm_exp and norm_exp else 0.0

        gen_tokens = Counter(tokenize(generated_answer))
        exp_tokens = Counter(tokenize(expected_answer))

        overlap = sum((gen_tokens & exp_tokens).values())
        precision = overlap / max(1, sum(gen_tokens.values()))
        recall = overlap / max(1, sum(exp_tokens.values()))
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        score = max(exact, f1)
        return self._build_score(
            score,
            threshold=self.pass_threshold,
            exact_match=bool(exact),
            unigram_precision=round(precision, 4),
            unigram_recall=round(recall, 4),
            unigram_f1=round(f1, 4),
            normalized_generated=norm_gen[:200],
            normalized_expected=norm_exp[:200],
        )
