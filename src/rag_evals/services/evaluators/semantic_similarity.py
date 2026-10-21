"""Evaluator de similaridade semântica por embeddings (cosine).

Usa um `EmbeddingProvider` abstraído:
- `LocalDeterministicEmbedder` (default): offline, determinístico, sem custo.
- `OpenAIEmbeddingProvider`: qualidade real quando há API key.

Score = (1 + cos(generated, expected)) / 2 → mapeado para [0, 1].
"""

from __future__ import annotations

from rag_evals.adapters.embedding_providers import EmbeddingProvider, LocalDeterministicEmbedder
from rag_evals.domain.evaluators import BaseEvaluator
from rag_evals.domain.models import EvalScore
from rag_evals.utils.vectors import cosine_similarity


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Similaridade de cosseno entre embeddings da resposta gerada e da esperada."""

    name = "semantic_similarity"

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        pass_threshold: float = 0.75,
    ) -> None:
        self.embedder = embedder or LocalDeterministicEmbedder()
        self.pass_threshold = pass_threshold

    async def evaluate(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str,
        context: str,
    ) -> EvalScore:
        gen_emb, exp_emb = await self.embedder.embed_batch([generated_answer, expected_answer])
        cosine = cosine_similarity(gen_emb, exp_emb)
        score = (1.0 + cosine) / 2.0
        return self._build_score(
            score,
            threshold=self.pass_threshold,
            cosine=round(cosine, 4),
            embedder=type(self.embedder).__name__,
        )
