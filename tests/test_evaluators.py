"""Testes dos evaluators: exact_match, semantic_similarity e faithfulness."""

from __future__ import annotations

import pytest

from rag_evals.adapters.embedding_providers import LocalDeterministicEmbedder
from rag_evals.domain.evaluators import BaseEvaluator
from rag_evals.services.evaluators import (
    ExactMatchEvaluator,
    FaithfulnessEvaluator,
    SemanticSimilarityEvaluator,
)


# --- ExactMatchEvaluator ------------------------------------------------------

@pytest.mark.asyncio
async def test_exact_match_identical():
    ev = ExactMatchEvaluator()
    score = await ev.evaluate("q", "FastAPI é um framework.", "FastAPI é um framework.", "ctx")
    assert score.score == 1.0
    assert score.passed
    assert score.details["exact_match"] is True


@pytest.mark.asyncio
async def test_exact_match_normalized_case_accent():
    ev = ExactMatchEvaluator()
    score = await ev.evaluate("q", "FASTAPI e um FRAMEWORK.", "FastAPI é um framework.", "ctx")
    assert score.score == 1.0
    assert score.passed


@pytest.mark.asyncio
async def test_exact_match_partial_unigram_overlap():
    ev = ExactMatchEvaluator()
    score = await ev.evaluate(
        "q",
        "FastAPI é um framework Python moderno.",
        "FastAPI é um framework web moderno para APIs.",
        "ctx",
    )
    assert 0.0 < score.score < 1.0
    assert score.details["unigram_f1"] > 0.3


@pytest.mark.asyncio
async def test_exact_match_completely_different():
    ev = ExactMatchEvaluator()
    score = await ev.evaluate("q", "Resposta totalmente diferente.", "Outra coisa completamente.", "ctx")
    assert score.score < 0.3
    assert not score.passed


# --- SemanticSimilarityEvaluator --------------------------------------------

@pytest.mark.asyncio
async def test_semantic_similarity_identical():
    ev = SemanticSimilarityEvaluator(embedder=LocalDeterministicEmbedder())
    score = await ev.evaluate("q", "python é uma linguagem", "python é uma linguagem", "ctx")
    assert score.score > 0.99
    assert score.passed


@pytest.mark.asyncio
async def test_semantic_similarity_related_higher_than_unrelated():
    ev = SemanticSimilarityEvaluator(embedder=LocalDeterministicEmbedder())
    related = await ev.evaluate(
        "q", "Python é uma linguagem de programação", "linguagem de programação Python", "ctx"
    )
    unrelated = await ev.evaluate(
        "q", "banana é amarela", "carro anda na estrada", "ctx"
    )
    assert related.score > unrelated.score


@pytest.mark.asyncio
async def test_semantic_similarity_deterministic():
    ev = SemanticSimilarityEvaluator(embedder=LocalDeterministicEmbedder())
    s1 = await ev.evaluate("q", "resposta A", "resposta A", "ctx")
    s2 = await ev.evaluate("q", "resposta A", "resposta A", "ctx")
    assert s1.score == s2.score


@pytest.mark.asyncio
async def test_semantic_score_within_bounds():
    ev = SemanticSimilarityEvaluator(embedder=LocalDeterministicEmbedder())
    score = await ev.evaluate("q", "qualquer coisa", "outra coisa", "ctx")
    assert 0.0 <= score.score <= 1.0


# --- FaithfulnessEvaluator ----------------------------------------------------

@pytest.mark.asyncio
async def test_faithfulness_fully_grounded():
    ev = FaithfulnessEvaluator()
    ctx = "FastAPI é um framework Python moderno para APIs web."
    score = await ev.evaluate("q", "FastAPI é um framework Python moderno.", "", ctx)
    assert score.score > 0.9
    assert score.passed


@pytest.mark.asyncio
async def test_faithfulness_hallucinated_tokens():
    ev = FaithfulnessEvaluator()
    ctx = "FastAPI é um framework Python para APIs."
    score = await ev.evaluate("q", "FastAPI é um framework Python com suporte a bancos NoSQL escaláveis.", "", ctx)
    assert score.score < 0.8
    assert not score.passed
    assert "noSQL" in " ".join(score.details["unsupported_tokens"]) or score.details["unsupported_tokens"]


@pytest.mark.asyncio
async def test_faithfulness_empty_answer():
    ev = FaithfulnessEvaluator()
    score = await ev.evaluate("q", "", "", "ctx qualquer")
    assert score.score == 0.0


# --- Contrato BaseEvaluator ---------------------------------------------------

def test_base_evaluator_is_abstract():
    with pytest.raises(TypeError):
        BaseEvaluator()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_all_evaluators_return_eval_score():
    evaluators: list[BaseEvaluator] = [
        ExactMatchEvaluator(),
        SemanticSimilarityEvaluator(embedder=LocalDeterministicEmbedder()),
        FaithfulnessEvaluator(),
    ]
    for ev in evaluators:
        score = await ev.evaluate("q", "resposta gerada", "resposta esperada", "contexto")
        assert 0.0 <= score.score <= 1.0
        assert score.evaluator == ev.name
