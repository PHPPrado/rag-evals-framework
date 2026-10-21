"""Fixtures compartilhadas da suíte de testes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_evals.domain.models import TestCase  # noqa: E402


@pytest.fixture
def sample_case() -> TestCase:
    return TestCase(
        id="sample-001",
        question="O que é FastAPI?",
        expected_answer="FastAPI é um framework web moderno e rápido para construir APIs com Python.",
        reference_context=(
            "FastAPI é um framework web moderno e rápido para construir APIs com Python, "
            "com suporte nativo a OpenAPI e validação via Pydantic."
        ),
        tags=["fastapi", "basico"],
    )


@pytest.fixture
def sample_dataset() -> list[TestCase]:
    return [
        TestCase(
            id="ds-1",
            question="O que é FastAPI?",
            expected_answer="FastAPI é um framework web moderno e rápido para construir APIs com Python.",
            reference_context="FastAPI é um framework web moderno e rápido para construir APIs com Python.",
            tags=["fastapi"],
        ),
        TestCase(
            id="ds-2",
            question="O que significa RAG?",
            expected_answer="RAG significa Retrieval-Augmented Generation, combinando recuperação e geração.",
            reference_context=(
                "RAG significa Retrieval-Augmented Generation, combinando recuperação e geração "
                "de texto por LLMs para respostas mais precisas."
            ),
            tags=["rag"],
        ),
    ]
