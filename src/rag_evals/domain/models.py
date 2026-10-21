"""Modelos de domínio (Pydantic v2) do framework de avaliação RAG."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TestCase(BaseModel):
    """Caso de teste do dataset de ground truth (pergunta → resposta esperada)."""

    id: str = Field(..., min_length=1, description="Identificador único do caso.")
    question: str = Field(..., min_length=1, description="Pergunta feita ao pipeline RAG.")
    expected_answer: str = Field(..., min_length=1, description="Resposta de referência (ground truth).")
    reference_context: str = Field(
        ..., min_length=1, description="Contexto documental que sustenta a resposta esperada."
    )
    tags: list[str] = Field(default_factory=list, description="Tags para filtragem/seleção de casos.")

    @field_validator("question", "expected_answer", "reference_context")
    @classmethod
    def _strip_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("campo não pode ser vazio/apenas espaços")
        return v


class EvalScore(BaseModel):
    """Resultado de um único evaluator aplicado a um caso."""

    evaluator: str = Field(..., description="Nome do evaluator (ex: exact_match).")
    score: float = Field(..., ge=0.0, le=1.0, description="Score normalizado em [0, 1].")
    passed: bool = Field(..., description="Se o score atingiu o threshold do evaluator.")
    details: dict[str, Any] = Field(default_factory=dict, description="Metadados explicativos do score.")


class RAGResponse(BaseModel):
    """Resposta crua retornada pelo pipeline RAG sob teste."""

    answer: str
    contexts: list[str] = Field(default_factory=list, description="Contextos recuperados pelo RAG.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """Resultado completo da avaliação de um caso: resposta gerada + scores."""

    case_id: str
    question: str
    expected_answer: str
    generated_answer: str
    latency_ms: float = Field(..., ge=0.0, description="Latência da chamada ao RAG em milissegundos.")
    scores: list[EvalScore] = Field(default_factory=list)
    passed: bool = Field(..., description="Se TODOS os evaluators passaram no caso.")

    @property
    def score_by_evaluator(self) -> dict[str, float]:
        return {s.evaluator: s.score for s in self.scores}


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class BenchmarkMetrics(BaseModel):
    """Métricas consolidadas do benchmark."""

    total_cases: int = Field(..., ge=0)
    passed_cases: int = Field(..., ge=0)
    accuracy_mean: float = Field(..., ge=0.0, le=1.0, description="Média do ExactMatchEvaluator.")
    relevance_mean: float = Field(..., ge=0.0, le=1.0, description="Média do SemanticSimilarityEvaluator.")
    faithfulness_mean: float = Field(..., ge=0.0, le=1.0, description="Média do FaithfulnessEvaluator.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Score consolidado ponderado.")
    latency_mean_ms: float = Field(..., ge=0.0)
    latency_p95_ms: float = Field(..., ge=0.0)


class BenchmarkReport(BaseModel):
    """Relatório completo de uma execução do benchmark."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(..., ge=0.0)
    dataset_path: str
    threshold: float = Field(..., ge=0.0, le=1.0)
    gate_status: GateStatus
    metrics: BenchmarkMetrics
    results: list[CaseResult] = Field(default_factory=list)

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)


class RunEvalsRequest(BaseModel):
    """Payload opcional do endpoint POST /api/v1/evals/run."""

    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="Override do threshold do gate.")
    tags: list[str] | None = Field(default=None, description="Filtra apenas casos com estas tags.")
    max_concurrency: int | None = Field(default=None, ge=1, le=256)


class EvalRunResponse(BaseModel):
    """Envelope JSON do relatório retornado pela API."""

    status: GateStatus
    report: BenchmarkReport
