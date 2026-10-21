"""Rotas da API de avaliação — POST /api/v1/evals/run."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag_evals.adapters.embedding_providers import build_embedding_provider
from rag_evals.adapters.rag_client import MockRAGClient, RAGClient
from rag_evals.config import settings as default_settings
from rag_evals.domain.models import EvalRunResponse, RunEvalsRequest, TestCase
from rag_evals.services.benchmark_runner import BenchmarkRunner
from rag_evals.services.dataset import filter_dataset_by_tags, load_dataset

router = APIRouter(prefix="/api/v1/evals", tags=["evals"])

# Em produção, substitua por um RAGClient real (HttpRAGClient apontando ao serviço).
_rag_client: RAGClient | None = None


def get_rag_client() -> RAGClient:
    """Resolvedor do RAGClient; mock por padrão para uso offline."""
    global _rag_client
    if _rag_client is None:
        cases = load_dataset(default_settings.dataset_path)
        _rag_client = MockRAGClient.from_test_cases(cases)
    return _rag_client


def set_rag_client(client: RAGClient) -> None:
    """Permite injetar um RAGClient real (usado em testes e produção)."""
    global _rag_client
    _rag_client = client


@router.post("/run", response_model=EvalRunResponse, summary="Executar benchmark de regressão")
async def run_evals(request: RunEvalsRequest | None = None) -> EvalRunResponse:
    """Carrega o dataset, executa o benchmark completo e retorna o relatório JSON."""
    request = request or RunEvalsRequest()
    settings = default_settings

    try:
        dataset = load_dataset(settings.dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar dataset: {exc}") from exc

    if request.tags:
        dataset = filter_dataset_by_tags(dataset, request.tags)
        if not dataset:
            raise HTTPException(
                status_code=400,
                detail=f"Nenhum caso corresponde às tags: {request.tags}",
            )

    try:
        embedder = build_embedding_provider(settings.embedding_provider)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    runner = BenchmarkRunner(
        rag_client=get_rag_client(),
        settings=settings,
    )
    # Se houver provedor customizado, injeta via evaluators do runner
    if embedder is not None:
        from rag_evals.services.benchmark_runner import default_evaluators

        runner.evaluators = default_evaluators(embedder=embedder)

    report = await runner.run(
        dataset,
        threshold=request.threshold,
        dataset_path=str(settings.dataset_path),
        max_concurrency=request.max_concurrency,
    )
    return EvalRunResponse(status=report.gate_status, report=report)


@router.get("/health", tags=["evals"])
async def evals_health() -> dict[str, str]:
    return {"status": "ok"}
