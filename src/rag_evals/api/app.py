"""Aplicação FastAPI que expõe o benchmark de regressão via REST."""

from __future__ import annotations

from fastapi import FastAPI

from rag_evals import __version__
from rag_evals.api.routes.evals import router as evals_router


def create_app() -> FastAPI:
    """Factory da aplicação FastAPI."""
    app = FastAPI(
        title="RAG Evals & Semantic Regression Benchmarking Framework",
        version=__version__,
        description=(
            "Execute testes de regressão semântica e avaliações automatizadas "
            "sobre pipelines de RAG/LLM antes de ir para produção."
        ),
    )
    app.include_router(evals_router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
