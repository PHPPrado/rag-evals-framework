"""Configuração central do framework via variáveis de ambiente (prefixo RAG_EVAL_)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto (sobe 3 níveis a partir de src/rag_evals/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings carregadas de `.env` ou variáveis de ambiente com prefixo `RAG_EVAL_`."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_EVAL_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Gate de regressão ---
    threshold: float = Field(default=0.80, ge=0.0, le=1.0,
                             description="Score mínimo consolidado para o gate passar.")
    latency_budget_ms: float | None = Field(
        default=None,
        description="Se definido, o gate também falha quando a latência p95 excede este orçamento.",
    )

    # --- Execução ---
    max_concurrency: int = Field(default=8, ge=1, le=256,
                                 description="Número máximo de casos avaliados em paralelo.")
    dataset_path: Path = Field(default=PROJECT_ROOT / "data" / "eval_dataset.json")
    report_dir: Path = Field(default=PROJECT_ROOT / "reports")
    report_path: Path | None = Field(
        default=None,
        description="Se definido (RAG_EVAL_REPORT_PATH), o teste-gate salva o relatório JSON aqui.",
    )

    # --- Pesos do score consolidado (soma ideal = 1.0) ---
    accuracy_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    faithfulness_weight: float = Field(default=0.20, ge=0.0, le=1.0)

    # --- Provedores ---
    embedding_provider: str = Field(
        default="local",
        description="'local' (determinístico, offline, sem custo) ou 'openai'.",
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    @property
    def weight_vector(self) -> tuple[float, float, float]:
        """Normaliza os pesos para que somem 1.0."""
        total = self.accuracy_weight + self.relevance_weight + self.faithfulness_weight
        if total <= 0:
            raise ValueError("A soma dos pesos (accuracy/relevance/faithfulness) deve ser > 0.")
        return (
            self.accuracy_weight / total,
            self.relevance_weight / total,
            self.faithfulness_weight / total,
        )


settings = Settings()
