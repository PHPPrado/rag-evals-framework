"""Carregamento e validação do dataset de avaliação (ground truth)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from rag_evals.domain.models import TestCase


def load_dataset(path: str | Path) -> list[TestCase]:
    """Lê um arquivo JSON com uma lista de casos e valida cada um como `TestCase`.

    Levanta `ValueError` com mensagem acionável para erros de estrutura ou schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset não é JSON válido: {path} ({exc})") from exc

    if not isinstance(raw, list):
        raise ValueError(f"Dataset deve ser uma lista de objetos TestCase: {path}")

    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        try:
            case = TestCase.model_validate(item)
        except ValidationError as exc:
            raise ValueError(f"Caso inválido no índice {index} do dataset {path}: {exc}") from exc
        if case.id in seen_ids:
            raise ValueError(f"ID duplicado no dataset: {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise ValueError(f"Dataset vazio: {path}")
    return cases


def filter_dataset_by_tags(cases: list[TestCase], tags: list[str]) -> list[TestCase]:
    """Mantém apenas os casos que contêm alguma das tags informadas."""
    wanted = {t.strip().lower() for t in tags}
    return [c for c in cases if wanted & {t.strip().lower() for t in c.tags}]
