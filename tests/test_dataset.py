"""Testes do carregamento e validação do dataset de avaliação."""

from __future__ import annotations

import json

import pytest

from rag_evals.domain.models import TestCase
from rag_evals.services.dataset import filter_dataset_by_tags, load_dataset


def test_load_real_dataset_file():
    from rag_evals.config import PROJECT_ROOT

    cases = load_dataset(PROJECT_ROOT / "data" / "eval_dataset.json")
    assert len(cases) >= 3
    assert all(isinstance(c, TestCase) for c in cases)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nao_existe.json")


def test_load_dataset_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON válido"):
        load_dataset(bad)


def test_load_dataset_not_a_list(tmp_path):
    bad = tmp_path / "obj.json"
    bad.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="lista"):
        load_dataset(bad)


def test_load_dataset_empty_list(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="vazio"):
        load_dataset(empty)


def test_load_dataset_duplicate_ids(tmp_path):
    dup = tmp_path / "dup.json"
    case = {
        "id": "x",
        "question": "q",
        "expected_answer": "a",
        "reference_context": "c",
        "tags": [],
    }
    dup.write_text(json.dumps([case, case]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicado"):
        load_dataset(dup)


def test_load_dataset_invalid_schema(tmp_path):
    bad = tmp_path / "schema.json"
    bad.write_text(json.dumps([{"id": "only-id"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="inválido"):
        load_dataset(bad)


def test_filter_by_tags(sample_dataset):
    filtered = filter_dataset_by_tags(sample_dataset, ["RAG"])
    assert len(filtered) == 1
    assert filtered[0].id == "ds-2"
