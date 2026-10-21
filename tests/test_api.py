"""Testes da API FastAPI — POST /api/v1/evals/run."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_evals.api.app import create_app
from rag_evals.api.routes.evals import get_rag_client, set_rag_client
from rag_evals.adapters.rag_client import MockRAGClient
from rag_evals.config import PROJECT_ROOT
from rag_evals.services.dataset import load_dataset


@pytest.fixture
def client():
    # Reseta o RAG client para o mock default antes de cada teste
    dataset = load_dataset(PROJECT_ROOT / "data" / "eval_dataset.json")
    set_rag_client(MockRAGClient.from_test_cases(dataset))
    app = create_app()
    with TestClient(app) as c:
        yield c
    set_rag_client(None)  # type: ignore[arg-type]


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_run_evals_returns_report(client: TestClient):
    response = client.post("/api/v1/evals/run", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in ("passed", "failed")
    report = payload["report"]
    assert report["metrics"]["total_cases"] >= 3
    assert 0.0 <= report["metrics"]["overall_score"] <= 1.0
    assert report["gate_status"] in ("passed", "failed")
    assert len(report["results"]) == report["metrics"]["total_cases"]


def test_run_evals_with_threshold_override(client: TestClient):
    response = client.post("/api/v1/evals/run", json={"threshold": 0.99})
    assert response.status_code == 200
    payload = response.json()
    # Threshold absurdamente alto → deve falhar
    assert payload["status"] == "failed"


def test_run_evals_with_tags_filter(client: TestClient):
    response = client.post("/api/v1/evals/run", json={"tags": ["fastapi"]})
    assert response.status_code == 200
    report = response.json()["report"]
    assert report["metrics"]["total_cases"] == 1


def test_run_evals_with_invalid_tags(client: TestClient):
    response = client.post("/api/v1/evals/run", json={"tags": ["tag-inexistente"]})
    assert response.status_code == 400
