"""Gatekeeper CLI de regressão semântica para CI/CD (via pytest).

Invoca a suíte `pytest` sobre o teste-gate (`tests/test_gate.py`) que executa o
benchmark completo e **falha (exit code 1)** se o score consolidado for
menor que o threshold configurado (`RAG_EVAL_THRESHOLD`, padrão 0.80).

Uso:
    python run_evals.py
    RAG_EVAL_THRESHOLD=0.90 python run_evals.py
    python run_evals.py --dataset data/eval_dataset.json --report reports/benchmark.json
    python run_evals.py -v -x
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
GATE_TEST = ROOT / "tests" / "test_gate.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa a suíte de benchmark de regressão semântica via pytest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Caminho do dataset JSON.")
    parser.add_argument("--threshold", type=float, default=None, help="Score mínimo para gate passar.")
    parser.add_argument("--report", type=Path, default=None, help="Salva relatório JSON neste caminho.")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument(
        "pytest_args",
        nargs="*",
        default=[],
        help="Args extras repassados ao pytest (ex: -v -x -k padrão).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Expõe configurações do CLI como env vars para o teste-gate ler.
    if args.dataset is not None:
        os.environ["RAG_EVAL_DATASET_PATH"] = str(args.dataset)
    if args.threshold is not None:
        os.environ["RAG_EVAL_THRESHOLD"] = str(args.threshold)
    if args.report is not None:
        os.environ["RAG_EVAL_REPORT_PATH"] = str(args.report)
    if args.max_concurrency is not None:
        os.environ["RAG_EVAL_MAX_CONCURRENCY"] = str(args.max_concurrency)

    pytest_args = [str(GATE_TEST), *args.pytest_args]
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("\n[GATE] PASSOU - pipeline pronto para producao.")
    else:
        print("\n[GATE] FALHOU - regressao detectada. Bloqueie o deploy.")
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
