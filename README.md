# RAG Evaluators & Semantic Regression Benchmarking Framework

Framework em **Python 3.11+** para executar **testes de regressão semântica** e
**avaliações automatizadas (Evals)** sobre pipelines de RAG/LLM — permitindo validar
se mudanças em prompts, modelos ou configurações **degradaram a qualidade** das
respostas **antes de ir para produção**.

> Roda **100% offline e sem custo** por padrão (embeddings determinísticos locais,
> RAG mockado). Pronto para funcionar como **gate de CI/CD** (exit code 1 abaixo do
> threshold) e exposto via **API REST (FastAPI)**.

---

## Sumário

- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Rodando os testes offline (sem custo)](#rodando-os-testes-offline-sem-custo)
- [Gate de regressão via CLI (CI/CD)](#gate-de-regressão-via-cli-cicd)
- [API REST](#api-rest)
- [Integração com CI/CD](#integração-com-cicd)
- [Dataset de avaliação (ground truth)](#dataset-de-avaliação-ground-truth)
- [Métricas e Evaluators](#métricas-e-evaluators)
- [Configuração (variáveis de ambiente)](#configuração-variáveis-de-ambiente)
- [Como estender o framework](#como-estender-o-framework)

---

## Arquitetura

Adotamos **Clean Architecture** em camadas simples e desacopladas:

```
Nova pasta/
├── pyproject.toml              # Metadados + config do pytest (asyncio_mode=auto)
├── requirements.txt            # Dependências (core + dev)
├── run_evals.py                # CLI gatekeeper (pytest) → exit code 0/1
├── data/
│   └── eval_dataset.json       # Dataset de ground truth (4 casos)
├── reports/                    # Relatórios JSON gerados pelo gate (gitignored)
├── src/rag_evals/
│   ├── config.py               # Settings via pydantic-settings (env RAG_EVAL_*)
│   ├── domain/                 # CAMADA DE DOMÍNIO (sem dependências externas)
│   │   ├── models.py           #   Pydantic v2: TestCase, EvalScore, BenchmarkReport...
│   │   └── evaluators.py       #   BaseEvaluator (abc.ABC) — contrato dos evaluators
│   ├── adapters/               # CAMADA DE ADAPTERS (detalhes de infraestrutura)
│   │   ├── embedding_providers.py  # EmbeddingProvider ABC + Local (mock) + OpenAI
│   │   └── rag_client.py           # RAGClient ABC + Mock + HTTP (pipeline real)
│   ├── services/               # CAMADA DE SERVIÇOS (casos de uso)
│   │   ├── dataset.py          #   Load/validação/filter do dataset
│   │   ├── benchmark_runner.py #   Motor asyncio (gather + semáforo) + agregação
│   │   └── evaluators/         #   Implementações concretas dos evaluators
│   │       ├── exact_match.py
│   │       ├── semantic_similarity.py
│   │       └── faithfulness.py
│   ├── utils/                  # Utilitários puros (normalização, cosine)
│   └── api/                    # CAMADA DE API (FastAPI)
│       ├── app.py              #   Factory da aplicação
│       └── routes/evals.py     #   POST /api/v1/evals/run
└── tests/                      # Suíte de testes (33 testes)
    ├── conftest.py
    ├── test_dataset.py
    ├── test_evaluators.py
    ├── test_benchmark_runner.py
    ├── test_api.py
    └── test_gate.py            # O "gate" executado por run_evals.py no CI
```

**Fluxo de execução:**

```
TestCase[] ──► BenchmarkRunner ──┬──► RAGClient.answer()   (assíncrono, paralelo via asyncio.gather)
                                 │
                                 └──► BaseEvaluator.evaluate() × 3  (assíncrono)
                                          │
                                          ▼
                          BenchmarkMetrics (médias + p95) ──► GateStatus (PASSED/FAILED)
```

---

## Instalação

```bash
# 1. (Opcional) ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 2. Dependências
pip install -r requirements.txt

# 3. (Opcional) instalar como pacote editável
pip install -e .
```

Requisitos: **Python 3.11+**, `pydantic v2`, `fastapi`, `pytest`, `pytest-asyncio`.

---

## Rodando os testes offline (sem custo)

Toda a suíte roda **sem internet, sem API key e sem custo** — o embeddings provider
é o `LocalDeterministicEmbedder` (hashing determinístico) e o RAG é o `MockRAGClient`.

```bash
# Suíte completa (33 testes)
python -m pytest tests/ -v

# Apenas os testes de unitários dos evaluators
python -m pytest tests/test_evaluators.py -v
```

### Executar o benchmark isoladamente

```bash
python run_evals.py --report reports/benchmark.json
```

Saída esperada (dados reais desta execução):

```
overall_score=0.9642 (threshold=0.80)
accuracy=1.0000  relevance=1.0000  faithfulness=0.8750
latency_mean=16.5ms  p95=16.5ms
[GATE] PASSOU - pipeline pronto para producao.
```

---

## Gate de regressão via CLI (CI/CD)

`run_evals.py` invoca a **suíte pytest** (`tests/test_gate.py`) e propaga o
**exit code do pytest**:

---

## API REST

Suba o servidor:

```bash
uvicorn rag_evals.api.app:app --app-dir src --reload
# ou, após `pip install -e .`:
uvicorn rag_evals.api.app:app --reload
```

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Healthcheck (`{"status":"ok","version":"0.1.0"}`) |
| `POST` | `/api/v1/evals/run` | Executa o benchmark e retorna o relatório JSON |
| `GET` | `/api/v1/evals/health` | Healthcheck específico do módulo de evals |
| `GET` | `/docs` | Swagger UI (OpenAPI) |

### Executar benchmark via REST

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/evals/run" \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.80}'
```

Resposta (truncada):

```json
{
  "status": "passed",
  "report": {
    "run_id": "b3f1c2d4e5a6",
    "threshold": 0.8,
    "gate_status": "passed",
    "metrics": {
      "total_cases": 4,
      "passed_cases": 4,
      "accuracy_mean": 1.0,
      "relevance_mean": 1.0,
      "faithfulness_mean": 0.875,
      "overall_score": 0.9642,
      "latency_mean_ms": 16.5,
      "latency_p95_ms": 16.5
    },
    "results": []
  }
}
```

### Payload opcional

```json
{
  "threshold": 0.9,
  "tags": ["fastapi"],
  "max_concurrency": 16
}
```

- `threshold`: override do threshold do gate
- `tags`: roda apenas casos com essas tags
- `max_concurrency`: paralelismo do runner

---

## Integração com CI/CD

### GitHub Actions

```yaml
name: semantic-regression-gate

on:
  pull_request:
    branches: [main]

jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Rodar gate de regressão semântica
        env:
          RAG_EVAL_THRESHOLD: "0.80"
        run: python run_evals.py --report reports/benchmark.json
      - name: Publicar relatório
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-report
          path: reports/benchmark.json
```

Se `python run_evals.py` retornar **1**, o job falha e o PR é bloqueado.

### GitLab CI

```yaml
semantic-eval:
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - RAG_EVAL_THRESHOLD=0.80 python run_evals.py --report reports/benchmark.json
  artifacts:
    when: always
    paths: [reports/benchmark.json]
```

---

## Dataset de avaliação (ground truth)

Arquivo: [`data/eval_dataset.json`](data/eval_dataset.json) — lista de objetos `TestCase`:

```json
{
  "id": "case-001",
  "question": "O que é FastAPI e quais são suas principais características?",
  "expected_answer": "FastAPI é um framework web moderno e rápido...",

---

## Métricas e Evaluators

### Contrato `BaseEvaluator` (ABC)

```python
# src/rag_evals/domain/evaluators.py
class BaseEvaluator(ABC):
    name: str = "base"

    @abstractmethod
    async def evaluate(
        self, question: str, generated_answer: str,
        expected_answer: str, context: str,
    ) -> EvalScore: ...
```

### Implementações

| Evaluator | Arquivo | O que mede | Score |
|---|---|---|---|
| `ExactMatchEvaluator` | `services/evaluators/exact_match.py` | Correspondência exata normalizada (sem acento/caixa) + F1 de unigramas (BLEU-1/ROUGE-1 simplificados) | `max(exact, f1)` |
| `SemanticSimilarityEvaluator` | `services/evaluators/semantic_similarity.py` | Similaridade de cosseno entre embeddings das respostas | `(1 + cos) / 2` |
| `FaithfulnessEvaluator` | `services/evaluators/faithfulness.py` | Quanto da resposta é sustentado pelo contexto (detector de alucinação) | fração de tokens suportados |

### Provedores de embeddings

| Provedor | Custo | Determinístico | Uso |
|---|---|---|---|
| `LocalDeterministicEmbedder` | **Grátis** | Sim | Default — CI offline |
| `OpenAIEmbeddingProvider` | Pago (API) | Não | Qualidade semântica real |

```bash
pip install openai
export OPENAI_API_KEY=sk-...
export RAG_EVAL_EMBEDDING_PROVIDER=openai
```

### Score consolidado

```
overall = w_acc * accuracy_mean + w_rel * relevance_mean + w_fai * faithfulness_mean
```

Pesos padrão: **0.40 / 0.40 / 0.20** (configuráveis via env). O **gate passa** quando
`overall >= threshold` (padrão `0.80`).

Latências: **média** e **p95** (interpolação linear) das chamadas ao RAG.

---

## Configuração (variáveis de ambiente)

Todas as settings usam o prefixo **`RAG_EVAL_`** (via `pydantic-settings`, lê também `.env`).

| Variável | Padrão | Descrição |
|---|---|---|
| `RAG_EVAL_THRESHOLD` | `0.80` | Score mínimo consolidado para o gate passar |
| `RAG_EVAL_MAX_CONCURRENCY` | `8` | Paralelismo do `asyncio.gather` |
| `RAG_EVAL_DATASET_PATH` | `data/eval_dataset.json` | Caminho do dataset |
| `RAG_EVAL_REPORT_PATH` | *(não definido)* | Se definido, o gate salva o relatório JSON |
| `RAG_EVAL_ACCURACY_WEIGHT` | `0.40` | Peso do exact_match |
| `RAG_EVAL_RELEVANCE_WEIGHT` | `0.40` | Peso do semantic_similarity |
| `RAG_EVAL_FAITHFULNESS_WEIGHT` | `0.20` | Peso do faithfulness |
| `RAG_EVAL_EMBEDDING_PROVIDER` | `local` | `local` ou `openai` |
| `OPENAI_API_KEY` | — | Necessário só quando provider = `openai` |

---

## Como estender o framework

### 1. Novo evaluator

```python
# src/rag_evals/services/evaluators/bertscore.py
from rag_evals.domain.evaluators import BaseEvaluator
from rag_evals.domain.models import EvalScore

class BertScoreEvaluator(BaseEvaluator):
    name = "bertscore"

    async def evaluate(self, question, generated_answer, expected_answer, context) -> EvalScore:
        score = await compute_bertscore(generated_answer, expected_answer)
        return self._build_score(score, threshold=0.85, model="bert-base-uncased")
```

Registre no `default_evaluators()` (services/benchmark_runner.py) — as médias por
nome de evaluator são agregadas automaticamente.

### 2. Conectar um RAG real

```python
from rag_evals.adapters.rag_client import HttpRAGClient
from rag_evals.services.benchmark_runner import BenchmarkRunner

client = HttpRAGClient("http://localhost:8001/rag/answer")
runner = BenchmarkRunner(rag_client=client)
report = await runner.run(dataset)
```

Na API, injete o client com `set_rag_client(HttpRAGClient(...))` antes de subir o app.

### 3. Trocar o embedder para OpenAI

```python
from rag_evals.adapters.embedding_providers import OpenAIEmbeddingProvider
from rag_evals.services.benchmark_runner import BenchmarkRunner, default_evaluators

runner = BenchmarkRunner(rag_client=client)
runner.evaluators = default_evaluators(embedder=OpenAIEmbeddingProvider(model="text-embedding-3-small"))
```

---

## Licença

MIT — use e adapte livremente.

  "reference_context": "FastAPI é um framework web moderno e rápido...",
  "tags": ["fastapi", "framework", "basico"]
}
```

**Campos (`src/rag_evals/domain/models.py`):**

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `id` | `str` | Sim | Identificador único (validado contra duplicatas) |
| `question` | `str` | Sim | Pergunta feita ao pipeline RAG |
| `expected_answer` | `str` | Sim | Resposta de referência (ground truth) |
| `reference_context` | `str` | Sim | Contexto documental que sustenta a resposta |
| `tags` | `list[str]` | Não | Tags para filtragem/seleção de casos |

Adicione novos casos editando o JSON — o loader valida schema com Pydantic e
rejeita IDs duplicados.


| Cenário | Exit code | Significado |
|---|---|---|
| Score consolidado **≥ threshold** | `0` | Deploy liberado |
| Score consolidado **< threshold** | `1` | Regressão — bloquear deploy |
| Erro de infra (dataset inválido etc.) | `2` | Erro de execução |

```bash
# Gate com threshold padrão (0.80)
python run_evals.py

# Threshold mais estrito + gerar relatório JSON
RAG_EVAL_THRESHOLD=0.90 python run_evals.py --report reports/benchmark.json

# Threshold via flag CLI (equivale à env var)
python run_evals.py --threshold 0.90

# Args extras repassados ao pytest
python run_evals.py -v -x
```

**Validação executada neste ambiente:**

- `python run_evals.py` → **exit 0** (score 0.9642 ≥ 0.80)
- `RAG_EVAL_THRESHOLD=0.99 python run_evals.py` → **exit 1** (score < 0.99)
