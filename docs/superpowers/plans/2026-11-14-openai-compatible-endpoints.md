# OpenAI-compatible Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point every LLM call (agents, generators, MLflow judges) at any OpenAI-compatible service, such as NVIDIA NIM, by editing `.env` only.

**Architecture:** `Settings` gains `LLM_API_KEY` (with `OPENAI_API_KEY` as a legacy alias) and an optional `LLM_BASE_URL`. `src/providers.py` builds one `OpenAIProvider` from them and switches the agent model to Chat Completions (`OpenAIChatModel`). It also exposes the judge model URI and the LiteLLM `inference_params` that every `make_judge` call passes.

**Tech Stack:** Python 3.12, pydantic-settings, pydantic-ai 1.47 (`OpenAIChatModel`, `OpenAIProvider`), MLflow 3.8.1 `make_judge` (LiteLLM-backed), pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-openai-compatible-endpoints-design.md`

## Global Constraints

- Env var names are exactly `LLM_API_KEY`, `LLM_BASE_URL`, `MODEL_NAME`, `JUDGE_MODEL_NAME`.
- `LLM_BASE_URL` unset or blank means OpenAI's default endpoint.
- A `.env` that sets only `OPENAI_API_KEY` must keep working.
- Agent and judges share one endpoint and key (no per-role endpoints).
- All agent calls use Chat Completions (`OpenAIChatModel`); no `OpenAIResponsesModel` remains in `src/`.
- Settings tests construct `Settings(_env_file=None, ...)` so the developer's local `.env` cannot leak in.
- Work on a feature branch, not `main`. Baseline before Task 1: `uv run pytest -q`, giving 41 passed and 6 skipped.

## Review Focus

1. `LLM_BASE_URL=` left blank (as copied from `.env.example`) must behave as the OpenAI default, not crash the client. Pinned in Task 1 (`test_blank_base_url_means_default`).
2. An old `.env` with only `OPENAI_API_KEY` must still load. Pinned in Task 1 (`test_legacy_openai_api_key_still_accepted`).
3. With both keys set, `LLM_API_KEY` must win. Pinned in Task 1 (`test_llm_api_key_wins_over_legacy`).
4. With no key at all, loading must fail with a validation error rather than a later 401. Pinned in Task 1 (`test_missing_key_fails`).
5. Namespaced model IDs with a slash (`meta/llama-3.3-70b-instruct`) must reach the API unchanged. Pinned in Task 2 (`test_build_model_accepts_namespaced_id`). The judge side is already verified: `openai:/meta/x` parses to model `meta/x`.

---

### Task 1: Settings for a generic endpoint

**Files:**
- Modify: `src/settings.py`
- Modify: `src/providers.py:11` (only rename `settings.OPENAI_API_KEY` to `settings.LLM_API_KEY`, to keep the suite green)
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `Settings.LLM_API_KEY: str`, validated from env `LLM_API_KEY` or else `OPENAI_API_KEY`. `Settings.LLM_BASE_URL: str | None`, which is `None` when the value is unset, empty or whitespace. The `OPENAI_API_KEY` field is removed.

- [ ] **Step 1: Rewrite `tests/test_settings.py`**

Update the three existing tests to use `LLM_API_KEY` and `_env_file=None`. In `test_settings_default_values`, also `delenv("LLM_BASE_URL")` and assert `settings.LLM_BASE_URL is None`. Add these tests (each calls `monkeypatch.delenv(..., raising=False)` on whichever of `LLM_API_KEY`, `OPENAI_API_KEY` and `LLM_BASE_URL` it doesn't set):

```python
def test_legacy_openai_api_key_still_accepted(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    assert Settings(_env_file=None).LLM_API_KEY == "legacy-key"


def test_llm_api_key_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "new-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    assert Settings(_env_file=None).LLM_API_KEY == "new-key"


def test_base_url_read_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    assert Settings(_env_file=None).LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_base_url_means_default(monkeypatch, blank):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", blank)
    assert Settings(_env_file=None).LLM_BASE_URL is None


def test_missing_key_fails(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
```

(Imports: `pytest` and `from pydantic import ValidationError`.)

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL. `Settings` has no `LLM_API_KEY` or `LLM_BASE_URL` yet.

- [ ] **Step 3: Implement the fields in `src/settings.py`**

- `LLM_API_KEY: str = Field(validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"))`, replacing `OPENAI_API_KEY`.
- `LLM_BASE_URL: str | None = None`, plus a `field_validator("LLM_BASE_URL", mode="before")` that strips the value and returns `None` when it's empty.
- Rename the attribute at `src/providers.py:11` to `settings.LLM_API_KEY`.

- [ ] **Step 4: Run the full suite and confirm it passes**

Run: `uv run pytest -q`
Expected: all pass (41 baseline plus the new settings tests), 6 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/settings.py src/providers.py tests/test_settings.py
git commit -m "feat: add LLM_API_KEY/LLM_BASE_URL settings with OPENAI_API_KEY fallback"
```

---

### Task 2: Chat-Completions provider and judge connection params

**Files:**
- Modify: `src/providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `Settings.LLM_API_KEY`, `Settings.LLM_BASE_URL`, `Settings.JUDGE_MODEL_NAME` (Task 1).
- Produces (all in `src/providers.py`):
  - `build_provider(s: Settings) -> OpenAIProvider`
  - `build_model(model_name: str) -> OpenAIChatModel` (same name as today; the return type changes)
  - `agent_model: OpenAIChatModel`
  - `judge_model_uri: str`, equal to `f"openai:/{settings.JUDGE_MODEL_NAME}"`
  - `build_judge_inference_params(s: Settings) -> dict[str, str]`
  - `judge_inference_params: dict[str, str]`, equal to `build_judge_inference_params(settings)`

- [ ] **Step 1: Write the failing tests in `tests/test_providers.py`**

In the existing test, change `OpenAIResponsesModel` to `OpenAIChatModel`. Add:

```python
NIM = "https://integrate.api.nvidia.com/v1"


def test_provider_uses_configured_base_url():
    s = Settings(LLM_API_KEY="k", LLM_BASE_URL=NIM, _env_file=None)
    assert providers.build_provider(s).base_url == NIM + "/"


def test_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    s = Settings(LLM_API_KEY="k", _env_file=None)
    assert providers.build_provider(s).base_url == "https://api.openai.com/v1/"


def test_build_model_accepts_namespaced_id():
    assert providers.build_model("meta/llama-3.3-70b-instruct").model_name == (
        "meta/llama-3.3-70b-instruct"
    )


def test_judge_params_include_base_url_when_set():
    s = Settings(LLM_API_KEY="k", LLM_BASE_URL=NIM, _env_file=None)
    assert providers.build_judge_inference_params(s) == {"api_key": "k", "api_base": NIM}


def test_judge_params_omit_base_url_when_unset():
    s = Settings(LLM_API_KEY="k", _env_file=None)
    assert providers.build_judge_inference_params(s) == {"api_key": "k"}


def test_judge_model_uri_uses_judge_model_name():
    assert providers.judge_model_uri == f"openai:/{settings.JUDGE_MODEL_NAME}"
```

(Imports: `from src import providers`, `from src.settings import Settings, settings`.)

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL with `AttributeError` on `build_provider` and `build_judge_inference_params`, plus the `OpenAIChatModel` isinstance check.

- [ ] **Step 3: Implement the new names in `src/providers.py`**

Swap the import to `OpenAIChatModel`. Set `_provider = build_provider(settings)`, which passes both `base_url` and `api_key`. Keep the MLflow setup lines and `default_model_settings` as they are. Update the `build_model` docstring to say "Chat Completions model".

- [ ] **Step 4: Run the full suite and confirm it passes**

Run: `uv run pytest -q && grep -rn OpenAIResponsesModel src tests`
Expected: all tests pass, and grep prints nothing.

- [ ] **Step 5: Commit**

```bash
git add src/providers.py tests/test_providers.py
git commit -m "feat: use Chat Completions provider with configurable base URL"
```

---

### Task 3: Wire judges, live-test gates, and docs

**Files:**
- Create: `tests/test_judge_config.py`
- Modify: `tests/evals/happy_path_operations/evaluate.py:89-108` (3 judges)
- Modify: `tests/evals/safety_confirmation/evaluate.py:52-57`
- Modify: `tests/evals/conflict_resolution/evaluate.py:53-58`
- Modify: skip gates and comments in `tests/test_negotiator.py:9-15`, `tests/evals/safety_confirmation/test_multiturn.py:16-22`, `tests/evals/conflict_resolution/test_conflict.py:14-20`, and the docstring line in `tests/evals/safety_confirmation/multiturn_evaluate.py:10`
- Modify: `.env.example`, `README.md:12,22`, `docs/DEVELOPMENT.md:13`

**Interfaces:**
- Consumes: `providers.judge_model_uri`, `providers.judge_inference_params` (Task 2).

- [ ] **Step 1: Write the failing test in `tests/test_judge_config.py`**

```python
import pytest

from src import providers
from tests.evals.conflict_resolution import evaluate as conflict_eval
from tests.evals.happy_path_operations import evaluate as happy_eval
from tests.evals.safety_confirmation import evaluate as safety_eval

JUDGES = [
    conflict_eval.conflict_judge,
    happy_eval.groundedness_judge,
    happy_eval.correctness_judge,
    happy_eval.retrieval_recall_judge,
    safety_eval.safety_judge,
]


@pytest.mark.parametrize("judge", JUDGES, ids=lambda j: j.name)
def test_judge_uses_shared_endpoint(judge):
    assert judge.model == providers.judge_model_uri
    assert judge.inference_params == providers.judge_inference_params
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/test_judge_config.py -v`
Expected: 5 FAIL, because `inference_params` is `None`.

- [ ] **Step 3: Point every `make_judge` at the shared endpoint**

In all 5 calls, replace `model=f"openai:/{settings.JUDGE_MODEL_NAME}"` with `model=judge_model_uri, inference_params=judge_inference_params`, imported from `src.providers`. Drop any `settings` import that is no longer used (each module still uses `settings.MLFLOW_TRACKING_URI`, so most will keep it).

- [ ] **Step 4: Update the live-test skip gates**

In all three gated test files, change the condition to `not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))` and the reason to `"live model eval; run manually with LLM_API_KEY"`. Change `OPENAI_API_KEY` to `LLM_API_KEY` in the surrounding comments and in the `multiturn_evaluate.py` docstring.

- [ ] **Step 5: Update the docs**

Write `.env.example` exactly as follows:

```
# Any OpenAI-compatible service. Leave LLM_BASE_URL empty to use OpenAI.
LLM_API_KEY=sk-proj-xxxx
LLM_BASE_URL=
MODEL_NAME=gpt-4.1-mini
JUDGE_MODEL_NAME=gpt-5.2

# NVIDIA NIM example (the model must support tool calling):
# LLM_API_KEY=nvapi-xxxx
# LLM_BASE_URL=https://integrate.api.nvidia.com/v1
# MODEL_NAME=meta/llama-3.3-70b-instruct
# JUDGE_MODEL_NAME=meta/llama-3.3-70b-instruct

MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DATABASE_URL=sqlite+aiosqlite:///database.db
```

README: change the prerequisite to "An API key for OpenAI or any OpenAI-compatible service (e.g. NVIDIA NIM)". Change the setup line to tell readers to set `LLM_API_KEY`, and `LLM_BASE_URL` for non-OpenAI services, and point them to the NIM block in `.env.example`. Mention that `OPENAI_API_KEY` still works.

`docs/DEVELOPMENT.md`: replace the `OPENAI_API_KEY` bullet with bullets for `LLM_API_KEY`, `LLM_BASE_URL`, `MODEL_NAME` and `JUDGE_MODEL_NAME`. Note that the model must support tool calling, and that judges work best with JSON structured output.

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest -q && uv run ruff check src tests && grep -rn 'openai:/' tests/evals`
Expected: all tests pass (with the 5 new judge tests), 6 skipped, ruff clean, and grep prints nothing.

- [ ] **Step 7: Commit**

```bash
git add tests/test_judge_config.py tests/evals tests/test_negotiator.py .env.example README.md docs/DEVELOPMENT.md docs/superpowers/specs/2026-11-14-openai-compatible-endpoints-design.md docs/superpowers/plans/2026-11-14-openai-compatible-endpoints.md
git commit -m "feat: route MLflow judges through the shared OpenAI-compatible endpoint"
```

- [ ] **Step 8: Live smoke test (manual, needs a real NIM key)**

With the NIM block uncommented in `.env`, run `uv run pytest tests/test_negotiator.py -v`.
Expected: PASS against NIM. If it fails on tool calls, try a different NIM model rather than changing code.
