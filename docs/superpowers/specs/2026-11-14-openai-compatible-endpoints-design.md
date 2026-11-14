# OpenAI-compatible endpoints (NVIDIA NIM and others) — design

Date: 2026-11-14
Size: small change

## Intent

Let the appointment agents (main agent, conflict negotiator), the eval dataset
generators, and the MLflow eval judges run on any OpenAI-compatible service —
NVIDIA NIM first (`https://integrate.api.nvidia.com/v1`, `nvapi-…` keys,
namespaced model IDs like `meta/llama-3.3-70b-instruct`) — by editing `.env`
only. No code changes to switch providers.

## Goals

- One base URL + one API key in `.env`, used by every LLM call in the project.
- `MODEL_NAME` / `JUDGE_MODEL_NAME` accept any model ID the service exposes.
- Leaving `LLM_BASE_URL` empty keeps talking to OpenAI as today.

## Non-goals

- Separate endpoints/keys for the judge vs. the agent (option B — declined).
- Native non-OpenAI APIs (Anthropic, Gemini SDKs).
- Detecting whether a chosen model supports tool calling / JSON output.

## Anti-goals

- Breaking an existing `.env` that only sets `OPENAI_API_KEY`.

## Approaches considered

1. **Always Chat Completions (picked).** Replace `OpenAIResponsesModel` with
   `OpenAIChatModel` everywhere. One code path that every compatible service
   speaks; the agent only uses `temperature`, so nothing Responses-specific is
   lost. Eval scores across providers come from the same agent code.
2. Auto-detect: Responses for OpenAI, Chat elsewhere. Keeps OpenAI behavior
   identical but adds two code paths and an implicit rule that misfires behind
   proxies.
3. Explicit `LLM_API=chat|responses` switch. No guessing, but one more knob to
   get wrong (wrong value against NIM → opaque 404).

## Design

**Settings (`src/settings.py`)**
- `LLM_API_KEY: str` — required. Also accepted under the old name
  `OPENAI_API_KEY` (pydantic `AliasChoices`), so existing `.env` files keep
  working. *(my call)*
- `LLM_BASE_URL: str | None = None` — `None` means OpenAI's default endpoint.
- `MODEL_NAME`, `JUDGE_MODEL_NAME`, `MODEL_TEMPERATURE` unchanged.

**Agent models (`src/providers.py`)**
- `OpenAIProvider(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)`.
- `build_model()` returns `OpenAIChatModel`; `agent_model` follows. Agent,
  negotiator, and both generators already go through `build_model` /
  `agent_model`, so they need no changes.

**Judges (`tests/evals/*/evaluate.py`, 5 `make_judge` calls in 3 files)**
- `src/providers.py` exposes `judge_model_uri` (`f"openai:/{JUDGE_MODEL_NAME}"`)
  and `judge_inference_params` (`{"api_key": ...}` plus `"api_base": ...` only
  when `LLM_BASE_URL` is set). Each `make_judge` passes
  `model=judge_model_uri, inference_params=judge_inference_params`. MLflow 3.8.1
  forwards these straight into `litellm.completion`. Verified: `openai:/meta/x`
  parses to provider `openai`, model `meta/x`.

**Live-test skip gates** (`test_negotiator.py`, `test_multiturn.py`) check for
`LLM_API_KEY` or `OPENAI_API_KEY`.

**Docs**: `.env.example` shows the OpenAI default with a commented NIM block;
README and `docs/DEVELOPMENT.md` describe the variables and the
"model must support tool calling" requirement.

## Known risks

- The chosen model must support tool calling (agent) and ideally JSON
  structured output (judges). MLflow retries judges without structured output
  when a model rejects it; the agent has no such fallback.
- Some reasoning models reject `temperature`; unchanged from today.

## Left to the builder

- Test layout for the new settings/provider behavior.
- A manual live smoke run against NIM to confirm end to end (needs a real key).
