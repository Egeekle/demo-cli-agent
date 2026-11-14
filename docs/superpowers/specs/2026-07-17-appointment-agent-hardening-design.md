# Appointment Agent Hardening — Design Spec

**Date:** 2026-07-17
**Status:** Approved (pending final user review)

## 1. Goal & Context

The CLI appointment agent (PydanticAI + SQLModel + MLflow) works for simple flows but has
structural gaps: its flagship conflict-resolution feature cannot actually detect time
conflicts, an `update_appointment` capability is referenced everywhere but does not exist,
destructive-action safety is enforced only by prompt prose, the DB layer is synchronous
inside async tools, and the eval suite has bias and coverage holes.

This spec covers a single, sequenced remediation across six workstreams. Scope is the whole
`demo-cli-agent` repo. The agent model stays `gpt-4.1-mini`.

### Confirmed design decisions

- **Safety gate:** structural tool approval via PydanticAI `requires_approval=True` +
  `DeferredToolRequests` / `DeferredToolResults` (verified available in installed
  `pydantic_ai==1.47.0`).
- **Eval judge model:** `gpt-5.2` (OpenAI), configurable via settings; separate from the agent.
- **Datetime:** injectable clock via `Deps.now`; datetimes stay naive local time (no timezone
  conversion). Kills brittle `patch("src.agent.datetime")` test hooks.
- **Async DB:** full async layer (`sqlite+aiosqlite`, SQLModel `AsyncSession`), session scoped
  per user turn.
- **Overlap policy:** `create_appointment` (and time-changing `update_appointment`) **always
  refuse** to persist an overlapping appointment via `ModelRetry`. No `force` override
  (YAGNI — can be added later if a real need appears). The "no double-booking" invariant
  lives entirely in code.

### Explicitly out of scope (YAGNI)

- Timezone-aware datetimes (store-UTC/convert-on-display).
- Switching the agent itself to a non-OpenAI provider.
- Postgres / connection-pool tuning beyond sensible aiosqlite defaults.

## 2. Workstreams

Ordered by dependency. Later workstreams assume earlier ones are in place.

### WS1 — Foundation: config & idiom cleanup

- **`src/providers.py`** (new): single place that builds the OpenAI provider + model and calls
  `mlflow.pydantic_ai.autolog()` **exactly once**. Exposes a `build_model(model_name)` helper
  plus the shared agent model. `agent.py`, `negotiator.py`, and eval/generator modules import
  from here instead of each re-initializing (autolog currently runs 3×).
- **`src/settings.py`**: add `JUDGE_MODEL_NAME: str = "gpt-5.2"` and
  `MODEL_TEMPERATURE: float = 0.0`. Remove the stray `# typing: ignore` noise.
- **`instructions=`** replaces `system_prompt=` on all agents (matches the generator, which
  already uses `instructions`). Dynamic "today" instruction reads from `ctx.deps.now()`.
- **`Deps`** (in `src/tools.py`):
  - `session: AsyncSession` (see WS6).
  - `negotiator_agent: Agent[None, ConflictResolution]` — typed and **required** (no `None`
    default / runtime `ValueError`).
  - `now: Callable[[], datetime]` — defaults to `datetime.now`; injected as a fixed clock in
    tests/evals.
- **Model settings**: apply `MODEL_TEMPERATURE` via `model_settings` for determinism.
- **Agent `retries`**: set a small retry budget (e.g. `retries=2`) so `ModelRetry` from tools
  has room to self-correct.

### WS2 — Conflict engine (core correctness fix)

- **`check_availability(ctx, start, end) -> list[Appointment]`** (new, read-only tool): SQL
  overlap predicate `Appointment.start_time < end AND Appointment.end_time > start`. Returns
  real rows. This is the deterministic ground truth for conflicts. (Text-based
  `search_appointments` remains, for keyword lookup only.)
- **`resolve_scheduling_conflicts` reworked**: signature becomes
  `(ctx, user_request: str, proposed_start: datetime, proposed_end: datetime)`. It calls
  `check_availability` itself to fetch the real overlapping `Appointment` rows and passes
  **structured appointment data** (not LLM-stringified summaries) to the negotiator. Ends the
  telephone game where the main LLM hand-summarized appointments into strings.
- **Negotiator** (`src/negotiator.py`): input prompt/context carries structured appointment
  fields (title, start, end, location). Output stays `ConflictResolution`
  (`ProposedSlot` list). Uses shared model from WS1 and `instructions=`.
- **Overlap backstop in writes**: `create_appointment` re-runs the overlap check on write; if
  an overlap exists it raises `ModelRetry("This slot conflicts with <appt>; propose an
  alternative or resolve the conflict")`. Never persists an overlap. The conflict guarantee is
  in code, not the prompt.
- **Agent instructions** updated: for any create/reschedule, call `check_availability` first;
  on conflict call `resolve_scheduling_conflicts` and present alternatives in the existing
  markdown table (`Time Slot`, `Reasoning`).

### WS3 — Update capability & input validation

- **`update_appointment(ctx, appointment_id, title?, start_time?, end_time?, location?,
  description?) -> Appointment`** (new): partial update (only provided fields change).
  `requires_approval=True` (WS4). If times change, run the WS2 overlap backstop (excluding the
  appointment being updated from its own conflict check). Returns the updated row; raises
  `ModelRetry` if the id does not exist.
- **Input validation** via `ModelRetry` on create/update: reject `end_time <= start_time` and
  `start_time` earlier than `ctx.deps.now()`.
- Register `update_appointment` on the agent; update agent instructions to advertise
  update/reschedule. This closes the gap where the prompt and the entire eval suite reference
  `update_appointment` but no tool existed.

### WS4 — Structural safety gate

- Mark `create_appointment`, `delete_appointment`, `update_appointment` with
  `requires_approval=True`. The write physically cannot execute until an approval is returned —
  enforced by PydanticAI, not prose.
- **`src/approvals.py`** (new): small helper encapsulating the deferred round-trip:
  - Given an `agent.run(...)` result whose `output` is a `DeferredToolRequests`, render each
    pending approval's tool name + args into a human summary.
  - Build `DeferredToolResults` from a decision (`ToolApproved` / `ToolDenied`).
  Used by both the CLI and the eval harness so they exercise the identical path.
- **`src/main.py` chat loop reworked**: when a run returns `DeferredToolRequests`, show the
  summary, ask the user (`yes` → `ToolApproved`; `no` → `ToolDenied`), then re-run with
  `deferred_tool_results=...` and `message_history`. Loop until the run returns a normal string
  output. A fresh async session (WS6) is opened for the turn that performs the approved write.
- Agent instructions keep the "summarize the action before doing it" prose (good UX) — now
  backed by a real gate rather than being the only line of defense.

### WS5 — Evaluation improvements

- **Judge model**: all `make_judge(...)` calls use `settings.JUDGE_MODEL_NAME` (gpt-5.2)
  instead of the agent model. Removes same-family self-eval bias.
- **Restore deterministic `retrieval_recall` code scorer as the primary** retrieval metric
  (exact, free, reads TOOL spans). Keep the LLM `retrieval_recall_judge` as a secondary/
  complementary signal rather than the replacement.
- **New multi-turn safety eval**: drives the real approval flow — turn 1 "delete X" must ask
  for confirmation and NOT write; turn 2 "yes" must execute; a separate case where turn 2 "no"
  must abort with no write. Assert against DB state, not just text. Uses `src/approvals.py`.
- **New conflict-resolution eval**: propose a slot overlapping a seeded appointment; assert the
  agent calls `check_availability`, detects the conflict, and offers alternatives grounded in
  real appointments (no hallucinated slots that themselves conflict).
- **Generator/seed updates**: inject the fixed clock via `Deps.now` instead of
  `patch(... datetime ...)`. `update_appointment` references in the generator now map to a real
  tool.
- **Bounded history**: add a lightweight `history_processors` trimmer (keep last N
  messages) so token growth is bounded. Minimal — no summarization model.

### WS6 — Async database layer

- **`pyproject.toml`**: add `aiosqlite` dependency.
- **`src/database.py`** rewritten async:
  - `create_async_engine("sqlite+aiosqlite:///...")` from `sqlalchemy.ext.asyncio`.
  - `async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)` where
    `AsyncSession` is imported from **`sqlmodel.ext.asyncio.session`**. `expire_on_commit=False`
    is required so `Appointment` objects returned by tools can be serialized after commit
    without triggering a lazy reload outside the async context.
  - `get_session()` becomes an `@asynccontextmanager` yielding an `AsyncSession`.
  - `create_db_and_tables()` becomes async:
    `async with engine.begin() as conn: await conn.run_sync(SQLModel.metadata.create_all)`.
- **All tools become `async def`** and await DB ops: `await session.exec(stmt)` then `.all()`,
  `await session.get(...)`, `await session.commit()`, `await session.refresh()`,
  `await session.delete()`.
- **Session lifecycle = per user turn** (per `agent.run`), replacing today's single
  session-for-the-whole-chat. The CLI opens a fresh session each turn and injects via `Deps`;
  this aligns the CLI with the eval harness and composes correctly with the approval
  round-trip (read-only turn 1 and the writing turn 2 use separate sessions). The negotiator
  subagent reuses the same turn's session via `ctx.deps.session`.
- **Eval harness** (`evaluate.py` × 2, generator): open async sessions; await
  `create_db_and_tables`; `predict_fn` stays async.

## 3. Data Flow (create-with-conflict, end to end)

```
User: "Book a sync tomorrow 2pm"
  -> agent.run(turn 1, session A)
     -> check_availability(2pm-3pm)                 [read, session A]
        -> overlaps ["Doctor's Appointment"]
     -> resolve_scheduling_conflicts(request, 2pm, 3pm)
        -> check_availability internally -> structured rows
        -> negotiator.run(structured data) -> ConflictResolution(alternatives)
     -> agent asks user (markdown table of alternatives)
User: "Take 4pm then"
  -> agent.run(turn 2, session B)
     -> wants create_appointment(4pm-5pm)  [requires_approval]
     -> returns DeferredToolRequests
  -> CLI renders summary, asks "Shall I proceed?"
User: "yes"
  -> agent.run(turn 3, session C, deferred_tool_results=approved)
     -> create_appointment(4pm-5pm)  [overlap backstop passes] [write, session C]
     -> returns confirmation string
```

## 4. Testing Strategy

- **TDD** for the unit layer: `check_availability`, `update_appointment`, overlap backstop,
  input validation, and the `approvals.py` helper each get failing tests first (in-memory
  `sqlite+aiosqlite`, injected fixed clock), then implementation.
- Update existing `tests/test_tools_read.py` / `test_tools_write.py` / `test_database.py` /
  `test_settings.py` / `test_agent_structure.py` for async + new `Deps` shape.
- **Evals** are the outer loop, run after the unit layer is green: happy-path, safety
  (multi-turn), and conflict-resolution.

## 5. Files Touched (summary)

- New: `src/providers.py`, `src/approvals.py`,
  `tests/evals/conflict_resolution/*` (generator/evaluate/tests),
  new safety multi-turn eval cases.
- Modified: `src/settings.py`, `src/database.py`, `src/tools.py`, `src/agent.py`,
  `src/negotiator.py`, `src/main.py`, `pyproject.toml`, both existing eval modules,
  existing unit tests, `docs/ARCHITECTURE.md` (reflect structural safety + async DB).

## 6. Risks & Mitigations

- **Approval round-trip API shape**: confirmed `requires_approval`, `DeferredToolRequests`,
  `DeferredToolResults`, `ToolApproved`, `ToolDenied` exist in `pydantic_ai==1.47.0`. Plan step
  will confirm exact `agent.run(deferred_tool_results=...)` parameter name against installed
  source before wiring the loop.
- **`expire_on_commit` footgun**: mitigated by setting it `False` on the sessionmaker.
- **gpt-5.2 judge availability**: model name lives in settings; if unavailable the fallback is
  a one-line settings change.
- **Eval cost/time** with a stronger judge: acceptable for a demo; runs are opt-in.
