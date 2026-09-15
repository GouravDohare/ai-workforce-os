# AI Workforce OS v0.3.12

Clean rebuild of the multi-agent orchestration foundation.

## Core loop

Human objective -> CEO plan -> dependency-aware tasks -> agents -> evidence -> QA -> report -> evaluator -> honest completion/incomplete state.

## Environment

- `OPENAI_API_KEY` required for model execution.
- `OPENAI_MODEL` defaults to `gpt-5-mini`.
- `ENABLE_WEB_RESEARCH=true` enables Responses API web search.
- `DEFAULT_GOAL_BUDGET_USD=5.0` is the server-side default.
- `OPENAI_TIMEOUT_SECONDS=90`.
- `OPENAI_MAX_RETRIES=1`.

## Diagnostics

- `/health`
- `/diagnostics/generation`
- `/diagnostics/web`

Both diagnostics return the application version so deployment identity can be verified.

## Important design choices

- Web-search calls never send `reasoning_effort`.
- Incomplete Responses are treated as failures, not empty successful answers.
- No fake/demo output is emitted.
- Required dependency failures block downstream tasks.
- Process restarts mark active runs/tasks as interrupted rather than leaving them falsely running.
- Task execution UI is ASCII-safe and does not depend on browser disclosure-marker rendering.
