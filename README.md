# AI Workforce OS v0.4.0

Major orchestration/reliability foundation for the AI Workforce OS.

Implemented: structured CEO planning; dynamic specialist workforce; machine-readable task contracts; dependency DAG and blocking; classified retries; evidence capture; evidence-linked confidence; handoffs; artifacts/version hashes; artifact dependency records; company/goal/unknown/contradiction memory; web-search tool boundary; configurable model routing; atomic budget reservation/settlement; model/task attempt telemetry; evaluator + targeted replan hooks; explicit incomplete/interrupted/cancelled states; cancellation/retry; live polling UI; isolated API diagnostics; private-beta token; T1-T15 benchmark registry.

v0.4 remains experimental: SQLite and a process-local executor are intentionally retained. Production hardening is v0.5: Postgres, durable queue/worker, full authorization/multi-tenancy, object storage, side-effect tool permissions and billing.

Deployment validation:
1. Deploy.
2. `/health?expected_version=0.4.0` must show `version_match:true`.
3. `/diagnostics/generation` must succeed.
4. `/diagnostics/web` must succeed.
5. Run the same industrial-automation objective and inspect tasks, evidence, evaluator and spend.
