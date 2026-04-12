# Project Instructions

These instructions are intended to be durable project-wide maintenance rules.

## Notebook / Script Parity

- The runnable scripts and the notebooks must stay behaviorally aligned.
- If a change is made to an operational script, update the corresponding notebook cells in the same task.
- If a change is made to a notebook, update the corresponding operational script in the same task.
- This parity rule applies to:
  - runtime configuration defaults
  - CLI-exposed behavior that mirrors notebook parameters
  - endpoint selection and request construction
  - retry, backoff, throttling, and loop-stopping behavior
  - progress monitoring and operator-facing inspection workflows
  - stage logic, limits, and output semantics
- Do not intentionally let the notebook and script implementations drift unless the divergence is explicitly documented and approved.

## Implementation Preference

- Prefer shared Python code for core crawler behavior where possible.
- If notebook and script duplication is unavoidable, treat notebook parity updates as part of the same change, not a later cleanup.

## Runtime Configuration

- Prefer environment-backed configuration for durable runtime knobs that operators may need to change in notebooks, scripts, or cluster jobs.
- If both env vars and CLI flags are supported for the same setting, document the precedence explicitly and keep it consistent across entrypoints.
- Keep `.env.example` aligned with env-backed knobs that are intended for operators.
- Defaults should remain conservative and cluster-safe unless there is a strong reason to optimize for local speed instead.

## Cluster / Operator Experience

- Long-running crawler output should be observable from stdout so Slurm and similar batch systems can be monitored with a single primary log stream.
- If a notebook is useful for operational monitoring, provide a script equivalent that can run on compute nodes without Jupyter.
- Operator-facing scripts should prefer clear summaries and stable text output over notebook-only presentation features.

## Stage Semantics

- Flags that cap crawl scope should have clearly defined resume semantics.
- If a scope-limiting flag is intended as a total cap across reruns, preserve that meaning consistently across stages and entrypoints.
- Avoid silent behavior changes in stage semantics; update docs when stage scope, resume behavior, or output meaning changes.

## Retry / Pagination Safety

- Retry, backoff, throttling, and pagination behavior are core pipeline semantics and must stay aligned across notebooks and scripts.
- Cursor-based pagination must have explicit termination guards for no-yield or cyclic progress, not just transport-level cursor movement.
- Avoid request parameters that are known to cause endpoint instability unless their use is explicitly justified and documented.

## Documentation Sync

- Update `README.md` when operator-visible behavior changes.
- Update `PLAN.md` when implementation behavior or stage semantics change.
- Treat documentation updates as part of the same task as the code change, especially for runtime defaults, endpoints, retry policy, scope limits, and monitoring workflows.
