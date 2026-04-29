# Strands Agentic App (Actuarial MVP)

Python app that demonstrates an agentic architecture for actuarial workflows:
- Orchestrator agent with child agents
- Lifecycle tracking for each agent
- Streamlit UI for interaction and run traceability
- Chart output (triangle heatmap, development factors, reserve by AY)
- Custom tools with deterministic Chain-Ladder actuarial behavior

## Quick start

1. Create virtual environment and activate it.
2. Install dependencies:

```bash
pip install -e ".[dev]"
```

3. Copy environment variables:

```bash
cp .env.example .env
```

4. Run the app (stable mode):

```bash
./scripts/run_app.sh
```

5. Run tests:

```bash
pytest
```

## Project structure

- `src/app/main.py`: Streamlit UI
- `src/agents/orchestrator.py`: Orchestrator workflow
- `src/agents/children.py`: Child agents (intake, data prep, method select, calc, explain)
- `src/core/models.py`: Typed request/result/lifecycle models
- `src/core/lifecycle.py`: Lifecycle event helpers
- `src/tools/validation.py`: Input validation tool
- `src/tools/reserving.py`: Deterministic Chain-Ladder reserving calculator
- `src/tools/charting.py`: Plotly chart builders
- `tests/`: Lifecycle and tool tests

## Agent lifecycle

Each run gets a unique `run_id`. Child agents are executed by the orchestrator and emit events:
- `running`
- `completed`
- `failed`

The UI shows this lifecycle as a tabular trace and exposes the full run artifact for download.

## Stable launch profile

The app uses a project-local Streamlit runtime directory to avoid permission issues:
- `STREAMLIT_HOME=.streamlit_runtime`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`
- `STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll`

If you previously saw:
- `PermissionError: ... ~/.streamlit`
- `Cannot start fsevents stream`

use only `./scripts/run_app.sh` (or `run-app`) to avoid those failures.

## Current behavior (deterministic Chain-Ladder mode)

The calculation layer now solves one actuarial problem end-to-end using deterministic Chain-Ladder-style reserving:
- Real-example cumulative triangle projection
- Volume-weighted age-to-age factors (LDF)
- CDF-based ultimate projection
- IBNR / reserve by accident year and total indicated reserve
- Deterministic outputs for repeatable tests/demo

## Input and output contract

Input triangle records require:
- `accident_year` (int)
- `development_period` (int)
- `value` (numeric cumulative amount)

Output includes:
- `factors` (LDF)
- `cdf`
- `latest_by_ay`
- `ultimate_by_ay`
- `ibnr_by_ay`
- `reserve_by_ay`
- `total_reserve`
- `diagnostics`

## Phase 2: Extend deterministic Chain-Ladder to Mack

Keep external contracts stable and swap internals:

1. Preserve `compute_chain_ladder_reserving()` output schema:
   - `triangle_records`
   - `factors`
   - `reserve_by_ay`
   - `total_reserve`
   - `diagnostics`
2. Add Mack uncertainty estimates alongside current deterministic outputs.
3. Update `MethodSelectionAgent` to select between Chain-Ladder and Mack (with assumptions).
4. Add tests for numeric sanity bounds and regression fixtures.
5. Keep `src/agents/orchestrator.py` and `src/app/main.py` unchanged except for enhanced diagnostics display.

This approach isolates actuarial math evolution without breaking the UI or orchestration layer.

