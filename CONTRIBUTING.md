# Contributing to VibeQuant

VibeQuant is an intent-driven quant research workbench built on top of the
[akquant](https://github.com/akfamily/akquant) engine (FastAPI backend, `vq`
CLI, bilingual web UI). This guide covers dev setup, the project's hard
architectural boundary, the Developer Certificate of Origin (DCO) sign-off,
and quickstarts for the most common contributions: a new strategy template
or a new data source.

For bug reports and feature requests, use the GitHub issue templates.

## Developer Certificate of Origin (DCO)

Every commit in a community pull request MUST carry a `Signed-off-by:`
trailer. We do not require a CLA — the DCO is a lightweight per-commit
attestation that you wrote the code or have the right to submit it under
the project's MIT license.

Sign your commits with `-s`:

```bash
git commit -s -m "feat(strategies): add bollinger breakout template"
```

This appends a trailer like:

```
Signed-off-by: Your Name <you@example.com>
```

PRs without a `Signed-off-by:` on every commit will be asked to rebase and
resign. To fix an unsigned series, run
`git rebase --signoff <base-branch>` and force-push the branch.

### DCO 1.1 (full text)

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Dev Setup

```bash
git clone https://github.com/transcend-0/VibeQuant.git
cd VibeQuant
pip install -e ".[dev]"

pytest tests/          # run the test suite
vq ui                  # web UI at http://127.0.0.1:8321
```

## The Hard Boundary: akquant Stays Untouched

VibeQuant integrates with `akquant` purely as a dependency. **Do not modify
akquant itself, and do not import it from anywhere outside
`src/adapters/`** (`akquant_engine.py`, `akquant_factor.py`). All new code
should go through these adapters or sit above them. PRs that add akquant
imports elsewhere, or that patch/vendor akquant source, will be asked to
route through the adapter layer instead.

The same separation applies to the LLM: the **research intelligence** layer
(idea extraction, revision proposals, agent self-iteration) is LLM-driven,
but the **execution core** — DSL, planner, backtest/factor engines,
statistical validation, report generation — must stay deterministic and never
call an LLM. A validator that depends on a sampling model can't serve as a
referee. If your change touches `src/dsl.py`, `src/planner.py`,
`src/runner.py`, `src/factors/validation.py`, or the akquant adapters, it
should not introduce any LLM call.

## Adding a New Strategy Template

1. Create a module under `src/strategies/` (see `ma_cross.py` for the
   smallest example). Define a `build(params) -> signal_fn` factory and
   `register(...)` it with a `StrategyTemplate` (`name`, bilingual
   `summary_en` / `summary_zh`, `defaults`).
2. `signal_fn(closes, position)` should return `None` for "no change" or a
   target position fraction; keep it pure and side-effect free.
3. Add a demo task YAML under `tasks/` and confirm it runs:
   ```bash
   vq run tasks/<your_demo>.yaml
   ```
4. Add a test under `tests/` (see `test_pipeline.py` / `test_rotation.py`
   for patterns).

## Adding a New Data Source / Market

1. Add the fetch/fallback logic to `src/data_sources/market.py` (or a new
   module under `src/data_sources/` for a distinct asset kind), following
   the existing fallback-chain pattern ordered by IP-ban risk (see
   `README.md#-data-sources--fallback`).
2. Cache raw bars under `data/raw/<kind>/`; respect the existing throttle —
   some free endpoints (e.g. eastmoney) temp-ban bursting IPs.
3. Wire the new market/universe into the `MARKETS` registry
   (`webui/server.py`) if it should appear in the UI.
4. Add a test that exercises the loader against cached/fixture data (avoid
   tests that require live network access).

## Code Style

- Type-annotate public function and method signatures.
- Keep docstrings focused on *why*, not *what* — the code should already say
  what it does.
- No hardcoded paths, secrets, or URLs — config via `config/*.yaml` or
  module-level constants.
- Delete unused code rather than commenting it out.

## Attribution

Do NOT add `Co-Authored-By:` trailers or AI-assistant attribution lines to
commit messages or PR descriptions. The DCO sign-off is the only required
trailer; keep commit metadata clean.

By contributing, you agree that your contributions are licensed under the
project's MIT license (see `LICENSE`).
