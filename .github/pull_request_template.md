## Summary

<!-- What does this PR do? 1-3 bullet points. -->

-

## Why

<!-- What problem does it solve? Link related issues with "Closes #123". -->

## Changes

<!-- List key changes. For new factors/data sources/templates, describe what they cover. -->

-

## Test Plan

- [ ] Existing tests pass (`pytest tests/`)
- [ ] New tests added (if applicable)
- [ ] Tested manually (describe below)

## Checklist

- [ ] No changes to the `akquant` engine itself — integration goes only through `src/adapters/`
- [ ] No hardcoded values (API keys, file paths, magic numbers)
- [ ] No LLM calls introduced in the deterministic execution core (DSL, planner, backtest/factor engines, validation, reports)
- [ ] Documentation updated (if user-facing change)
