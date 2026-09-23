# Contributing

## Workflow

1. Create or claim a GitHub issue with an acceptance checklist.
2. Branch from `main` using `feature/<issue>-<short-name>`, `fix/<issue>-<short-name>`, or `docs/<issue>-<short-name>`.
3. Keep commits focused and use messages such as `feat(data): add grouped split validation`.
4. Run `python -m pytest` and `ruff check .` before opening a pull request.
5. Open a pull request, link the issue, attach evidence, and request review from a teammate.
6. Merge only after CI passes and at least one teammate reviews the change.

## Evidence required for experiment changes

A training or evaluation pull request should identify:

- configuration file and random seed;
- dataset version and split-manifest hash;
- Git commit SHA used for the run;
- hardware/runtime details;
- paths to metrics, figures, and checkpoint checksums;
- what changed relative to the prior run.

Do not commit datasets, checkpoints, credentials, notebook caches, or arbitrary generated output. Add only reviewed report-ready artifacts under `reports/`.

## Notebook policy

Notebooks should explain and call functions from `src/`; they must not contain a second independent implementation of training or evaluation. Clear unnecessary cell output before committing. A teammate should be able to reproduce the notebook from a clean environment.

## Team accountability

All members are expected to contribute meaningful work weekly. Use issues, pull requests, reviews, tests, documentation, and experiment records as genuine evidence. Do not split work into artificial commits or backdate activity.

