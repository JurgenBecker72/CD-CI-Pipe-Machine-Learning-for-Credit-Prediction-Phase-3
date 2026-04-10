# Second-Eyes Review — Digest of Issues Fixed

Project: Machine Learning for Credit Prediction (Phase 3)
Review date: April 2026
Reviewer: Internal second-eyes review of the initial ChatGPT-drafted pipeline

This document records the issues found during a structured second pass over
the pipeline, what was wrong, why it mattered, and how it was fixed. It is
intended as an honest engineering log — the kind of thing you would attach to
a model risk sign-off package.

---

## 1. Critical: ID column not being removed before training

**Where:** `src/data/preprocess.py`, function `remove_ids`

**What was wrong:** the original code dropped any column whose name *contained*
the substring `"id"`. On this dataset that matched `dummy_id` (correct) but it
would also have caught unrelated legitimate features in a future dataset, and
more importantly it made the behaviour fragile and invisible.

**Why it mattered:** a literal ID column leaked into the feature matrix means
the model can memorise rows. In production, fresh IDs won't match anything
seen in training, and performance will collapse without warning.

**Fix:** introduced an explicit `ID_COLUMNS` list in `src/config.py` and
rewrote `remove_ids` to do exact-name matching only. Same treatment applied
to `pipelines/run_pipeline.py`.

---

## 2. Critical: Target leakage from performance-window features

**Where:** `src/data/preprocess.py`, `pipelines/run_pipeline.py`

**What was wrong:** the dataset contains three columns measured *during* the
performance window — `num_accounts_perf`, `highest_arrears_perf`,
`age_oldest_perf`. These describe what happened *after* the point at which a
credit decision would actually be made. They were being passed into the model
as features.

**Why it mattered:** this is textbook target leakage. The model would have
reported artificially high AUC/KS during development and would have failed
catastrophically when deployed against applicants who, by definition, have no
performance-window data yet.

**Fix:** added a `LEAKAGE_COLUMNS` list to `config.py`, added a dedicated
`remove_leakage` step to preprocessing, and made sure both the scorecard and
the random-forest benchmark drop these columns before any fitting. Assessment-
window features (`*_assess`) are explicitly kept — they are observable at
decision time and are legitimate.

---

## 3. Critical: Imputation fit on the full dataset before split

**Where:** `src/data/preprocess.py`, function `handle_missing`, and
`pipelines/run_pipeline.py`

**What was wrong:** median imputation and scaling were being applied to the
entire frame *before* the train/validation/test split. The medians and
scaling parameters therefore incorporated information from validation and
test rows.

**Why it mattered:** this is a subtle but real form of information leakage
that inflates held-out performance and undermines the whole point of a
validation set. It is a common ChatGPT / tutorial mistake.

**Fix:** reordered the pipeline so the split happens first. Median imputation
is now fit on `X_train` only and those same medians are applied to
`X_val` and `X_test`. StandardScaler gets the same treatment — `fit` on
train, `transform` on val and test. A helper `impute_train_test` was added
to `run_pipeline.py` to make the pattern explicit.

---

## 4. Random-forest one-hot encoding fit on the wrong frame

**Where:** `pipelines/run_pipeline.py`, random-forest benchmark section

**What was wrong:** `pd.get_dummies` was being called on the combined frame,
then re-split. If a rare category appears only in training or only in test,
this works by accident — but it violates fit-on-train discipline and will
break the first time a new category appears in production.

**Fix:** `get_dummies` now runs on `X_train` only, and `X_test` is reindexed
against the training columns with `fill_value=0`. Any unseen category in test
becomes an all-zeros row, which is the correct production behaviour.

---

## 5. Dual-pipeline drift between `src/` and `pipelines/`

**Where:** across the repo

**What was wrong:** there were effectively two pipelines — a modular one
under `src/` and a monolithic one under `pipelines/run_pipeline.py`. They had
diverged, so a fix applied in one place didn't propagate to the other.

**Why it mattered:** model risk auditors will ask *which* pipeline produced
the model artefacts in production. Having two answers is a compliance problem.

**Fix:** applied every leakage / imputation / ID fix to both pipelines so they
are consistent. Longer-term, `pipelines/run_pipeline.py` should be rewritten
as a thin orchestrator that calls functions from `src/` — this is noted as
follow-up work in the README.

---

## 6. Legacy ChatGPT / Claude references scattered through the codebase

**Where:** folder names, README files, internal docstrings in archived code

**What was wrong:** two earlier iterations of the project were saved in
folders named `Claude/` and `Claude fix OpenAI code/`. Several files had
byline strings identifying the model that produced them.

**Why it mattered:** the repo is going to a portfolio / potential customers.
Identity references to specific AI tools inside a "finished" deliverable
look unprofessional and distract from the actual engineering.

**Fix:** renamed folders to `archive/v1_initial/` and `archive/v2_refactor/`.
Scrubbed identity strings from README, DIGEST, and script headers inside
those folders. The current working tree under `src/` and `pipelines/` was
already clean. The archive is kept (not deleted) so the iteration history is
still auditable.

---

## 7. Stale `.pyc` files tracked in git

**Where:** `pipelines/__pycache__/*.pyc` — three files

**What was wrong:** these had been committed before `.gitignore` existed, so
they were still being tracked even after the ignore rules were added.

**Fix:** removed them from the index with `git rm --cached` while keeping the
files on disk. The `.gitignore` pattern `__pycache__/` now does its job.

---

## 8. `.gitignore` did not cover archive data / models / reports

**Where:** `.gitignore`

**What was wrong:** the ignore rules covered `data/raw/*`, `models/*`, and
`reports/*` at the top level, but the `archive/v2_refactor/` folder contained
its own copies of raw Excel, processed CSVs, and HTML / JSON reports that
would have been committed on the first `git add .`.

**Why it mattered:** the raw dataset contains simulated credit data that we
do not want in a public GitHub repo, and committed binary artefacts bloat the
repo unnecessarily.

**Fix:** extended `.gitignore` with `archive/**/data/raw/*`,
`archive/**/data/processed/*`, `archive/**/models/*`, and the corresponding
report globs. A dry-run `git add -n .` now shows only source code, config,
README, and CI files — no data, models, or reports.

---

## 9. Misspelled package init files

**Where:** `src/_init_.py`, `src/data/_init_.py`, `src/features/_init_.py`,
`src/models/_init_.py`

**What was wrong:** all four package `__init__.py` files were named with
single underscores (`_init_.py`) instead of double (`__init__.py`). This
meant Python did not actually treat these directories as packages, and any
`from src.data import ...` style import only worked by accident because of
the way scripts were being executed.

**Fix:** replaced all four with properly named `__init__.py` files. Pipeline
now runs with the recommended `python -m pipelines.run_pipeline` invocation
from the project root, which was previously broken.

---

## 10. No CI, no tests, no pinned dependencies

**Where:** repo root

**What was wrong:** there was no `requirements.txt`, no test suite, no
GitHub Actions workflow. Any claim that the project is "production-ready"
would have been hard to defend.

**Fix:**

- Added `requirements.txt` with pinned versions of pandas, scikit-learn,
  scipy, openpyxl, joblib, matplotlib, seaborn, pytest, pytest-cov, ruff,
  and black.
- Added `.github/workflows/ci.yml` with two jobs: `lint-and-test` (runs ruff,
  black --check, and pytest across Python 3.10 / 3.11 / 3.12) and
  `smoke-run` (generates a 500-row fixture dataset and runs the full
  pipeline end to end, uploading the reports folder as a build artefact).
- Added `tests/test_config.py` locking in the invariants from issues 1, 2,
  and 3 above, so the fixes cannot silently regress.

---

## Summary

Of the ten issues, three are genuinely critical (ID leakage, performance-
window leakage, pre-split imputation) and would have made any model produced
by this pipeline unsafe to deploy. The other seven are engineering hygiene
issues that matter for maintainability and audit, but not for model validity.

All ten are now fixed on the current branch and the three most critical ones
are locked in by unit tests.

The next steps are documented in `README.md` under the production roadmap
section: merge the two pipelines into one, add a smoke test dataset fixture
to the repo, and wire up the CI workflow against the real GitHub branch.
