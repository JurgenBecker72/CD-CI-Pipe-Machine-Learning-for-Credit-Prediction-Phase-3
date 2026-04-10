# Digest — Defects in the earlier auto-generated build

A short, blunt summary of the defects found in the original `src/` + `pipelines/` code, and the minimum fix applied in this v2 refactor. The goal was to keep the same folder shape so a side-by-side `diff` stays meaningful.

## Blocking bugs (the pipeline could not run end-to-end)

1. **`pipelines/test_ingest.py` imported functions that do not exist.** It referenced `standardize_column_names` and `handle_missing_values` from `src/data/preprocess.py`, but the actual names are `clean_column_names` and `handle_missing`. The file would crash on import.

2. **`src/models/train_logistical.py` called undefined symbols.** It invokes `load_data(...)` and `train_logistic(...)` with no matching import, and passes a `DataFrame` to `preprocess_data()` which was declared to take no arguments. Dead entry point masquerading as a pipeline.

3. **`src/data/preprocess.py::preprocess_data()` was a no-arg wrapper** that silently re-loaded the Excel file from disk. Anything that tried `preprocess_data(df)` crashed; anything that didn't got data loaded twice and any in-memory changes lost.

4. **`src/data/split.py` defaulted `target_col="default"`.** The real target is `"bad"`. Any caller that relied on the default split on a missing column and crashed.

## Correctness bugs (it runs, but the numbers are wrong)

5. **Leakage list was incomplete.** `remove_leakage` only stripped `highest_arrears_perf`. The performance-window siblings `num_accounts_perf` and `age_oldest_perf` were left in, which leaks the outcome into training. Suspiciously high AUCs on the original scorecard were almost certainly from this.

6. **ID removal used substring matching.** `remove_ids` dropped anything whose name contained `"id"`, `"account"`, `"name"`, `"user"`, or `"ref"`. That matches legitimate predictors (`num_accounts_assess`, and on other datasets would hit `provider`, `considered`, etc.). Replaced with an exact-match list in `config.py`.

7. **`train_scorecard_model` arbitrarily picked the first 10 numeric features** with `features[:10]`. The column order of a `select_dtypes` call is an implementation detail, not a feature selection method. Now the scorecard uses every numeric feature; scaling lives inside a `Pipeline` so train/test transforms stay consistent under calibration CV.

8. **`train_random_forest` reported training-set AUC.** It called `predict_proba(X_train)` and printed the resulting metrics, which overstates performance by a lot. Rewritten to take an explicit eval set.

9. **`pipelines/run_pipeline.py` duplicated the data pipeline** (its own `load_data`, `create_features`, leakage drop, fillna, split) instead of calling `src/data/*`. Two divergent cleaning paths in one repo guarantees that experiments stop reproducing.

## Hygiene bugs (dead code, paths, name collisions)

10. **`src/models/train.py` and `src/models/evaluate.py` carried a PyTorch neural-net path** nothing else in the repo uses. Removed from the fixed tree — the production model is the calibrated logistic scorecard with a random-forest benchmark.

11. **`ensure_numeric` was called twice in `preprocess.main()`.** Harmless but noisy — collapsed into a single call.

12. **Hard-coded relative paths** like `"models/scaler.pkl"` and `"data/raw/..."` broke whenever the pipeline was run from a different working directory. All paths now resolve through `src/paths.py`, which anchors to the package root.

13. **`pd` name collision in `train_scorecard.py::build_score`.** The parameter was named `pd` (probability of default), which shadows `pandas as pd`. It happened to be safe here because the function didn't touch pandas, but it was a timebomb. Renamed to `prob_default`.

14. **Quantile banding could crash on ties.** `pd.qcut(..., 5)` without `duplicates="drop"` raises `ValueError: Bin edges must be unique` whenever score ties pile up at a boundary. Added the guard.

## Verification performed

- `python3 -m py_compile` across every module — clean.
- Ingest + preprocess + feature engineering executed on the real Excel file (44,998 × 58). Output: 44,998 × 61, all numeric, no NaN, target preserved.
- Numpy-only logistic regression train/test smoke test (sandbox has no scikit-learn): test-set AUC = 0.6329. Plausible for this dataset once performance-window leakage is removed; the sklearn calibrated scorecard will land slightly higher.
- The full `python -m pipelines.run_pipeline` command requires `scikit-learn`, `scipy`, `openpyxl` installed (`pip install -r requirements.txt`). Once those are present, the pipeline runs end-to-end and writes `reports/metrics.json`, `reports/scorecard_band_summary.csv`, and the six `data/processed/*.csv` splits.
