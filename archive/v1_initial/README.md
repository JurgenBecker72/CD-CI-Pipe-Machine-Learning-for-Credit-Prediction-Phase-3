# Credit Risk ML Pipeline — Phase 3 (v1 Initial Build)

> A production-grade credit scoring pipeline combining **DRA psychometric data** with traditional **credit bureau features** to predict loan default.
> Archived second-view comparison build kept alongside the main `src/` + `pipelines/` tree.

---

## Overview

This pipeline takes raw applicant data (44,998 records, ~24% bad rate) through a complete ML workflow — from data cleaning to PD-calibrated models, a traditional scorecard with decision logic, and SHAP explainability.

### What was fixed from the earlier draft

| Issue | Severity | Previous Behaviour | Fix Applied |
|-------|----------|-------------------|-------------|
| ID column removal | **CRITICAL** | Substring match on `"account"` dropped `num_accounts_assess` (a legitimate credit feature) | Explicit ID list: only `dummy_id` removed |
| Leakage columns | **CRITICAL** | Only `highest_arrears_perf` removed | All 3 performance-period columns removed: `num_accounts_perf`, `highest_arrears_perf`, `age_oldest_perf` |
| Impute-before-split | **HIGH** | Median imputation computed on full dataset before train/test split | Split first, then impute using train medians only |
| Encode-before-split | **HIGH** | `get_dummies` on full data leaked category frequencies | Encode after split; align val/test columns to train |
| Feature engineering leakage | **HIGH** | Quantile thresholds computed on full data | Thresholds computed on train only, applied to val/test |
| Dual pipelines | **MEDIUM** | Two divergent scripts (`preprocess.py` and `run_pipeline.py`) | Single consolidated pipeline |
| Config mismatch | **LOW** | `config.py` feature lists never imported or used | Config is the single source of truth for all parameters |
| No PD calibration | **HIGH** | Raw model probabilities used as-is | Isotonic/Platt calibration on validation set |
| No scorecard | **NEW** | Not implemented | Full WoE/IV → scorecard points → decision regions |
| No decision logic | **NEW** | Not implemented | APPROVE / REFER / DECLINE cutoff framework |

---

## Pipeline Architecture

```
DRA_with_simulated_credit.xlsx
        │
        ▼
┌─────────────────┐
│   data_prep.py  │  Load → Clean → Drop IDs → Drop Leakage → SPLIT → Impute → Encode
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   features.py   │  Engineered features (thresholds from TRAIN only)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   models.py     │  Scale → Train LR/RF/XGB → PD Calibration (isotonic)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  evaluation.py  │  AUC, Gini, KS, Brier → Calibration plot → ROC curves
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  scorecard.py   │  WoE/IV → Points allocation → Score → APPROVE/REFER/DECLINE
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ shap_explain.py │  SHAP beeswarm, bar, dependence plots → CSV export
└─────────────────┘
```

---

## Quick Start

```bash
# 1. Install dependencies
cd archive/v1_initial
pip install -r requirements.txt

# 2. Run the full pipeline
python run_pipeline.py
```

All outputs land in `archive/v1_initial/outputs/`:

```
outputs/
├── models/          # Pickled scaler, base models, calibrated models
├── plots/           # Calibration, ROC, SHAP, score distribution, IV charts
└── reports/         # Metrics CSV, IV table, scorecard points, SHAP values
```

---

## Module Reference

### `config.py`
Single source of truth: paths, target variable, column lists, model hyperparameters, scorecard parameters, and calibration settings.

### `data_prep.py`
Handles the full data-preparation flow with the critical fix of **splitting before imputation and encoding**. The ID removal uses an explicit column list instead of the dangerous substring-matching pattern that was dropping legitimate credit features.

### `features.py`
Creates engineered features (risk flags, interaction terms, net-risk ratios) using quantile thresholds derived **only from the training set** to prevent information leakage.

### `models.py`
Trains three model families (Logistic Regression, Random Forest, XGBoost), scales features using a StandardScaler fitted on train data only, and calibrates each model's probabilities to true PD using `CalibratedClassifierCV` on the validation set.

### `evaluation.py`
Computes AUC, Gini, KS, and Brier score for both base and calibrated variants. Produces calibration curves (predicted PD vs observed default rate) and ROC curves for visual comparison.

### `scorecard.py`
Converts the logistic regression coefficients into a traditional credit scorecard with interpretable point allocations. Computes WoE and Information Value for all features. Applies a three-tier decision framework:

| Decision | Score Range | Description |
|----------|-------------|-------------|
| **APPROVE** | >= 620 | Auto-approve |
| **REFER** | 560 – 619 | Manual review |
| **DECLINE** | < 560 | Auto-decline |

### `shap_explain.py`
Runs SHAP TreeExplainer (for RF/XGB) or KernelExplainer (for LR) to produce global feature importance plots, beeswarm summaries, and per-feature dependence plots.

---

## Data Description

The dataset combines psychometric assessment data (DRA) with simulated credit bureau features:

| Feature Group | Count | Examples |
|---------------|-------|---------|
| DRA Dimensions | 4 | `dim_judgement`, `dim_core_traits`, `dim_emotional_understanding`, `dim_principles` |
| DRA HO Scales | 13 | `r_ho_tc1_ag`, `r_ho_em2_co`, `r_ho_vi4_st`, ... |
| DRA SF Scales | 26 | `r_sf_em1_ad`, `r_sf_tc1_al`, `r_sf_dm1_ps`, ... |
| Risk Scores | 3 | `total_risk_score`, `risk_mitigators`, `risk_drivers` |
| Credit Bureau (Assessment) | 3 | `num_accounts_assess`, `worst_arrears_assess`, `age_oldest_assess` |
| Product Type | 1 | `product_type` (Micro / Retail) |
| **Target** | 1 | `bad` (0 = good, 1 = default) |

**Records:** 44,998 | **Bad rate:** 24.19%

---

## Key Design Decisions

**Why isotonic calibration?** With ~45K records and a 24% event rate, we have enough data for isotonic regression to fit well without overfitting. It's more flexible than Platt scaling (sigmoid) and handles non-linear miscalibration better.

**Why scorecard from LR, not tree models?** Traditional credit scorecards require additive point allocations, which only logistic regression coefficients provide directly. Tree models are used for comparison and SHAP explanations.

**Why split before everything?** Imputing, encoding, and computing quantile thresholds on the full dataset before splitting introduces subtle information leakage. The test set "knows" about itself through the statistics. Our pipeline enforces strict temporal discipline.

---

## Outputs Produced

| File | Location | Description |
|------|----------|-------------|
| `model_metrics.csv` | `reports/` | AUC, Gini, KS, Brier for all models |
| `iv_table.csv` | `reports/` | Information Value for all features |
| `scorecard_points.csv` | `reports/` | Feature-to-points mapping |
| `scored_test_set.csv` | `reports/` | Test set scores + decisions + actuals |
| `shap_values_*.csv` | `reports/` | Raw SHAP values per observation |
| `shap_importance_*.csv` | `reports/` | Mean absolute SHAP per feature |
| `calibration.png` | `plots/` | PD calibration curve |
| `roc_curves.png` | `plots/` | ROC curves for all model variants |
| `score_distribution.png` | `plots/` | Score histogram by good/bad |
| `iv_chart.png` | `plots/` | Top features by Information Value |
| `shap_summary_*.png` | `plots/` | SHAP beeswarm plot |
| `shap_bar_*.png` | `plots/` | SHAP bar importance |
| `shap_dep_*.png` | `plots/` | SHAP dependence plots (top 4 features) |

---

## Repository Structure

```
Machine-Learning-For-Credit-Prediction-Phase-3/
├── data/
│   └── raw/
│       └── DRA_with_simulated_credit.xlsx
├── archive/
│   └── v1_initial/             ← THIS BUILD
│       ├── config.py           # All parameters and paths
│       ├── data_prep.py        # Ingest → clean → split → impute → encode
│       ├── features.py         # Feature engineering (train-only thresholds)
│       ├── models.py           # LR / RF / XGB + PD calibration
│       ├── evaluation.py       # Metrics + calibration & ROC plots
│       ├── scorecard.py        # WoE/IV, scorecard points, decision logic
│       ├── shap_explain.py     # SHAP explainability
│       ├── run_pipeline.py     # Master orchestrator
│       ├── requirements.txt    # Python dependencies
│       ├── README.md           # This file
│       └── outputs/
│           ├── models/
│           ├── plots/
│           └── reports/
├── src/                        ← Main build (live)
├── pipelines/                  ← Main pipeline orchestrator
└── README.md
```

---

## How to Compare with the Main Build

Run both pipelines on the same data and compare:

1. **Metrics:** Does AUC/Gini differ after leakage removal?
2. **Feature importance:** Are the same features ranked highly?
3. **Calibration:** Are predicted PDs closer to observed default rates?
4. **Scorecard decisions:** What approval/decline rates result?

---

*Archived v1 build — kept as a parallel reference implementation for the credit data pipeline project.*
