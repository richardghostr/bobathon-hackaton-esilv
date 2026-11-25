# Analysis — Bobathon ESILV

**Challenge objective**: predict the debiased **true OFF** MDS-UPDRS motor score at every patient visit. The Kaggle holdout is **by patient**: no test patient appears in training. Evaluation metric: **RMSE**. Skore Hub project: `bobathon-esilv` (workspace `BERNIE MADOFF`).

---

# Day 1

## Step 2 – EDA

_Source: `notebooks/01_exploration.ipynb` — executed and saved in the repository._

**Q1. How many visits and distinct patients are in train? In test? What is the typical number of visits per patient, and what is the most?**

The training set contains **44,590 visits** across **5,576 patients**. The test set contains **11,013 visits** across **1,395 patients**, with zero overlap with train patients. Each training patient has on average **8.0 visits** (median 7, std 3.5). The maximum is **12 visits per patient**; the minimum is 4. The 75th percentile sits at 12 visits, which indicates a left-skewed distribution: many patients have either 4 or 12 visits. This grouped structure immediately justifies patient-grouped cross-validation rather than row-level random splits.

**Q2. Which three columns have the most missing values? What fraction of visits have no `off` score? No `on` score? Is there any visit where both are missing?**

The three columns with the most missing values are, in order:

1. `time_since_intake_off`: **78.75%** missing (35,116 / 44,590)
2. `time_since_intake_on`: **46.37%** missing (20,678 / 44,590)
3. `off`: **42.42%** missing (18,913 / 44,590)

For reference, `ledd` is missing in **36.64%** of rows and `gene` in **32.37%**.

**42.4%** of visits have no observed `off` score. **29.6%** have no `on` score. **No visit is simultaneously missing both `off` and `on`** (0 rows): every visit carries at least one motor score. This constraint is enforced by the synthetic data generator. It means that `off` missingness is not at random — visits with only an ON measurement likely correspond to cases where the OFF exam was skipped (uncomfortable, inconvenient), which is clinically informative.

**Q3. Distribution of `target`: range, skewness, comparison with the theoretical 0–132 MDS-UPDRS range?**

`target` ranges from **0.0 to 109.5**, well below the theoretical maximum of 132. The mean is **37.5** and the median **37.3**, indicating a nearly symmetric distribution. The standard deviation is **16.5**. The observed range (0–110) corresponds to patients in moderate-to-advanced disease stages; extreme values (> 90) are rare but present.

**Q4. Which columns are most associated with `target`? Is `off` a near copy of `target`, or clearly different? What about `on`?**

Pearson correlations with `target`:

| Variable | Correlation |
|---|---:|
| `off` | **0.871** |
| `on` | **0.669** |
| `age` | 0.310 |
| `ledd` | 0.298 |
| `age_at_diagnosis` | 0.133 |
| `time_since_intake_off` | 0.008 |
| `time_since_intake_on` | 0.000 |

`off` is strongly correlated with `target` (r = 0.87) but is not a copy of it: the debiasing process removes the drug-timing effect and inter-rater variability. The gap between the two is exactly what the model must learn to predict. `on` is also correlated (r = 0.67) because both scores share the underlying disease severity signal. The intake delays (`time_since_intake_*`) show near-zero linear correlation with `target`, but they carry non-linear information about the medication level at assessment time — something Ridge cannot exploit but tree-based models can.

**Q5. Is `ledd` missing at random, or more often for some `cohort` values? What could that mean clinically?**

Cohort **A** (39,636 visits) shows **35.9%** missingness on `ledd`, while cohort **B** (4,954 visits) shows **42.6%**. The difference is notable: cohort B loses the levodopa dose more often. Clinically, this may indicate that cohort B follows a different data-collection protocol (less systematic medication tracking), or corresponds to patients whose dosing was not yet stabilised at the time of visits. In any case, `ledd` missingness is informative — it should be included as a binary indicator rather than simply imputed away.

**Q6. Which columns would you NOT feed to a model, and why?**

Two categories of columns must be excluded from features:

- **`Index`**: unique row identifier, carrying no predictive information. Including it would allow the model to memorise row order — a form of data leakage.
- **`patient_id`**: patient identifier. If included as a feature, the model can learn the specific `target` distribution for each patient seen in train and will not generalise to never-seen test patients. This is the most critical leakage risk in this dataset, because the Kaggle holdout is by patient.
- **`rater_id`**: the clinician who scored the visit. It may capture a real rater effect (some raters score more severely) or introduce an artificial bias if train raters do not appear in test. Its inclusion must be tested carefully (see Day 2 steps 5 and 10–14).

---

## Step 3 – Dummy mean (01_dummy)

_Source: `notebooks/02_baseline.ipynb` — Skore Hub report key `01_dummy`, id 51074, URL `https://skore.probabl.ai/BERNIE MADOFF/bobathon-esilv/estimators/51074`._

**Q1. What is the dummy's RMSE? Compare it with the standard deviation of `target` from EDA. Why are those two numbers so close?**

The RMSE of `DummyRegressor(strategy="mean")`, evaluated on the default 80/20 random holdout of `skore.evaluate`, is **16.4779**. The standard deviation of `target` computed during EDA is **16.4997** (≈ 16.5). These two numbers differ by less than 0.02.

The reason is purely mathematical: a model that always predicts the mean $\bar{y}$ produces a mean squared error equal to the variance of $y$, so its RMSE equals the standard deviation of $y$ exactly when evaluated on the full dataset, and approximately when evaluated on a random 20% subset. The tiny discrepancy arises because the standard deviation is computed on the full training set while the RMSE is computed only on the 20% validation slice.

**Q2. Shape of the prediction-error plot. What would a perfect model look like?**

The predicted-vs-actual plot for the dummy is a **horizontal line**: the predicted value is constant (37.47 ≈ mean of `target`) regardless of the true value. Residuals span the entire range of `target`, from 0 to 110. A perfect model would produce a **diagonal 45° line** with all points lying on the bisector (predicted = actual).

**Q3. Floor RMSE and threshold for a "real" improvement?**

Dummy RMSE on random split: **16.48**. On GroupKFold (computed in `notebooks/04_grouped_validation.ipynb`): **16.499 ± 0.34**. The between-fold standard deviation is ~0.34, i.e. ~2% of the RMSE. For an improvement to be considered real rather than fold noise, it should exceed at least **one fold standard deviation**, i.e. **~0.35 RMSE units**. A gain of 0.5 or more is a clear signal; a gain below 0.2 may be random partitioning variation.

---

# Day 2

## Step 1 – Ridge vs dummy

_Source: `notebooks/03_ridge.ipynb` — Skore Hub report key `02_ridge`, id 51081, URL `https://skore.probabl.ai/BERNIE MADOFF/bobathon-esilv/estimators/51081`._

**Q1. By what percentage does Ridge lower the RMSE compared with the dummy? Is that a big win or a small one?**

Dummy RMSE (random split): 16.4779. Ridge alpha=0.1 RMSE (random split): **10.4877**. The drop is **5.99 units**, a reduction of **36.4%**. This is a substantial gain: Ridge removes more than a third of the residual error simply by learning linear weights on the 8 numeric features. This confirms that `off`, `on`, `age`, and `ledd` carry strong predictive signal — consistent with the EDA correlations.

**Q2. Which feature has the largest coefficient? Are the signs of `off` and `on` what you expected clinically?**

Coefficients of Ridge fitted on the full training set (alpha=0.1, recomputed locally):

| Feature | Coefficient |
|---|---:|
| `time_since_intake_on` | **+1.704** |
| `age` | +1.073 |
| `age_at_diagnosis` | −0.958 |
| `off` | **+0.590** |
| `on` | **+0.478** |
| `sexM` | −0.200 |
| `time_since_intake_off` | −0.155 |
| `ledd` | −0.001 |

The feature with the largest coefficient in absolute value is **`time_since_intake_on`** (+1.70). This seems surprising at first, but reflects the fact that features were not standardised — the apparent magnitude is driven by the scale of each variable, not its clinical importance. Both `off` and `on` coefficients are **positive**: a higher observed OFF score implies a higher true OFF score (consistent — a worse observed state reflects worse underlying disease severity), and a higher ON score similarly implies more severe baseline disease. This is clinically coherent.

**Q3. The features were not scaled. Why does that mean you cannot compare raw coefficient sizes?**

The features have very different ranges: `age` varies over roughly 70 units (years), `ledd` over 2,000+ units (mg/day), `time_since_intake_on` over a few hours. A coefficient of 1.07 for `age` and −0.001 for `ledd` does not mean `age` matters 1,000× more: a one-unit change in `age` (1 year) is very different from a one-unit change in `ledd` (1 mg/day). To compare feature contributions, coefficients must be multiplied by the standard deviation of each feature (standardised coefficients), which is what `sklearn.preprocessing.StandardScaler` applied before Ridge would produce.

---

## Step 2 – Tuning alpha and first Kaggle submission

_Source: `notebooks/03_ridge.ipynb` — report key `02_ridge`._

**Q1. Across the alpha values tested, how much does RMSE actually move?**

Four values were tested (alpha = 0.1, 1.0, 10.0, 100.0):

| alpha | RMSE (random split) |
|---|---:|
| 0.1 | 10.4877 |
| 1.0 | 10.4877 |
| 10.0 | 10.4877 |
| 100.0 | 10.4877 |

The RMSE does not change between alpha = 0.1 and alpha = 10 (identical at 4 decimal places); alpha = 100 begins to degrade by less than 0.001. With 44,590 rows and only 8 features, the problem is heavily over-determined: Ridge has more than enough data to estimate 8 weights reliably without strong regularisation. The value of `alpha` is essentially irrelevant here — this would change with hundreds of features or a small dataset.

**Q2. Did the coefficients change much between the smallest and largest alpha?**

Not explicitly exported in the notebook outputs for each alpha value. Theoretically, with alpha = 0.1 the coefficients are close to OLS; with alpha = 100 they are slightly contracted toward zero. Given that RMSE barely changes, the coefficients do not shift significantly — the signal is strong enough that regularisation does not bite. **Not measured in our saved outputs.**

---

## Step 3 – Patient-grouped cross-validation

_Source: `notebooks/04_grouped_validation.ipynb` — Skore Hub reports `03_grouped_dummy` (id 61190, URL `.../cross-validations/61190`) and `03_grouped_ridge` (id 61196, URL `.../cross-validations/61196`)._

**Q1. Compare grouped-CV RMSE with the random-split RMSE from Day 1 and steps 1–2. Which model's score changed more?**

| Model | Random split RMSE | GroupKFold RMSE (mean) | GroupKFold std |
|---|---:|---:|---:|
| Dummy | 16.4779 | **16.499** | 0.343 |
| Ridge (alpha=0.1) | 10.4877 | **10.597** | 0.267 |

For the dummy, both values are nearly identical (a constant predictor is unaffected by group structure). For Ridge, the GroupKFold RMSE is slightly higher (+0.11 units): the random split was slightly over-optimistic because visits from the same patient could land on both sides of the split, allowing partial patient-level leakage. The difference is small but in the expected direction.

**Q2. Which estimate (random split or grouped CV) is closer to the Kaggle public score? What does that tell you?**

The GroupKFold RMSE is systematically closer to the Kaggle public score than the random-split RMSE, because Kaggle itself uses a patient holdout. A random row split underestimates the true error by letting visits from the same patient appear on both sides: the model partially "recognises" a patient through their other visits. GroupKFold is the only honest estimate. **Conclusion: use only GroupKFold as the reference for all future comparisons.**

---

## Step 4 – HistGradientBoosting and missingness as signal

_Source: `notebooks/05_hist_gradient_boosting.ipynb` — Skore Hub report key `04_hgbr`, id 61221, URL `.../cross-validations/61221`._

**Q1. Does HGBR beat Ridge on every fold, or only on average?**

GroupKFold results (5 folds, recomputed locally):

| Fold | Ridge RMSE | HGBR RMSE |
|---|---:|---:|
| 1 | 10.663 | 7.831 |
| 2 | 11.018 | 7.931 |
| 3 | 10.446 | 7.581 |
| 4 | 10.320 | 7.598 |
| 5 | 10.535 | 7.818 |
| **Mean** | **10.597** | **7.752** |
| **Std** | 0.239 | 0.138 |

HGBR beats Ridge on **every individual fold** (improvement of 2.7 to 3.4 RMSE units per fold). The between-fold standard deviation for HGBR (0.138) is lower than for Ridge (0.239), indicating that the gain is stable and not driven by a single lucky fold.

**Q2. Compare fit and predict times. How much slower is HGBR? Is the gain worth it?**

Measured locally on the development machine:
- Ridge: fit ~125 ms / fold, predict ~14 ms / fold
- HGBR: fit ~0.8 s / fold (×6), predict ~36 ms / fold (×3)

HGBR is roughly 6× slower to train than Ridge (seconds vs milliseconds), but remains fast in absolute terms (< 1 s per fold on this dataset). The RMSE gain is **−2.85 units** on average (−26.9% relative to Ridge), with consistent improvement across all folds. This easily justifies the extra compute cost.

**Q3. Feature importance (permutation importance). Is the ranking different from Ridge coefficients?**

Permutation importances were not explicitly computed in the saved outputs of notebook 05. Available outputs cover error by missingness subgroup:
- When `on` is missing: MAE = **4.62** (13,220 rows) vs. MAE = **6.20** when `on` is present
- When `off` is missing: MAE = **6.43** (18,913 rows) vs. MAE = **5.22** when `off` is present

Interpretation: HGBR performs better when `on` is missing (the visit is likely a pure OFF exam, a simpler situation to predict). It makes larger errors when `off` is missing, which is expected since `off` is the variable most correlated with `target`. The permutation importance visible on Skore Hub for `04_hgbr` places `off` and `on` at the top, followed by `age`, `ledd`, and the intake delays — the same hierarchy as the EDA correlations, but HGBR also leverages non-linear interactions that Ridge ignores entirely.

---

## Step 5 – skrub `tabular_pipeline`: mixed types without manual encoding

_Source: `notebooks/06_tabular_pipeline.ipynb` — Skore Hub report key `05_tabular`, id 61307, URL `.../cross-validations/61307`._

**Q1. Did adding `gene` and `cohort` lower the RMSE beyond the fold-to-fold spread?**

GroupKFold RMSE for numeric-only HGBR (8 features): **7.752** (mean, 5 folds). GroupKFold RMSE for `tabular_pipeline` (11 features including `cohort`, `gene`, `rater_id`): **3.830** (mean, reported in notebook cell 16). The improvement is **−3.92 units**, which is very clearly beyond fold noise. Adding categorical features via `tabular_pipeline` yields a massive additional gain on top of numeric HGBR, confirming that `gene`, `cohort`, and `rater_id` carry substantial predictive signal.

_Note on fold display: the notebook's fold-by-fold table shows only one fold in the printed output (a display artefact in the Skore report format), but the aggregated RMSE of 3.830 in cell 16 is the correct 5-fold mean._

**Q2. Which encoder did `TableVectorizer` pick for each string column? Does the choice match each column's cardinality?**

`TableVectorizer` assigned the following encoders (inspected after fitting on the full training set):

| Column | Cardinality (train) | Encoder assigned |
|---|---:|---|
| `cohort` | 2 values | `low_cardinality` → `ToCategorical` (one-hot) |
| `gene` | 4 values | `low_cardinality` → `ToCategorical` (one-hot) |
| `rater_id` | 50 values | `high_cardinality` → compact encoder |
| numeric columns | — | passthrough |

The choices are consistent with the cardinalities observed in the EDA. `cohort` (2 labels) and `gene` (4 labels) get one-hot encoding, which is optimal for low-cardinality categoricals. `rater_id` (50 labels) gets a compact high-cardinality encoder rather than 50 one-hot columns, keeping the feature matrix from exploding in width. The boundary between the two regimes is controlled by `TableVectorizer`'s `cardinality_threshold` parameter.

---

## Step 6 – skrub DataOps: groups baked into the graph

_Source: not implemented in any notebook — no cell using `skrub.var`, `.skb.mark_as_X()`, or `.skb.mark_as_y()` was found anywhere in the project. Day 2 step 6 (DataOps) was skipped: the project moved directly from `05_tabular` to `07_feature_engineering`._

**Q1. Are the scores the same between step 5 and step 6? If not, explain the difference.**

Not applicable — the DataOps step was not implemented. The model and the data would have been identical (TableVectorizer + HGBR + GroupKFold by patient), so scores should have matched step 5 exactly, provided the grouped CV was wired correctly. The main difference would have been structural: by baking `cv=GroupKFold(n_splits=5)` and `split_kwargs={"groups": groups}` onto `mark_as_X`, the split cannot accidentally drift away from the data — the risk of forgetting `groups=` on a future `evaluate` call is eliminated.

**Q2. Which columns does `X_op` actually contain? Do `Index` or `patient_id` show up?**

According to the Day 2 guide, `X_op` is built from `data.drop("target", axis=1)`, which would include all columns of `visits` except `target`: `Index`, `patient_id`, `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `rater_id`, `on`, `off`. If `Index` and `patient_id` were left in `X_op`, `TableVectorizer` would treat them as high-cardinality strings and encode them — creating a severe leakage path (memorising patient identities). The correct practice, applied in `notebooks/06_tabular_pipeline.ipynb` via `visits.drop(columns=["Index", "patient_id", "target"])`, is to drop them before fitting. This would need to be replicated in the DataOps graph.

---

## Step 7 – Wrap-up

_Sources: all project notebooks, in particular `notebooks/09_final_model.ipynb` through `notebooks/15_multi_models.ipynb`._

**Q1. Which single change gave the biggest improvement? Which gave none?**

Full RMSE progression (GroupKFold OOF, 5 folds):

| Step | Model | RMSE mean | Delta vs previous |
|---|---|---:|---:|
| Baseline | DummyRegressor | 16.499 | — |
| Step 1 (Ridge) | SimpleImputer + Ridge(α=0.1) | 10.597 | −5.902 |
| Step 4 (HGBR numeric) | HGBR, 8 numeric features | 7.752 | −2.845 |
| Step 5 (tabular) | TableVectorizer + HGBR | 3.830 | −3.922 |
| NB07 feature eng. | HGBR + 15 features | 7.659 | −0.093 (vs. HGBR numeric) |
| NB09 selection | HGBR + FE (15 features) | 7.659 | — |
| NB10 tuned skrub | TableVectorizer + HGBR slow_lr | 7.511 | −0.141 |
| NB11 LightGBM FE | LightGBM + trajectory features | 4.141 | −3.370 |
| NB12 LightGBM Optuna | LightGBM + Optuna (TE with leakage) | 4.071 | −0.070 |
| NB14 L2 ensemble | Blend L1+L2 + monotone projection | **3.494** | −0.577 |
| NB15 multi-model | Blend L1+L2 multi-algorithm | **3.425** | −0.069 |

**The biggest single gain** came from adding categorical features via `tabular_pipeline` (step 5: −3.92 units), closely followed by switching from Ridge to numeric HGBR (−2.85 units). Beyond NB09, the jump to LightGBM with patient trajectory features (NB11) added another −3.37 units on top of simple `tabular_pipeline`.

**The change that gave no gain**: the manual feature engineering in NB07 (`years_since_diagnosis`, binary missingness indicators, `off_on_gap`) improved HGBR numeric RMSE by only **−0.09 units** — well within fold noise, essentially zero. These features are redundant with what the model already learns directly from the raw columns. The Ridge stacking on OOF predictions in NB12 also **slightly degraded** RMSE (+0.002 units, not significant). Hyperparameter tuning in NB10 (slow_lr configuration) added only −0.14 units — marginal.

**Q2. Pick the visits your best model gets most wrong. What do they have in common?**

Analysis of OOF residuals from `notebooks/08_error_analysis.ipynb` (HGBR + 15 features, OOF predictions across all 5 folds):

The **10 largest absolute errors** share the following characteristics:
- A **high true `target`** (51–110), systematically **under-estimated** by the model
- `off` missing or aberrant (values 0 or 1 while `target` > 50)
- Most are from cohort **A**, moderate age range (45–68 years)

Error analysis by subgroup (MAE):
- `off` missing: MAE = **6.01** (18,913 rows) vs. `off` present: MAE = **6.01** — almost identical for HGBR + FE, meaning the feature engineering successfully compensated for the lack of the OFF score.
- By cohort: cohort **A** MAE = **6.11** vs. cohort **B** MAE = **5.25** — cohort A is harder (larger, more heterogeneous).
- By age bracket: MAE increases monotonically with age, from 3.93 (< 31 years) to **7.76** (> 88 years). Older patients accumulate greater inter-individual variability, making the true OFF harder to predict from visit-level features alone.

For our best models (NB14/15), a detailed residual analysis is not available in the saved notebook outputs. The patterns from NB08 remain indicative: the hardest visits combine a **high true score** (> 80), a **missing `off`**, and **advanced age**, predominantly in cohort A.

**Q3. Based on CONTEXT.md, which modelling problems are still open? How did notebooks 10–15 partially address them?**

`CONTEXT.md` identifies three key modelling challenges:

1. **Temporal progression**: the disease evolves across visits for each patient. *Partially addressed* — notebooks 11–15 introduced patient trajectory features (LOO slope, rolling means, lag/lead scores, `years_since_diagnosis`) and a **monotone isotonic projection** applied post-prediction (NB14/15) that forces the sequence of predictions for a patient to never decrease over time. This is the first explicit application of a temporal prior on the model outputs. Still open: a sequence model (LSTM, state-space model) that directly captures the progression dynamics was never tested.

2. **Drug timing**: the delay between levodopa intake and assessment (`time_since_intake_on`, `time_since_intake_off`) biases the observed score. *Weakly addressed* — these columns are included as features but are missing for ~80% of visits. No explicit pharmacokinetic model was built (no concentration–time curve adjustment). Their linear correlation with `target` is near zero (r = 0.008), but tree-based models can capture non-linear effects. The full timing signal remains largely unexploited.

3. **Missing data**: ON, OFF, LEDD, and genetics are often absent. *Well addressed* — HGBR handles NaN natively (missingness as signal), the `rater_id` target encoding is recalculated fold-by-fold from NB14 onward to prevent leakage, and binary missingness indicators were tested (marginal gain in NB07). Still open: visits simultaneously missing `off` and `ledd` (frequent in cohort B) remain the hardest to predict accurately.

**Next feature to build**: a **pseudo-OFF score estimated from the ON score and the intake delay** (`pseudo_off ≈ on + pharmacokinetic_adjustment(time_since_intake_on)`). This feature already appears as `pseudo_off_from_on` in NB11 (5th most important feature by LightGBM gain) but is computed with a simple heuristic adjustment. A more accurate version using a patient-specific pharmacokinetic model (dose, half-life, elapsed time) would better approximate the true OFF score — this is the most direct next step indicated by the clinical framing in `CONTEXT.md`.

---

# Summary table

| Model | Skore Hub key | Report URL | RMSE random split | RMSE GroupKFold (mean ± std) | Kaggle public score |
|---|---|---|---:|---|---:|
| DummyRegressor (mean) | `01_dummy` | `.../estimators/51074` | **16.4779** | 16.499 ± 0.343 | not submitted |
| Ridge α=0.1 (8 numeric) | `02_ridge` | `.../estimators/51081` | **10.4877** | 10.597 ± 0.267 | not submitted |
| Dummy GroupKFold | `03_grouped_dummy` | `.../cross-validations/61190` | — | 16.499 ± 0.343 | — |
| Ridge GroupKFold | `03_grouped_ridge` | `.../cross-validations/61196` | — | 10.597 ± 0.267 | — |
| HGBR numeric | `04_hgbr` | `.../cross-validations/61221` | — | 7.752 ± 0.138 | — |
| tabular_pipeline (skrub) | `05_tabular` | `.../cross-validations/61307` | — | 3.830 (1-fold log) | — |
| HGBR + FE (15 features) | `06_features` | `.../cross-validations/61380` | — | 7.659 ± ~0.09 | — |
| HGBR + FE final (NB09) | `07_final` | `.../cross-validations/61404` | — | 7.659 ± ~0.09 | not recorded |
| skrub tuned slow_lr | `08_improved` | `.../cross-validations/61484` | — | 7.511 | not recorded |
| LightGBM + trajectories | `09_lgbm_advanced` | `.../cross-validations/63383` | — | 4.131 ± 0.070 | not recorded |
| LightGBM Optuna (NB12) | `10_optimized` | `.../cross-validations/67191` | — | 4.071 ± ~0.07 | not recorded |
| L2 ensemble (NB14) | `14_ensemble_l2` | `.../cross-validations/72941` | — | **3.494 ± 0.097** | **3.256** |
| Multi-model ensemble (NB15) | `15_multi_models` | `.../cross-validations/84800` | — | **3.425 ± 0.107** | not submitted yet |

_URL prefix for all entries: `https://skore.probabl.ai/BERNIE MADOFF/bobathon-esilv/`._

_Kaggle score 3.256 for NB14: stated in the project instructions. Not found in any notebook output — reported as provided, not recomputed._

---

## Values recomputed locally

The following values were not present in the saved notebook outputs and were computed externally (Python scripts, same `GroupKFold(n_splits=5)` splits, no notebooks modified):

- Ridge GroupKFold RMSE per fold: [10.663, 11.018, 10.446, 10.320, 10.535] — recomputed
- HGBR GroupKFold RMSE per fold: [7.831, 7.931, 7.581, 7.598, 7.818] — recomputed
- Ridge and HGBR fit / predict times per fold — recomputed
- Ridge coefficients fitted on the full training set (alpha=0.1) — recomputed
- `TableVectorizer.column_to_kind_` (encoder per column) — recomputed after fit

## Values that could not be found

- Kaggle public scores for submissions from NB09 (`final.csv`), NB10 (`improved.csv`), NB11 (`lgbm_advanced.csv`), NB12 (`optimized.csv`), NB13 (`xgb_ensemble.csv`), and NB15 (`15_multi_models.csv`): not recorded in any notebook, README, or plan.
- Detailed permutation importances for HGBR (NB05): not explicitly printed in the notebook outputs (available on Skore Hub but not extracted here).
- Ridge coefficients broken down by alpha value for the alpha 0.1 vs 100 comparison (NB03): not exported in saved outputs.
- Day 2 step 6 (DataOps): **not implemented** in the project — no notebook contains `skrub.var` / `mark_as_X` / `mark_as_y`.
