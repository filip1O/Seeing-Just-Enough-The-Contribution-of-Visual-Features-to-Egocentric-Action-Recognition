# Reduction Experiment demos

This directory contains three manually configured analysis demonstrations: response-to-ground-truth semantic scoring, random-forest classification with SHAP, and Boruta-style feature assessment.

Install the root `requirements.txt` and run commands from the repository root so the example relative paths resolve correctly.

## Response-to-ground-truth semantic similarity

`SBERT_framework_response-GT.py` compares each participant response with its Human Ground Truth label.

It combines:

- Sentence-BERT similarity between the complete response and `HGT`;
- Word2Vec similarity between extracted object nouns;
- Word2Vec similarity between extracted action verbs;
- the study's weighting and opposite-action correction rules.

The example input is:

```text
Reduction Experiment/example_video_responses/pooled_LL_LL_LL_03159.csv
```

Required columns include `Response` and `HGT`. The script overwrites the input CSV after adding a `sem_sim` column, so work on a copy if the original file must remain unchanged.

Before running, set:

```python
input_path = "Reduction Experiment/example_video_responses/pooled_LL_LL_LL_03159.csv"
filename = os.path.basename(input_path)
```

Then run:

```bash
python "Reduction Experiment/demos/SBERT_framework_response-GT.py"
```

The first run may download and cache `all-mpnet-base-v2` and `word2vec-google-news-300`.

## Leave-one-out random forest and SHAP

`loo_randomforest_classification_and_shap.py` evaluates a binary random forest with leave-one-out cross-validation and computes feature importance and TreeSHAP values.

The default feature set contains four object-region measures and seven GBVS saliency measures. The script log-transforms those features, optionally imputes missing values, fits a 500-tree random forest, aggregates classification metrics, and generates feature-importance and SHAP plots.

Before running, review:

- `input_path`: classification workbook and sheet;
- `out_path`: output workbook;
- `plots_dir` and `model_out_path`;
- `columns_to_keep` and `skewed_features`;
- target `y`: `MIRC`, `Easy_classification_difficulty`, or `ts_rel_rec`;
- `save_plots`, `miss_values`, and `total_runs`.

Example classification workbooks are under `Reduction Experiment/binary_classification_sets_and_results/`.

Run:

```bash
python "Reduction Experiment/demos/loo_randomforest_classification_and_shap.py"
```

Outputs include:

- an Excel workbook with `results`, `feature_importances`, and `SHAP_Values` sheets;
- `random_forest_model.pkl`;
- `feature_importances_all.png` and `shap_summary_all.png` when plot saving is enabled.

The script displays Matplotlib windows; close each plot window to let execution continue.

## Boruta-style SHAP bootstrap

`loo_randomforest_boruta_bootstrap.py` adds permuted shadow copies of every feature, performs leave-one-out random-forest TreeSHAP analysis, and bootstraps feature importance 1,000 times.

It reports:

- mean absolute SHAP importance and 95% intervals;
- strict and loose shadow-feature thresholds;
- Boruta keep/reject decisions;
- reference ranks, rank intervals, and rank stability.

Before running, review `input_path`, `feat_out_path`, `columns_to_keep`, `skewed_features`, target `y`, `miss_values`, and bootstrap count `B`.

Run:

```bash
python "Reduction Experiment/demos/loo_randomforest_boruta_bootstrap.py"
```

The output is the CSV specified by `feat_out_path`. If `miss_values = 1` is enabled, also import `KNNImputer` from `sklearn.impute`; the checked-in snapshot leaves missing-value handling disabled.
