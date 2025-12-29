import numpy as np, shap, pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
import os

# specify if there are missing feature values for some samples
miss_values = 0

# Load the excel file wwith samples and features
input_path = "Reduction Experiment/binary_classification_sets_and_results/Easy_vs_Hard_MIRCs/MIRCs_Easy_Hard_sample.xlsx"      # Replace with path to your downloaded excel file with samples for classification
df = pd.read_excel(input_path, sheet_name='Sheet1')

# boruta output
feat_out_path = '.../boruta.csv'          # Replace with .csv path to save the Boruta SHAP results for main feature effects

# Specify columns to keep (features to use)
columns_to_keep = ['active_object', 'active_hand', 'contextual_objects', 'background', 'salient_colour', 'salient_dklcolour', 'salient_flicker', 'salient_intensity', 'salient_motion', 'salient_orientation', 'salient_contrast']  # replace with your desired features

# Identify skewed features for log transformation
skewed_features = ['active_object', 'active_hand', 'contextual_objects', 'background', 'salient_colour', 'salient_dklcolour', 'salient_flicker', 'salient_intensity', 'salient_motion', 'salient_orientation', 'salient_contrast']  # Replace with your actual skewed feature names
# Apply log transformation to skewed features
for feature in skewed_features:
    df[feature] = np.log1p(df[feature])  # Use log1p to handle zero values (log1p(x) = log(x + 1))

# Specify the label column
y = df['MIRC'] # set to 'MIRC' for MIRC vs. nonMIRC, 'Easy_classification_difficulty' for Easy vs. Hard MIRC, 'ts_rel_rec' for spatial vs. spatiotemporal MIRC

# Prepare the data
X = df[columns_to_keep]
# Loop through all columns and check their data type, gather, dummy encode
categorical_columns_multi = []
for column in X.columns:
    print(column)
    if (X[column].dtype == 'object' or X[column].dtype.name == 'category') and column not in ['video']:
        # Check the number of unique values (levels) in the column
        unique_values = X[column].nunique()
        if unique_values > 2:
            # If the column has more than 2 levels, add it to the list for one-hot encoding
            categorical_columns_multi.append(column)
        elif unique_values == 2:
            # If the column has exactly 2 levels, map the two levels to 0 and 1
            levels = X[column].astype('category').cat.categories  # Get the levels
            X[column] = X[column].astype('category').cat.codes
            # Print the message with the assigned values
            print(f"In column [{column}] level [{levels[1]}] was assigned 1 and level [{levels[0]}] was assigned 0.")

# Apply one-hot encoding to columns with more than 2 levels
X = pd.get_dummies(X, columns=categorical_columns_multi, drop_first=True)
# Replace underscores with empty spaces
X.columns = X.columns.str.replace('_', ' ')

# resolve missing values, if any
# Store the positions of all empty values in a dictionary
if miss_values == 1:
    nan_positions = {}
    for col in X.columns:
        nan_positions[col] = X[col][X[col].isna()].index.tolist()
    # use KNN Imputer to impute missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    X_imputed = knn_imputer.fit_transform(X)
    # Convert back to DataFrame to keep column names
    X = pd.DataFrame(X_imputed, columns=X.columns)
    # After imputing replace all missing values with -999 - placeholder
    for col, positions in nan_positions.items():
        X.loc[positions, col] = -999

# ============================================================
#  Random Forest + SHAP + Boruta-style feature significance
#  - Performs Leave-One-Out CV with TreeSHAP
#  - Compares real vs shadow (permuted) features
#  - Bootstraps to estimate uncertainty and rank stability
# ============================================================

# ---------- 0) Initialize model ----------
rf = RandomForestClassifier(
    n_estimators=500,          # Number of trees (higher = more stable, slower)
    criterion='gini',          # Classification criterion
    bootstrap=True,            # Bootstrap sampling
    max_depth=None,            # No depth limit
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt'        # Standard heuristic for classification
)

# ---------- Prepare constants ----------
feat_names = X.columns.tolist()
p = len(feat_names)                     # number of real features
p2 = p * (p - 1) // 2                   # number of unique feature–feature pairs

# Helper to create permuted (shadow) features for Boruta-style comparison
def add_shadow(df, rng):
    shadow = df.copy()
    for c in df.columns:
        shadow[c] = rng.permutation(df[c].values)
    shadow.columns = [f"shadow_{c}" for c in df.columns]
    return pd.concat([df, shadow], axis=1)

# ============================================================
# 1) Leave-One-Out CV with SHAP
# ============================================================
loo, rng = LeaveOneOut(), np.random.default_rng(0)
row_list = []

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr = y.iloc[train_idx]

    # Add permuted "shadow" copies of each feature
    X_tr_sh = add_shadow(X_tr, rng)
    X_te_sh = add_shadow(X_te, rng)

    # Train forest and compute TreeSHAP explanations
    rf.fit(X_tr_sh, y_tr)
    expl = shap.TreeExplainer(rf)

    # ----- Main-effect SHAP values -----
    sv = expl.shap_values(X_te_sh)
    if isinstance(sv, list): arr = sv[-1]  # class-1 shap values
    else: arr = sv
    if arr.ndim == 3: arr = arr[0, :, -1]
    else: arr = arr.reshape(-1)
    row_list.append(np.abs(arr))           # take absolute SHAP importances

# Stack LOOCV results
shap_rows = np.vstack(row_list)
real_imp = shap_rows[:, :p]
shadow_imp = shap_rows[:, p:]

# ============================================================
# 2) Feature-level Boruta decision + bootstrap confidence
# ============================================================
B = 1000
boot_mean = np.empty((B, p))
for b in range(B):
    # Bootstrap resampling of samples
    idx = resample(np.arange(shap_rows.shape[0]), replace=True, random_state=rng.integers(1e9))
    boot_mean[b] = real_imp[idx].mean(0)

# Mean and CI of feature importances
mean_imp = boot_mean.mean(0)
ci_lo, ci_hi = np.percentile(boot_mean, [2.5, 97.5], axis=0)

# Boruta thresholds:
# strict → 95th percentile of max shadow importance
# loose → 95th percentile of all shadow values
thr_strict = np.percentile(shadow_imp.max(1), 95)
thr_loose = np.quantile(shadow_imp.ravel(), 0.95)
boruta_keep_strict = mean_imp > thr_strict
boruta_keep_loose = mean_imp > thr_loose

# ============================================================
# 3) Feature rank stability across bootstraps
# ============================================================
ref_order = mean_imp.argsort()[::-1]
ref_ranks = np.empty_like(ref_order)
ref_ranks[ref_order] = np.arange(p)

boot_ranks = np.empty((B, p), dtype=int)
for b in range(B):
    ord_b = boot_mean[b].argsort()[::-1]
    rk_b = np.empty_like(ord_b)
    rk_b[ord_b] = np.arange(p)
    boot_ranks[b] = rk_b

# Spearman correlations quantify global rank stability
rho = np.array([spearmanr(ref_ranks, boot_ranks[b]).correlation for b in range(B)])
rank_ci_lo, rank_ci_hi = np.percentile(boot_ranks, [2.5, 97.5], axis=0)
rank_width = rank_ci_hi - rank_ci_lo
stable_rank = rank_width <= 2   # rank "stable" if varies ≤ 2 positions

# ============================================================
# 4) Summary
# ============================================================
summary = pd.DataFrame({
    "feature": feat_names,
    "mean|SHAP|": mean_imp,
    "ci_low": ci_lo,
    "ci_high": ci_hi,
    "keep_boruta_strict": boruta_keep_strict,
    "keep_boruta_loose": boruta_keep_loose,
    "thr_strict": thr_strict,
    "thr_loose": thr_loose,
    "rank_ref": ref_ranks,
    "rank_ci_lo": rank_ci_lo,
    "rank_ci_hi": rank_ci_hi,
    "rank_width": rank_width,
    "stable_rank": stable_rank
}).sort_values("mean|SHAP|", ascending=False).reset_index(drop=True)


# ============================================================
# 7) Output
# ============================================================
print("Global rank-stability ρ (mean over bootstraps):", rho.mean().round(3))
print(summary.head(15))
print(thr_strict)
print(thr_loose)

summary.to_csv(feat_out_path, index=False)


