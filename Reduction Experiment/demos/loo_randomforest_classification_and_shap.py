import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import KNNImputer
from openpyxl import load_workbook
import shap
import seaborn as sns
import joblib

# specify if you want to save plots automatically, plot will be saved automatically after you close its pop-up window
save_plots = 1
# specify if there are missing feature values for some samples
miss_values = 0

# Load the excel file
input_path = 'Reduction Experiment/binary_classification_sets_and_results/Easy_vs_Hard_MIRCs/MIRCs_Easy_Hard_sample.xlsx'      # Replace with path to your downloaded excel file with samples and features for classification
df = pd.read_excel(input_path, sheet_name='Easy')

# set output paths
out_path = '.../results.xlsx'             # Replace with .xlsx path to where you want the classification performance metrics and SHAP analysis to be written
plots_dir = '.../...'                     # Replace with path to were you want SHAP plots to be saved automatically
model_out_path = plots_dir + '/random_forest_model.pkl'
# Initialize an empty DataFrame if it doesn't exist yet
results_df = pd.DataFrame()

# Specify columns to keep (features to use)
columns_to_keep = ['active_object', 'active_hand', 'contextual_objects', 'background', 'salient_colour', 'salient_dklcolour', 'salient_flicker', 'salient_intensity', 'salient_motion', 'salient_orientation', 'salient_contrast']

# Identify skewed features for log transformation
skewed_features = ['active_object', 'active_hand', 'contextual_objects', 'background', 'salient_colour', 'salient_dklcolour', 'salient_flicker', 'salient_intensity', 'salient_motion', 'salient_orientation', 'salient_contrast']  # Replace with your actual skewed feature names

# Apply log transformation to skewed features
for feature in skewed_features:
    df[feature] = np.log1p(df[feature])  # Use log1p to handle zero values (log1p(x) = log(x + 1))


# Specify the label column
y = df['MIRC']
# Prepare the data
X = df[columns_to_keep]
# Loop through all columns and check their data type, gather, dummy encode categorical features
categorical_columns_multi = []
for column in X.columns:
    print(column)
    if (X[column].dtype == 'object' or X[column].dtype.name == 'category') and column not in ['MIRC']:
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

# Apply one-hot encoding to categorical columns with more than 2 levels
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

# Initialize LOOCV
loo = LeaveOneOut()
# Initialize the RandomForestClassifier with the specified hyperparameters
model = RandomForestClassifier(
    n_estimators=500,          # Number of decision trees - Soydaner uses 150, more should only increase computational demands while maybe increasing performance
    criterion='gini',          # Soydaner uses MSE for a regression task
    bootstrap=True,            # Use bootstrap samples
    max_depth=None,            # No explicit maximum depth
    min_samples_split=2,       # Minimum number of samples required to split an internal node
    min_samples_leaf=1,        # Minimum number of samples required to be at a leaf node
    max_features='sqrt'        # Number of features considered for the best split (use square root of features)
)
# Initialize lists to store results
all_accuracies = []
all_f1_scores = []
all_precisions = []
all_recalls = []
all_feature_importances_easy = np.zeros((10, X.shape[1]))
all_feature_importances = np.zeros((10, X.shape[1]))
all_interaction_importances = []
all_interaction_importances_easy = []
all_shap_values_array = np.zeros((X.shape[0], X.shape[1]))
all_shap_values_array_easy = np.zeros((X.shape[0], X.shape[1]))
# Initialize lists for class-specific and difficulty-specific accuracies
class_0_accuracies = []
class_1_accuracies = []
# Perform LOOCV 10 times
total_runs = 1
start_time = time.time()
for run in range(total_runs):
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    feature_importances = np.zeros(X.shape[1])
    class_0_accuracy = []
    class_1_accuracy = []
    # Initialize an array to accumulate interaction importances for this run
    interaction_importances_run = []
    shap_values_run = np.zeros((X.shape[0], X.shape[1]))
    # Accumulate predictions and true labels
    all_y_true = []
    all_y_pred = []
    # Loop over leave-one-out splits
    for train_index, test_index in loo.split(X):
        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train the model
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        # Accumulate true and predicted values
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        # SHAP feature importances for both class 0 and class 1
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)  # shap_values contains SHAP values for both classes in one array
        # Extract SHAP values for class 0 and class 1 from the same array
        shap_values_class_0 = shap_values[0][:, 0]  # SHAP values for class 0
        shap_values_class_1 = shap_values[0][:, 1]  # SHAP values for class 1
        # Class-specific accuracies
        accuracy = accuracy_score(y_test, y_pred)
        if y_test.values[0] == 0:  # Class 0 case
            class_0_accuracy.append(accuracy)
            # SHAP feature importance for class 0
            shap_importance_0 = np.abs(shap_values_class_0)  # Class 0 SHAP values
            feature_importances += shap_importance_0  # Accumulate SHAP importances for class 0
            # SHAP interaction values for class 0
            shap_interaction_values = explainer.shap_interaction_values(X_test)
            shap_interaction_class_0 = shap_interaction_values[0][:, :, 0]  # Class 0 interaction
            np.fill_diagonal(shap_interaction_class_0, 0)
            interaction_importances_run.append(shap_interaction_class_0)
            # Collect SHAP values for class 0 for dependence plot
            shap_values_run[test_index, :] = shap_values_class_0
        else:  # Class 1 case
            class_1_accuracy.append(accuracy)
            # SHAP feature importance for class 1
            shap_importance_1 = np.abs(shap_values_class_1)  # Class 1 SHAP values
            feature_importances += shap_importance_1  # Accumulate SHAP importances for class 1
            # SHAP interaction values for class 1
            shap_interaction_values = explainer.shap_interaction_values(X_test)
            shap_interaction_class_1 = shap_interaction_values[0][:, :, 1]  # Class 1 interaction
            np.fill_diagonal(shap_interaction_class_1, 0)
            interaction_importances_run.append(shap_interaction_class_1)
            # Collect SHAP values for class 1 for dependence plot
            shap_values_run[test_index, :] = shap_values_class_1
    # Calculate the mean accuracy and F1-score for this run
    mean_accuracy = accuracy_score(all_y_true, all_y_pred)
    mean_f1 = f1_score(all_y_true, all_y_pred, zero_division=1)  # Set to 1 to avoid undefined metric warning
    # Calculate precision and recall for this run
    mean_precision = precision_score(all_y_true, all_y_pred, zero_division=1)
    mean_recall = recall_score(all_y_true, all_y_pred, zero_division=1)
    all_accuracies.append(mean_accuracy)
    all_f1_scores.append(mean_f1)
    all_precisions.append(mean_precision)
    all_recalls.append(mean_recall)
    # Store class-specific and difficulty-specific accuracies
    class_0_accuracies.append(np.mean(class_0_accuracy))
    class_1_accuracies.append(np.mean(class_1_accuracy))
    # Average SHAP feature importances for this run
    feature_importances /= loo.get_n_splits(X)
    all_feature_importances[run] = feature_importances
    # Aggregate interaction importances for this run
    # After completing all LOO splits for the current run, average interaction importances across splits
    mean_interaction_importances_run = np.mean(interaction_importances_run, axis=0)
    # Append the result to the list of all runs
    all_interaction_importances.append(mean_interaction_importances_run)
    # After completing LOOCV for this run, aggregate SHAP values across all splits
    all_shap_values_array += shap_values_run
    # Calculate elapsed time and estimate remaining time
    elapsed_time = time.time() - start_time
    avg_time_per_run = elapsed_time / (run + 1)
    remaining_runs = total_runs - (run + 1)
    estimated_remaining_time = remaining_runs * avg_time_per_run
    print(f"Run {run + 1}/{total_runs} completed.")
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")
    print(f"Estimated time remaining for this condition: {estimated_remaining_time:.2f} seconds.")


# Save the model to a file (model based on the last loocv iteration)
joblib.dump(model, model_out_path) # to load use: model = joblib.load('random_forest_model.pkl')
# Calculate overall mean accuracy, precision, recall, F1-score
overall_mean_accuracy = np.mean(all_accuracies)
overall_std_accuracy = np.std(all_accuracies)
overall_mean_f1 = np.mean(all_f1_scores)
overall_std_f1 = np.std(all_f1_scores)
overall_mean_precision = np.nanmean(all_precisions)
overall_std_precision = np.nanstd(all_precisions)
overall_mean_recall = np.nanmean(all_recalls)
overall_std_recall = np.nanstd(all_recalls)
# Calculate overall mean and standard deviation for class-specific accuracies
overall_mean_class_0_accuracy = np.mean(class_0_accuracies)
overall_std_class_0_accuracy = np.std(class_0_accuracies)
overall_mean_class_1_accuracy = np.mean(class_1_accuracies)
overall_std_class_1_accuracy = np.std(class_1_accuracies)
# Append the results to the results DataFrame, including both means and standard deviations
new_row = pd.DataFrame([{
    'mean_accuracy': overall_mean_accuracy,
    'std_accuracy': overall_std_accuracy,
    'mean_f1': overall_mean_f1,
    'std_f1': overall_std_f1,
    'mean_precision': overall_mean_precision,
    'std_precision': overall_std_precision,
    'mean_recall': overall_mean_recall,
    'std_recall': overall_std_recall,
    'mean_not_rec_accuracy': overall_mean_class_0_accuracy,
    'std_not_rec_accuracy': overall_std_class_0_accuracy,
    'mean_rec_accuracy': overall_mean_class_1_accuracy,
    'std_rec_accuracy': overall_std_class_1_accuracy,
    'all_y_true': all_y_true,
    'all_y_pred': all_y_pred,
}])
# Concatenate the new row to the results DataFrame
results_df = pd.concat([results_df, new_row], ignore_index=True)
print(f"Mean accuracy: {mean_accuracy:.2f}.")
print(f"Mean F1-score: {mean_f1:.2f}.")
# Write the results DataFrame to an .xlsx file
with pd.ExcelWriter(out_path, engine='openpyxl', mode='w') as writer:
    results_df.to_excel(writer, sheet_name='results', index=False)


# Plot and save feature importances
# calculate overall feature importances
mean_feature_importances = np.mean(all_feature_importances, axis=0)
std_feature_importances = np.std(all_feature_importances, axis=0)
feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_feature_importances,
    'SD': std_feature_importances
}).sort_values(by='Importance', ascending=False)
# Load the existing Excel file
with pd.ExcelWriter(out_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
    # Write the DataFrame to a new sheet
    feature_importances_df.to_excel(writer, sheet_name='feature_importances', index=False)

# Plot feature importances
sorted_indices = np.argsort(mean_feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), mean_feature_importances[sorted_indices], align='center', alpha=0.6,
        label='Model Importances')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.legend()
# Save plot if required
if save_plots == 1:
    plt.savefig(os.path.join(plots_dir, 'feature_importances_all.png'))

plt.show()


# shap summary plot for all items
# average shap values across all runs
all_shap_values_array /= total_runs
shap_df = pd.DataFrame(all_shap_values_array, columns=X.columns)
# Open the Excel file and write the SHAP values to a new sheet
with pd.ExcelWriter(out_path, engine='openpyxl', mode='a') as writer:
    shap_df.to_excel(writer, sheet_name='SHAP_Values')

# Plot and save SHAP summary plot for all items
shap.summary_plot(all_shap_values_array, X, plot_type="dot", show=False)  # For class 0 in binary classification
if save_plots == 1:
    plt.savefig(os.path.join(plots_dir, 'shap_summary_all.png'))

plt.show()

# Calculate overall interaction importances
# After all runs, aggregate the interaction importances across all runs
mean_interaction_importances_all = np.mean(all_interaction_importances, axis=0)
# Ensure that the interaction matrix has the correct shape: (n_features, n_features)
assert mean_interaction_importances_all.shape == (X.shape[1], X.shape[1]), \
    f"Expected shape {(X.shape[1], X.shape[1])}, but got {mean_interaction_importances_all.shape}"
# Convert to a DataFrame for better visualization (n_features, n_features)
interaction_df = pd.DataFrame(mean_interaction_importances_all, index=X.columns, columns=X.columns)
# Convert the interaction matrix to a ranked list of interactions
interaction_rankings = interaction_df.unstack().reset_index()
interaction_rankings.columns = ['Feature 1', 'Feature 2', 'Interaction Importance']
# Remove self-interactions (diagonal elements) where Feature 1 == Feature 2
interaction_rankings = interaction_rankings[interaction_rankings['Feature 1'] != interaction_rankings['Feature 2']]
# Sort by Interaction Importance in descending order
interaction_rankings = interaction_rankings.sort_values(by='Interaction Importance', ascending=False)
# Write interaction importances to the Excel file
# Load the existing Excel file
with pd.ExcelWriter(out_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
    # Write the DataFrame to a new sheet
    interaction_rankings.to_excel(writer, sheet_name='interaction_importances', index=False)

# plot actual interaction importances
# Set up a figure with larger size and white background
plt.figure(figsize=(12, 10))  # Adjusted size for better label visibility
plt.gcf().set_facecolor('white')  # Set background to white
# Plot the heatmap with padding to allow for better label visibility
sns.heatmap(interaction_df, annot=False, cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": 0.8})
# Add title
plt.title("SHAP Interaction Importance Heatmap", fontsize=16)
# Adjust the layout to ensure labels are not cut off
plt.tight_layout(pad=2.0)
# Save the plot with larger area for labels
if save_plots == 1:
    plt.savefig(os.path.join(plots_dir, 'shap_interaction_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')

# Show the plot
plt.show()
# plot absolute interaction importances
# get rid of minus
mean_interaction_importances_all_abs = np.abs(mean_interaction_importances_all)
# Convert to a DataFrame for better visualization (n_features, n_features)
interaction_df_abs = pd.DataFrame(mean_interaction_importances_all_abs, index=X.columns, columns=X.columns)
# Set up a figure with larger size and white background
plt.figure(figsize=(12, 10))  # Adjusted size for better label visibility
plt.gcf().set_facecolor('white')  # Set background to white
# Plot the heatmap with padding to allow for better label visibility
sns.heatmap(interaction_df_abs, annot=False, cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": 0.8})
# Add title
plt.title("SHAP Interaction Importance Heatmap", fontsize=16)
# Adjust the layout to ensure labels are not cut off
plt.tight_layout(pad=2.0)
# Save the plot with larger area for labels
if save_plots == 1:
    plt.savefig(os.path.join(plots_dir, 'shap_interaction_heatmap_abs.png'), dpi=300, bbox_inches='tight', facecolor='white')

# Show the plot
plt.show()


# Plot a dependence plot to examine specific interaction
# Now plot the dependence plot using the accumulated SHAP values
shap.dependence_plot('active hand', all_shap_values_array, X, interaction_index='contextual object') # can use all_shap_values_array_easy or all_shap_values_array_hard or all_shap_values_array

shap.dependence_plot('background', all_shap_values_array, X, interaction_index='active object') # can use all_shap_values_array_easy or all_shap_values_array_hard or all_shap_values_array

