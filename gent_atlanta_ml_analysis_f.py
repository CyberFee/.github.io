"""
Atlanta Gentrification Machine Learning Analysis
Comprehensive ML pipeline including:
- Unsupervised Learning: K-Means and Hierarchical Clustering with PCA
- Supervised Classification: Random Forest, Gradient Boosting, Logistic Regression
- Supervised Regression: Predicting magnitude of demographic change and rent increase
- Complete evaluation metrics for all models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("ATLANTA GENTRIFICATION MACHINE LEARNING ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: DATA LOADING AND PREPARATION")
print("=" * 80)

# Load data
df = pd.read_csv('atlanta_metro_gentrification_with_changes.csv')
print(f"\nDataset loaded: {df.shape[0]} census tracts, {df.shape[1]} variables")

# ============================================================================
# FEATURE ENGINEERING FOR ML
# ============================================================================

print("\n--- Feature Selection for Machine Learning ---")

# Select features for ML analysis
# Using 2009 baseline features and change variables (2009-2019, 2000-2020)

baseline_features = [
    # 2009 Economic baseline
    '2009a_med_income', '2009a_med_rent', '2009a_med_home_value',

    # 2009 Demographics baseline
    '2009a_pct_renters', '2009a_pct_bachelors', '2009a_pct_graduate',
    '2009a_poverty_rate', '2009a_vacancy_rate', '2009a_cost_burden_rate',
    '2009a_pct_pre1980',

    # 2000 Demographics baseline
    '2000c_pct_black', '2000c_pct_white', '2000c_pct_asian', '2000c_pct_hispanic',
    '2000c_total_pop'
]

change_features = [
    # Demographic changes
    'change_pct_black_00_20', 'change_pct_white_00_20',
    'change_total_pop_00_20',
    'black_pop_change_pct', 'nonblack_pop_change_pct',
    'racial_replacement_index',

    # Economic changes
    'change_income_09_19', 'change_rent_09_19', 'change_home_value_09_19',

    # Education changes
    'change_bachelors_09_19', 'change_graduate_09_19',

    # Housing changes
    'change_pct_renters_09_19', 'change_vacancy_09_19', 'change_cost_burden_09_19'
]

# Combine all features
ml_features = baseline_features + change_features

# Create ML dataset
df_ml = df[ml_features + ['displacement_score', 'displacement_severity',
                          'gentrification_composite', 'classic_black_displacement',
                          'tract', 'county']].copy()

# Handle missing values
print(f"\nMissing values before imputation:\n{df_ml[ml_features].isnull().sum().sum()} total missing")

# Impute missing values with median for numerical features
for col in ml_features:
    if df_ml[col].isnull().sum() > 0:
        median_val = df_ml[col].median()
        df_ml[col].fillna(median_val, inplace=True)
        print(f"  Imputed {col} with median: {median_val:.2f}")

print(f"\nMissing values after imputation: {df_ml[ml_features].isnull().sum().sum()}")

# Remove any remaining rows with missing target variables
df_ml = df_ml.dropna(subset=['displacement_score', 'gentrification_composite', 'classic_black_displacement'])
print(f"\nFinal ML dataset: {df_ml.shape[0]} tracts, {len(ml_features)} features")

# Prepare feature matrix
X = df_ml[ml_features].values
feature_names = ml_features

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features used: {len(feature_names)}")

# ============================================================================
# PART 2: UNSUPERVISED LEARNING - CLUSTERING ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: UNSUPERVISED LEARNING - CLUSTERING ANALYSIS")
print("=" * 80)

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nFeatures standardized (mean=0, std=1)")

# ----------------------------------------------------------------------------
# 2.1: Principal Component Analysis (PCA)
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("2.1: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("-" * 80)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine number of components for 90% variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"\nPCA Results:")
print(f"  Total components: {len(pca.explained_variance_ratio_)}")
print(f"  Components for 90% variance: {n_components_90}")
print(f"  Components for 95% variance: {n_components_95}")
print(f"\nTop 10 components explain: {cumulative_variance[9]:.2%} of variance")

# Create PCA visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Scree plot
axes[0].bar(range(1, min(21, len(pca.explained_variance_ratio_) + 1)),
            pca.explained_variance_ratio_[:20], alpha=0.7, color='steelblue')
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].set_title('PCA Scree Plot - Top 20 Components', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             'o-', color='steelblue', linewidth=2, markersize=4)
axes[1].axhline(y=0.90, color='red', linestyle='--', label='90% variance', alpha=0.7)
axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95% variance', alpha=0.7)
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[1].set_title('Cumulative Variance Explained by PCA', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
print("\nPCA visualization saved: pca_analysis.png")

# Display top features for first 3 principal components
print("\n--- Top 5 Features for First 3 Principal Components ---")
for i in range(3):
    component = pca.components_[i]
    top_indices = np.argsort(np.abs(component))[-5:][::-1]
    print(f"\nPC{i + 1} (explains {pca.explained_variance_ratio_[i]:.2%} variance):")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {component[idx]:.3f}")

# Use reduced dimensions for clustering
X_pca_reduced = X_pca[:, :n_components_90]
print(f"\nUsing {n_components_90} components for clustering (90% variance)")

# ----------------------------------------------------------------------------
# 2.2: K-Means Clustering
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("2.2: K-MEANS CLUSTERING")
print("-" * 80)

# Determine optimal number of clusters using elbow method and silhouette score
inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("\nEvaluating different numbers of clusters...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca_reduced)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca_reduced, kmeans.labels_))
    print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

# Plot elbow curve and silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Elbow plot
axes[0].plot(K_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12)
axes[0].set_title('K-Means Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(K_range, silhouette_scores, 'o-', color='coral', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('K-Means Silhouette Scores', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_evaluation.png', dpi=300, bbox_inches='tight')
print("\nK-Means evaluation plots saved: kmeans_evaluation.png")

# Select optimal k (using silhouette score)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters (highest silhouette): k={optimal_k}")

# Fit final K-Means model
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_pca_reduced)

# Calculate clustering metrics
silhouette = silhouette_score(X_pca_reduced, kmeans_labels)
calinski = calinski_harabasz_score(X_pca_reduced, kmeans_labels)
davies = davies_bouldin_score(X_pca_reduced, kmeans_labels)

print(f"\n--- K-Means Clustering Metrics (k={optimal_k}) ---")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"  Interpretation: [-1 to 1], higher is better, >0.5 is good")
print(f"Calinski-Harabasz Index: {calinski:.2f}")
print(f"  Interpretation: Higher is better (more separated, compact clusters)")
print(f"Davies-Bouldin Index: {davies:.4f}")
print(f"  Interpretation: Lower is better (less similarity between clusters)")

# Cluster sizes
unique, counts = np.unique(kmeans_labels, return_counts=True)
print(f"\nCluster sizes:")
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} tracts ({count / len(kmeans_labels) * 100:.1f}%)")

# Add cluster labels to dataframe
df_ml['kmeans_cluster'] = kmeans_labels

# Analyze cluster characteristics
print("\n--- K-Means Cluster Characteristics ---")
cluster_stats = df_ml.groupby('kmeans_cluster')[
    ['change_pct_black_00_20', 'change_rent_09_19', 'change_income_09_19',
     'displacement_score', '2000c_pct_black', 'racial_replacement_index']
].mean()

print("\nMean values by cluster:")
print(cluster_stats.round(2))

# Visualize clusters in 2D PCA space
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels,
                     cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2,
           label='Cluster Centers')
ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title(f'K-Means Clustering (k={optimal_k}) in PCA Space', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig('kmeans_clusters_pca.png', dpi=300, bbox_inches='tight')
print("\nK-Means cluster visualization saved: kmeans_clusters_pca.png")

# ----------------------------------------------------------------------------
# 2.3: Hierarchical Clustering
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("2.3: HIERARCHICAL CLUSTERING")
print("-" * 80)

# Compute linkage matrix
print("\nComputing hierarchical clustering with Ward linkage...")
linkage_matrix = linkage(X_pca_reduced, method='ward')

# Create dendrogram
fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(linkage_matrix, ax=ax, no_labels=True, color_threshold=None)
ax.set_xlabel('Census Tract', fontsize=12)
ax.set_ylabel('Distance', fontsize=12)
ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Cutting threshold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
print("Dendrogram saved: hierarchical_dendrogram.png")

# Fit hierarchical clustering with optimal k from K-Means
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_pca_reduced)

# Calculate clustering metrics
silhouette_hier = silhouette_score(X_pca_reduced, hierarchical_labels)
calinski_hier = calinski_harabasz_score(X_pca_reduced, hierarchical_labels)
davies_hier = davies_bouldin_score(X_pca_reduced, hierarchical_labels)

print(f"\n--- Hierarchical Clustering Metrics (k={optimal_k}) ---")
print(f"Silhouette Score: {silhouette_hier:.4f}")
print(f"Calinski-Harabasz Index: {calinski_hier:.2f}")
print(f"Davies-Bouldin Index: {davies_hier:.4f}")

# Cluster sizes
unique_hier, counts_hier = np.unique(hierarchical_labels, return_counts=True)
print(f"\nCluster sizes:")
for cluster_id, count in zip(unique_hier, counts_hier):
    print(f"  Cluster {cluster_id}: {count} tracts ({count / len(hierarchical_labels) * 100:.1f}%)")

# Add hierarchical cluster labels
df_ml['hierarchical_cluster'] = hierarchical_labels

# Compare clustering methods
print("\n--- Comparison: K-Means vs Hierarchical ---")
print(f"K-Means Silhouette: {silhouette:.4f}")
print(f"Hierarchical Silhouette: {silhouette_hier:.4f}")
print(f"K-Means Calinski-Harabasz: {calinski:.2f}")
print(f"Hierarchical Calinski-Harabasz: {calinski_hier:.2f}")
print(f"K-Means Davies-Bouldin: {davies:.4f}")
print(f"Hierarchical Davies-Bouldin: {davies_hier:.4f}")

# Visualize hierarchical clusters
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels,
                     cmap='plasma', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title(f'Hierarchical Clustering (k={optimal_k}) in PCA Space', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig('hierarchical_clusters_pca.png', dpi=300, bbox_inches='tight')
print("Hierarchical cluster visualization saved: hierarchical_clusters_pca.png")

# ============================================================================
# PART 3: SUPERVISED LEARNING - CLASSIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: SUPERVISED LEARNING - CLASSIFICATION")
print("=" * 80)

# Prepare targets for classification
# Target 1: High displacement severity (binary)
df_ml['high_displacement'] = (df_ml['displacement_score'] > 40).astype(int)

# Check class distribution
print("\n--- Target Variable Distributions ---")
print(f"\nHigh Displacement (score > 40):")
print(df_ml['high_displacement'].value_counts())
print(f"  Class balance: {df_ml['high_displacement'].mean():.1%} positive class")

print(f"\nGentrification Composite:")
print(df_ml['gentrification_composite'].value_counts())
print(f"  Class balance: {df_ml['gentrification_composite'].mean():.1%} positive class")

print(f"\nClassic Black Displacement:")
print(df_ml['classic_black_displacement'].value_counts())
print(f"  Class balance: {df_ml['classic_black_displacement'].mean():.1%} positive class")

# ----------------------------------------------------------------------------
# 3.1: Classification - Predicting High Displacement
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("3.1: BINARY CLASSIFICATION - HIGH DISPLACEMENT PREDICTION")
print("-" * 80)

# Prepare data
X_class = df_ml[ml_features].values
y_class = df_ml['high_displacement'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train positive class: {y_train.mean():.1%}")
print(f"Test positive class: {y_test.mean():.1%}")

# Scale features
scaler_class = StandardScaler()
X_train_scaled = scaler_class.fit_transform(X_train)
X_test_scaled = scaler_class.transform(X_test)

# Dictionary to store results
classification_results = {}

# ----------------------------------------------------------------------------
# Model 1: Logistic Regression
# ----------------------------------------------------------------------------

print("\n" + "-" * 60)
print("Model 1: Logistic Regression")
print("-" * 60)

lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Evaluate
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"\n--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-Score: {f1_lr:.4f}")
print(f"ROC AUC: {roc_auc_lr:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Low/Moderate', 'High Displacement']))

print(f"\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)
print(f"  True Negatives: {cm_lr[0, 0]}")
print(f"  False Positives: {cm_lr[0, 1]}")
print(f"  False Negatives: {cm_lr[1, 0]}")
print(f"  True Positives: {cm_lr[1, 1]}")

# Store results
classification_results['Logistic Regression'] = {
    'accuracy': accuracy_lr, 'precision': precision_lr, 'recall': recall_lr,
    'f1': f1_lr, 'roc_auc': roc_auc_lr, 'y_pred': y_pred_lr,
    'y_pred_proba': y_pred_proba_lr, 'confusion_matrix': cm_lr
}

# Top features
feature_importance_lr = np.abs(lr.coef_[0])
top_features_idx_lr = np.argsort(feature_importance_lr)[-10:][::-1]
print(f"\nTop 10 Most Important Features:")
for idx in top_features_idx_lr:
    print(f"  {feature_names[idx]}: {lr.coef_[0][idx]:.4f}")

# ----------------------------------------------------------------------------
# Model 2: Random Forest
# ----------------------------------------------------------------------------

print("\n" + "-" * 60)
print("Model 2: Random Forest Classifier")
print("-" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\n--- Random Forest Results ---")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"ROC AUC: {roc_auc_rf:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Low/Moderate', 'High Displacement']))

print(f"\nConfusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

# Store results
classification_results['Random Forest'] = {
    'accuracy': accuracy_rf, 'precision': precision_rf, 'recall': recall_rf,
    'f1': f1_rf, 'roc_auc': roc_auc_rf, 'y_pred': y_pred_rf,
    'y_pred_proba': y_pred_proba_rf, 'confusion_matrix': cm_rf
}

# Feature importance
feature_importance_rf = rf.feature_importances_
top_features_idx_rf = np.argsort(feature_importance_rf)[-10:][::-1]
print(f"\nTop 10 Most Important Features:")
for idx in top_features_idx_rf:
    print(f"  {feature_names[idx]}: {feature_importance_rf[idx]:.4f}")

# ----------------------------------------------------------------------------
# Model 3: Gradient Boosting
# ----------------------------------------------------------------------------

print("\n" + "-" * 60)
print("Model 3: Gradient Boosting Classifier")
print("-" * 60)

gb = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
y_pred_proba_gb = gb.predict_proba(X_test_scaled)[:, 1]

# Evaluate
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb)

print(f"\n--- Gradient Boosting Results ---")
print(f"Accuracy: {accuracy_gb:.4f}")
print(f"Precision: {precision_gb:.4f}")
print(f"Recall: {recall_gb:.4f}")
print(f"F1-Score: {f1_gb:.4f}")
print(f"ROC AUC: {roc_auc_gb:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_gb, target_names=['Low/Moderate', 'High Displacement']))

print(f"\nConfusion Matrix:")
cm_gb = confusion_matrix(y_test, y_pred_gb)
print(cm_gb)

# Store results
classification_results['Gradient Boosting'] = {
    'accuracy': accuracy_gb, 'precision': precision_gb, 'recall': recall_gb,
    'f1': f1_gb, 'roc_auc': roc_auc_gb, 'y_pred': y_pred_gb,
    'y_pred_proba': y_pred_proba_gb, 'confusion_matrix': cm_gb
}

# Feature importance
feature_importance_gb = gb.feature_importances_
top_features_idx_gb = np.argsort(feature_importance_gb)[-10:][::-1]
print(f"\nTop 10 Most Important Features:")
for idx in top_features_idx_gb:
    print(f"  {feature_names[idx]}: {feature_importance_gb[idx]:.4f}")

# ----------------------------------------------------------------------------
# Classification Model Comparison
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("CLASSIFICATION MODEL COMPARISON")
print("-" * 80)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [accuracy_lr, accuracy_rf, accuracy_gb],
    'Precision': [precision_lr, precision_rf, precision_gb],
    'Recall': [recall_lr, recall_rf, recall_gb],
    'F1-Score': [f1_lr, f1_rf, f1_gb],
    'ROC AUC': [roc_auc_lr, roc_auc_rf, roc_auc_gb]
})

print("\n", comparison_df.to_string(index=False))

# Identify best model
best_model_idx = comparison_df['ROC AUC'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\nBest performing model (by ROC AUC): {best_model_name}")

# ----------------------------------------------------------------------------
# Visualizations for Classification
# ----------------------------------------------------------------------------

print("\n--- Creating Classification Visualizations ---")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ROC Curves
ax1 = fig.add_subplot(gs[0, :])
for model_name, results in classification_results.items():
    fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
    ax1.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={results['roc_auc']:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves - High Displacement Classification', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Confusion Matrices
cms = [cm_lr, cm_rf, cm_gb]
titles = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
for idx, (cm, title) in enumerate(zip(cms, titles)):
    ax = fig.add_subplot(gs[1, idx])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Predicted Low', 'Predicted High'],
                yticklabels=['Actual Low', 'Actual High'])
    ax.set_title(f'{title}\nConfusion Matrix', fontsize=11, fontweight='bold')

# Metrics Comparison Bar Charts
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
for idx, metric in enumerate(metrics):
    if idx < 3:
        ax = fig.add_subplot(gs[2, idx])
    else:
        # Create additional row for remaining metrics
        continue

    values = comparison_df[metric].values
    bars = ax.bar(comparison_df['Model'], values, color=['steelblue', 'coral', 'mediumseagreen'])
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Classification Model Performance - High Displacement Prediction',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
print("Classification results visualization saved: classification_results.png")

# Feature Importance Comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Logistic Regression coefficients
ax = axes[0]
top_lr = pd.DataFrame({
    'feature': [feature_names[i] for i in top_features_idx_lr],
    'importance': [np.abs(lr.coef_[0][i]) for i in top_features_idx_lr]
}).sort_values('importance', ascending=True)
ax.barh(top_lr['feature'], top_lr['importance'], color='steelblue')
ax.set_xlabel('Absolute Coefficient Value', fontsize=11)
ax.set_title('Logistic Regression\nTop 10 Features', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Random Forest importance
ax = axes[1]
top_rf_df = pd.DataFrame({
    'feature': [feature_names[i] for i in top_features_idx_rf],
    'importance': [feature_importance_rf[i] for i in top_features_idx_rf]
}).sort_values('importance', ascending=True)
ax.barh(top_rf_df['feature'], top_rf_df['importance'], color='coral')
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Random Forest\nTop 10 Features', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Gradient Boosting importance
ax = axes[2]
top_gb_df = pd.DataFrame({
    'feature': [feature_names[i] for i in top_features_idx_gb],
    'importance': [feature_importance_gb[i] for i in top_features_idx_gb]
}).sort_values('importance', ascending=True)
ax.barh(top_gb_df['feature'], top_gb_df['importance'], color='mediumseagreen')
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Gradient Boosting\nTop 10 Features', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_classification.png', dpi=300, bbox_inches='tight')
print("Feature importance visualization saved: feature_importance_classification.png")

# ============================================================================
# PART 4: SUPERVISED LEARNING - REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: SUPERVISED LEARNING - REGRESSION")
print("=" * 80)

# ----------------------------------------------------------------------------
# 4.1: Regression Target 1 - Predicting Displacement Score (Continuous)
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("4.1: REGRESSION - DISPLACEMENT SCORE PREDICTION")
print("-" * 80)

# Prepare data
X_reg1 = df_ml[ml_features].values
y_reg1 = df_ml['displacement_score'].values

# Train-test split
X_train_r1, X_test_r1, y_train_r1, y_test_r1 = train_test_split(
    X_reg1, y_reg1, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train_r1.shape[0]} samples")
print(f"Test set: {X_test_r1.shape[0]} samples")
print(f"Target range: [{y_reg1.min():.2f}, {y_reg1.max():.2f}]")
print(f"Target mean: {y_reg1.mean():.2f}, std: {y_reg1.std():.2f}")

# Scale features
scaler_r1 = StandardScaler()
X_train_r1_scaled = scaler_r1.fit_transform(X_train_r1)
X_test_r1_scaled = scaler_r1.transform(X_test_r1)

# Dictionary to store regression results
regression_results = {}

# Model 1: Ridge Regression (Linear)
print("\n" + "-" * 60)
print("Model 1: Ridge Regression")
print("-" * 60)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_r1_scaled, y_train_r1)
y_pred_ridge = ridge.predict(X_test_r1_scaled)

# Evaluate
mse_ridge = mean_squared_error(y_test_r1, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test_r1, y_pred_ridge)
r2_ridge = r2_score(y_test_r1, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)

print(f"\n--- Ridge Regression Results ---")
print(f"R² Score: {r2_ridge:.4f}")
print(f"Mean Squared Error (MSE): {mse_ridge:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_ridge:.4f}")
print(f"Mean Absolute Error (MAE): {mae_ridge:.4f}")

regression_results['Ridge Regression'] = {
    'r2': r2_ridge, 'mse': mse_ridge, 'rmse': rmse_ridge, 'mae': mae_ridge,
    'y_pred': y_pred_ridge
}

# Model 2: Random Forest Regressor
print("\n" + "-" * 60)
print("Model 2: Random Forest Regressor")
print("-" * 60)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_r1_scaled, y_train_r1)
y_pred_rf_reg = rf_reg.predict(X_test_r1_scaled)

# Evaluate
mse_rf_reg = mean_squared_error(y_test_r1, y_pred_rf_reg)
mae_rf_reg = mean_absolute_error(y_test_r1, y_pred_rf_reg)
r2_rf_reg = r2_score(y_test_r1, y_pred_rf_reg)
rmse_rf_reg = np.sqrt(mse_rf_reg)

print(f"\n--- Random Forest Regressor Results ---")
print(f"R² Score: {r2_rf_reg:.4f}")
print(f"Mean Squared Error (MSE): {mse_rf_reg:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf_reg:.4f}")
print(f"Mean Absolute Error (MAE): {mae_rf_reg:.4f}")

regression_results['Random Forest Reg'] = {
    'r2': r2_rf_reg, 'mse': mse_rf_reg, 'rmse': rmse_rf_reg, 'mae': mae_rf_reg,
    'y_pred': y_pred_rf_reg
}

# Feature importance
feature_importance_rf_reg = rf_reg.feature_importances_
top_features_idx_rf_reg = np.argsort(feature_importance_rf_reg)[-10:][::-1]
print(f"\nTop 10 Most Important Features:")
for idx in top_features_idx_rf_reg:
    print(f"  {feature_names[idx]}: {feature_importance_rf_reg[idx]:.4f}")

# Model 3: Gradient Boosting Regressor
print("\n" + "-" * 60)
print("Model 3: Gradient Boosting Regressor")
print("-" * 60)

gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb_reg.fit(X_train_r1_scaled, y_train_r1)
y_pred_gb_reg = gb_reg.predict(X_test_r1_scaled)

# Evaluate
mse_gb_reg = mean_squared_error(y_test_r1, y_pred_gb_reg)
mae_gb_reg = mean_absolute_error(y_test_r1, y_pred_gb_reg)
r2_gb_reg = r2_score(y_test_r1, y_pred_gb_reg)
rmse_gb_reg = np.sqrt(mse_gb_reg)

print(f"\n--- Gradient Boosting Regressor Results ---")
print(f"R² Score: {r2_gb_reg:.4f}")
print(f"Mean Squared Error (MSE): {mse_gb_reg:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_gb_reg:.4f}")
print(f"Mean Absolute Error (MAE): {mae_gb_reg:.4f}")

regression_results['Gradient Boosting Reg'] = {
    'r2': r2_gb_reg, 'mse': mse_gb_reg, 'rmse': rmse_gb_reg, 'mae': mae_gb_reg,
    'y_pred': y_pred_gb_reg
}

# Feature importance
feature_importance_gb_reg = gb_reg.feature_importances_
top_features_idx_gb_reg = np.argsort(feature_importance_gb_reg)[-10:][::-1]
print(f"\nTop 10 Most Important Features:")
for idx in top_features_idx_gb_reg:
    print(f"  {feature_names[idx]}: {feature_importance_gb_reg[idx]:.4f}")

# Regression Model Comparison
print("\n" + "-" * 80)
print("REGRESSION MODEL COMPARISON - DISPLACEMENT SCORE")
print("-" * 80)

reg_comparison_df = pd.DataFrame({
    'Model': ['Ridge Regression', 'Random Forest Reg', 'Gradient Boosting Reg'],
    'R² Score': [r2_ridge, r2_rf_reg, r2_gb_reg],
    'RMSE': [rmse_ridge, rmse_rf_reg, rmse_gb_reg],
    'MAE': [mae_ridge, mae_rf_reg, mae_gb_reg],
    'MSE': [mse_ridge, mse_rf_reg, mse_gb_reg]
})

print("\n", reg_comparison_df.to_string(index=False))

best_reg_idx = reg_comparison_df['R² Score'].idxmax()
best_reg_model = reg_comparison_df.loc[best_reg_idx, 'Model']
print(f"\nBest performing model (by R² Score): {best_reg_model}")

# ----------------------------------------------------------------------------
# 4.2: Regression Target 2 - Predicting Rent Change Percentage
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("4.2: REGRESSION - RENT CHANGE PERCENTAGE PREDICTION")
print("-" * 80)

# Prepare data (using baseline features to predict future rent change)
baseline_for_prediction = [
    '2009a_med_income', '2009a_med_rent', '2009a_med_home_value',
    '2009a_pct_renters', '2009a_pct_bachelors', '2009a_poverty_rate',
    '2009a_vacancy_rate', '2009a_cost_burden_rate',
    '2000c_pct_black', '2000c_pct_white', '2000c_total_pop'
]

X_reg2 = df_ml[baseline_for_prediction].values
y_reg2 = df_ml['change_rent_09_19'].values

# Remove any infinite or very large outliers
valid_mask = np.isfinite(y_reg2) & (np.abs(y_reg2) < 500)
X_reg2 = X_reg2[valid_mask]
y_reg2 = y_reg2[valid_mask]

# Train-test split
X_train_r2, X_test_r2, y_train_r2, y_test_r2 = train_test_split(
    X_reg2, y_reg2, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train_r2.shape[0]} samples")
print(f"Test set: {X_test_r2.shape[0]} samples")
print(f"Target range: [{y_reg2.min():.2f}%, {y_reg2.max():.2f}%]")
print(f"Target mean: {y_reg2.mean():.2f}%, std: {y_reg2.std():.2f}%")

# Scale features
scaler_r2 = StandardScaler()
X_train_r2_scaled = scaler_r2.fit_transform(X_train_r2)
X_test_r2_scaled = scaler_r2.transform(X_test_r2)

# Train models (using same three types)
print("\n--- Training Models for Rent Change Prediction ---")

# Ridge
ridge_r2 = Ridge(alpha=1.0, random_state=42)
ridge_r2.fit(X_train_r2_scaled, y_train_r2)
y_pred_ridge_r2 = ridge_r2.predict(X_test_r2_scaled)

r2_ridge_r2 = r2_score(y_test_r2, y_pred_ridge_r2)
rmse_ridge_r2 = np.sqrt(mean_squared_error(y_test_r2, y_pred_ridge_r2))
mae_ridge_r2 = mean_absolute_error(y_test_r2, y_pred_ridge_r2)

print(f"\nRidge Regression:")
print(f"  R² Score: {r2_ridge_r2:.4f}")
print(f"  RMSE: {rmse_ridge_r2:.2f}%")
print(f"  MAE: {mae_ridge_r2:.2f}%")

# Random Forest
rf_reg_r2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg_r2.fit(X_train_r2_scaled, y_train_r2)
y_pred_rf_r2 = rf_reg_r2.predict(X_test_r2_scaled)

r2_rf_r2 = r2_score(y_test_r2, y_pred_rf_r2)
rmse_rf_r2 = np.sqrt(mean_squared_error(y_test_r2, y_pred_rf_r2))
mae_rf_r2 = mean_absolute_error(y_test_r2, y_pred_rf_r2)

print(f"\nRandom Forest Regressor:")
print(f"  R² Score: {r2_rf_r2:.4f}")
print(f"  RMSE: {rmse_rf_r2:.2f}%")
print(f"  MAE: {mae_rf_r2:.2f}%")

# Gradient Boosting
gb_reg_r2 = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb_reg_r2.fit(X_train_r2_scaled, y_train_r2)
y_pred_gb_r2 = gb_reg_r2.predict(X_test_r2_scaled)

r2_gb_r2 = r2_score(y_test_r2, y_pred_gb_r2)
rmse_gb_r2 = np.sqrt(mean_squared_error(y_test_r2, y_pred_gb_r2))
mae_gb_r2 = mean_absolute_error(y_test_r2, y_pred_gb_r2)

print(f"\nGradient Boosting Regressor:")
print(f"  R² Score: {r2_gb_r2:.4f}")
print(f"  RMSE: {rmse_gb_r2:.2f}%")
print(f"  MAE: {mae_gb_r2:.2f}%")

# Comparison
print("\n" + "-" * 80)
print("REGRESSION MODEL COMPARISON - RENT CHANGE PREDICTION")
print("-" * 80)

rent_reg_comparison = pd.DataFrame({
    'Model': ['Ridge Regression', 'Random Forest Reg', 'Gradient Boosting Reg'],
    'R² Score': [r2_ridge_r2, r2_rf_r2, r2_gb_r2],
    'RMSE (%)': [rmse_ridge_r2, rmse_rf_r2, rmse_gb_r2],
    'MAE (%)': [mae_ridge_r2, mae_rf_r2, mae_gb_r2]
})

print("\n", rent_reg_comparison.to_string(index=False))

# ----------------------------------------------------------------------------
# Regression Visualizations
# ----------------------------------------------------------------------------

print(" Regression Visualizations")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Displacement Score predictions
models_disp = ['Ridge', 'Random Forest', 'Gradient Boosting']
predictions_disp = [y_pred_ridge, y_pred_rf_reg, y_pred_gb_reg]
r2_scores_disp = [r2_ridge, r2_rf_reg, r2_gb_reg]

for idx, (model_name, y_pred, r2) in enumerate(zip(models_disp, predictions_disp, r2_scores_disp)):
    ax = axes[0, idx]
    ax.scatter(y_test_r1, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax.plot([y_test_r1.min(), y_test_r1.max()],
            [y_test_r1.min(), y_test_r1.max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Displacement Score', fontsize=11)
    ax.set_ylabel('Predicted Displacement Score', fontsize=11)
    ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Rent Change predictions
predictions_rent = [y_pred_ridge_r2, y_pred_rf_r2, y_pred_gb_r2]
r2_scores_rent = [r2_ridge_r2, r2_rf_r2, r2_gb_r2]

for idx, (model_name, y_pred, r2) in enumerate(zip(models_disp, predictions_rent, r2_scores_rent)):
    ax = axes[1, idx]
    ax.scatter(y_test_r2, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax.plot([y_test_r2.min(), y_test_r2.max()],
            [y_test_r2.min(), y_test_r2.max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Rent Change (%)', fontsize=11)
    ax.set_ylabel('Predicted Rent Change (%)', fontsize=11)
    ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Regression Model Performance\nTop: Displacement Score | Bottom: Rent Change %',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
print("Regression results visualization saved: regression_results.png")

# ============================================================================
# PART 5: MODEL SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("PART 5:  MODEL SUMMARY")
print("=" * 80)

print("\n" + "-" * 80)
print("UNSUPERVISED LEARNING SUMMARY")
print("-" * 80)
print(f"\nPCA: {n_components_90} components explain 90% of variance")
print(f"K-Means Optimal Clusters: {optimal_k}")
print(f"  Silhouette Score: {silhouette:.4f}")
print(f"  Calinski-Harabasz: {calinski:.2f}")
print(f"  Davies-Bouldin: {davies:.4f}")
print(f"\nHierarchical Clustering:")
print(f"  Silhouette Score: {silhouette_hier:.4f}")
print(f"  Calinski-Harabasz: {calinski_hier:.2f}")
print(f"  Davies-Bouldin: {davies_hier:.4f}")

print("\n" + "-" * 80)
print("SUPERVISED CLASSIFICATION SUMMARY")
print("-" * 80)
print("\nBest Model:", best_model_name)
print("\nAll Models Performance:")
print(comparison_df.to_string(index=False))

print("\n" + "-" * 80)
print("SUPERVISED REGRESSION SUMMARY")
print("-" * 80)
print("\nDisplacement Score Prediction:")
print(reg_comparison_df.to_string(index=False))
print(f"\nBest Model: {best_reg_model}")

print("\nRent Change Prediction:")
print(rent_reg_comparison.to_string(index=False))

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("""
Based on the comprehensive ML analysis: (from Chat GPT)

1. CLUSTERING INSIGHTS:
   - Use identified clusters to understand different neighborhood typologies
   - Target policy interventions based on cluster characteristics
   - Monitor neighborhoods in high-risk clusters for early warning signs

2. CLASSIFICATION MODEL SELECTION:
   - Random Forest or Gradient Boosting recommended for production use
   - Balance precision/recall based on policy goals:
     * High precision: Fewer false alarms, targeted interventions
     * High recall: Catch more at-risk neighborhoods, broader support

3. REGRESSION MODEL APPLICATIONS:
   - Use displacement score predictions to prioritize resource allocation
   - Rent change predictions can inform affordability policy planning
   - Feature importance guides which neighborhood characteristics to monitor

4. NEXT STEPS:
   - Implement best models in production pipeline
   - Set up monitoring system for early warning indicators
   - Create interactive dashboards for policymakers and community organizations
   - Regular model retraining as new data becomes available

5. POLICY IMPLICATIONS:
   - Focus interventions on neighborhoods with:
     * High predicted displacement scores
     * Strong gentrification indicators
     * Vulnerable populations (high renters, low income)
   - Proactive rather than reactive policy approaches
""")

# Add predictions to dataframe
df_ml['predicted_displacement_score'] = np.nan
df_ml.loc[df_ml.index[valid_mask], 'predicted_displacement_score_test'] = np.nan

# Save enhanced dataset
df_ml.to_csv('atlanta_ml_results.csv', index=False)
print("ML results saved to: atlanta_ml_results.csv")

# Save model comparison results
comparison_df.to_csv('classification_model_comparison.csv', index=False)
reg_comparison_df.to_csv('regression_displacement_comparison.csv', index=False)
rent_reg_comparison.to_csv('regression_rent_comparison.csv', index=False)
print("Model comparison tables saved")

print("\n" + "=" * 80)
print("MACHINE LEARNING ANALYSIS OUTPUTS")
print("=" * 80)
print("\nGenerated outputs:")
print("  1. pca_analysis.png")
print("  2. kmeans_evaluation.png")
print("  3. kmeans_clusters_pca.png")
print("  4. hierarchical_dendrogram.png")
print("  5. hierarchical_clusters_pca.png")
print("  6. classification_results.png")
print("  7. feature_importance_classification.png")
print("  8. regression_results.png")
print("  9. atlanta_ml_results.csv")
print("  10. classification_model_comparison.csv")
print("  11. regression_displacement_comparison.csv")
print("  12. regression_rent_comparison.csv")