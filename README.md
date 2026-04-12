Machine Learning in Python

A comprehensive, hands-on repository documenting a structured journey through core machine learning concepts, algorithms, and workflows — implemented from the ground up in clean, well-commented Python.


Overview
This repository serves as a living reference implementation of fundamental and intermediate machine learning concepts, built progressively through practical Python code. Rather than relying on black-box abstractions, every notebook and script in this repository prioritizes conceptual clarity — exposing the mechanics behind each algorithm, preprocessing decision, and evaluation strategy with explicit, readable implementations.
The codebase spans the full supervised and unsupervised learning landscape: from raw data ingestion and feature engineering through model training, hyperparameter tuning, performance benchmarking, and result visualization. Each topic is treated as a self-contained, reproducible module — making this repository equally useful as a learning resource, a technical interview reference, and a launchpad for applied ML projects.
All implementations are grounded in the Python scientific stack — NumPy, Pandas, scikit-learn, Matplotlib, and Seaborn — reflecting the tooling used across industry data science and machine learning engineering roles.

What This Repository Covers
Core Machine Learning Foundations

Supervised vs. unsupervised vs. semi-supervised learning paradigms
The bias-variance tradeoff and its practical implications for model selection
Overfitting, underfitting, and generalization — conceptual grounding with visual examples
The train/validation/test split methodology and why it matters
Cross-validation strategies — k-fold, stratified k-fold, leave-one-out

Data Preprocessing & Feature Engineering

Handling missing values — imputation strategies (mean, median, KNN, iterative)
Encoding categorical variables — label encoding, one-hot encoding, target encoding
Feature scaling — standardization (Z-score), min-max normalization, robust scaling
Outlier detection and treatment — IQR method, Z-score thresholding
Feature selection — variance thresholding, correlation analysis, recursive feature elimination (RFE)
Dimensionality reduction — Principal Component Analysis (PCA) with explained variance analysis
Building reproducible preprocessing pipelines with scikit-learn Pipeline and ColumnTransformer

Classification Algorithms

Logistic Regression — decision boundaries, regularization (L1/L2), class weighting
K-Nearest Neighbors (KNN) — distance metrics, optimal K selection via cross-validation
Decision Trees — splitting criteria (Gini impurity, entropy), pruning, depth control
Random Forest — ensemble mechanics, feature importance, out-of-bag error estimation
Support Vector Machines (SVM) — kernel trick, margin maximization, C and gamma tuning
Gradient Boosting — sequential ensemble learning, learning rate vs. tree depth tradeoffs
Naive Bayes — probabilistic classification, Gaussian and Multinomial variants

Regression Algorithms

Linear Regression — ordinary least squares, assumptions, residual analysis
Ridge & Lasso Regression — L2 and L1 regularization, coefficient shrinkage, feature selection via Lasso
Polynomial Regression — feature expansion, degree selection, overfitting risk
Decision Tree Regression — non-linear regression without explicit feature transformation
Random Forest Regression — variance reduction through bootstrap aggregation

Clustering Algorithms

K-Means Clustering — centroid initialization (k-means++), inertia, optimal K via elbow method and silhouette score
DBSCAN — density-based clustering, epsilon and min_samples tuning, noise point identification
Hierarchical Clustering — agglomerative linkage strategies, dendrogram interpretation
Cluster evaluation — silhouette coefficient, Davies-Bouldin index, Calinski-Harabasz score

Model Evaluation & Performance Metrics

Classification metrics — accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix interpretation
Regression metrics — MAE, MSE, RMSE, R² score, adjusted R²
Multiclass evaluation — macro vs. micro vs. weighted averaging
Calibration curves and probability threshold tuning
Learning curves — diagnosing underfitting and overfitting from training dynamics

Hyperparameter Optimization

Grid Search with cross-validation (GridSearchCV)
Randomized Search for high-dimensional parameter spaces (RandomizedSearchCV)
Understanding the parameter vs. hyperparameter distinction

Data Visualization with Seaborn & Matplotlib

Distribution analysis — histograms, KDE plots, box plots, violin plots
Relationship exploration — scatter plots, pair plots, joint plots
Correlation heatmaps for multivariate feature analysis
Classification boundary visualization
Feature importance bar charts and model comparison plots
Confusion matrix heatmaps with annotation


Tech Stack
LibraryRolePython 3.10+Core runtimeNumPyNumerical computation, array operations, linear algebraPandasData loading, manipulation, exploratory analysisscikit-learnML algorithms, preprocessing, pipelines, evaluationMatplotlibLow-level plotting, custom figure compositionSeabornStatistical visualization, high-level plotting APIJupyter NotebookInteractive experimentation and result documentation

Repository Structure
bashmachine-learning/
├── preprocessing/
│   ├── missing_values.py            # Imputation strategies — mean, median, KNN, iterative
│   ├── encoding.py                  # Label encoding, one-hot encoding, target encoding
│   ├── scaling.py                   # StandardScaler, MinMaxScaler, RobustScaler comparison
│   ├── feature_selection.py         # RFE, variance threshold, correlation-based selection
│   ├── pca_dimensionality.py        # PCA implementation with explained variance plotting
│   └── pipelines.py                 # End-to-end sklearn Pipeline + ColumnTransformer examples
│
├── classification/
│   ├── logistic_regression.py       # Binary and multiclass logistic regression with regularization
│   ├── knn_classifier.py            # KNN with optimal K selection via cross-validation
│   ├── decision_tree.py             # Decision tree training, pruning, and boundary visualization
│   ├── random_forest.py             # Ensemble training with feature importance analysis
│   ├── svm_classifier.py            # SVM with RBF kernel, C/gamma grid search
│   ├── gradient_boosting.py         # GBM with learning rate and depth tuning
│   └── naive_bayes.py               # Gaussian and Multinomial Naive Bayes variants
│
├── regression/
│   ├── linear_regression.py         # OLS regression with residual and assumption analysis
│   ├── ridge_lasso.py               # L1/L2 regularization paths and coefficient shrinkage
│   ├── polynomial_regression.py     # Feature expansion, degree selection, overfitting demo
│   ├── decision_tree_regression.py  # Non-linear regression with depth control
│   └── random_forest_regression.py  # Bootstrap aggregation for variance reduction
│
├── clustering/
│   ├── kmeans.py                    # K-Means with elbow method and silhouette analysis
│   ├── dbscan.py                    # Density-based clustering with noise point handling
│   ├── hierarchical.py              # Agglomerative clustering with dendrogram visualization
│   └── cluster_evaluation.py        # Silhouette score, Davies-Bouldin, Calinski-Harabasz
│
├── evaluation/
│   ├── classification_metrics.py    # Confusion matrix, precision, recall, F1, ROC-AUC
│   ├── regression_metrics.py        # MAE, MSE, RMSE, R², adjusted R²
│   ├── cross_validation.py          # K-fold, stratified k-fold, leave-one-out strategies
│   ├── learning_curves.py           # Bias-variance diagnosis from training dynamics
│   └── hyperparameter_tuning.py     # GridSearchCV and RandomizedSearchCV workflows
│
├── visualization/
│   ├── seaborn_basics.py            # Distribution plots, pair plots, categorical plots
│   ├── correlation_analysis.py      # Heatmaps, feature correlation matrices
│   ├── model_visualization.py       # Decision boundaries, feature importance, ROC curves
│   └── confusion_matrix_plot.py     # Annotated confusion matrix heatmaps
│
├── notebooks/
│   ├── end_to_end_classification.ipynb   # Full workflow — EDA → preprocessing → training → evaluation
│   ├── end_to_end_regression.ipynb       # Complete regression pipeline with residual analysis
│   └── clustering_exploration.ipynb      # Unsupervised learning with visual cluster analysis
│
├── requirements.txt
└── README.md

Getting Started
Prerequisites

Python 3.10 or higher
pip or conda package manager
Jupyter Notebook or JupyterLab (for interactive notebooks)

Installation
bash# Clone the repository
git clone https://github.com/Rajan-Upadhyay21/machine-learning.git
cd machine-learning

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter for interactive notebooks
jupyter notebook notebooks/

Sample Workflows
End-to-End Classification Pipeline
pythonimport pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('data/sample.csv')
X, y = df.drop('target', axis=1), df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline — preprocessing + model in one reproducible object
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validated training
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Final evaluation on held-out test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
Optimal K Selection for KNN
pythonfrom sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 31)
cv_scores = [
    cross_val_score(
        KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5
    ).mean()
    for k in k_range
]

plt.figure(figsize=(10, 5))
plt.plot(k_range, cv_scores, marker='o', linewidth=2)
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Optimal K Selection via Cross-Validation')
plt.grid(True, alpha=0.3)
plt.show()

optimal_k = k_range[cv_scores.index(max(cv_scores))]
print(f"Optimal K: {optimal_k} — CV Accuracy: {max(cv_scores):.4f}")
K-Means Elbow Method
pythonfrom sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia_values = []
k_range = range(2, 12)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(k_range, inertia_values, marker='o', linewidth=2)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method — Optimal K Selection')
plt.grid(True, alpha=0.3)
plt.show()

Key Concepts Reference
ConceptOne-Line DefinitionBias-Variance TradeoffHigh bias → underfitting; high variance → overfitting; the goal is the optimal balanceCross-ValidationRotating train/val splits to produce a reliable estimate of generalization performanceRegularizationPenalizing model complexity to prevent overfitting — L1 (Lasso) promotes sparsity, L2 (Ridge) shrinks weightsFeature ScalingNormalizing feature magnitudes so distance-based and gradient-based algorithms converge correctlyEnsemble LearningCombining multiple weak learners (trees) into a strong predictor through bagging or boostingSilhouette ScoreMeasures how well-separated clusters are — ranges from -1 (poor) to +1 (ideal)ROC-AUCArea under the receiver operating characteristic curve — model discrimination ability across all thresholdsStratified SplitPreserving class proportions in train/test splits — critical for imbalanced datasetsPipelineChaining preprocessing and modeling steps into a single, leak-proof, reproducible objectFeature ImportanceA Random Forest's measure of how much each feature reduces impurity across all trees

Learning Progression
This repository is structured to support a deliberate, bottom-up learning progression:
Stage 1 — Foundations
  └── Python scientific stack (NumPy, Pandas, Matplotlib)
  └── Data loading, EDA, and descriptive statistics

Stage 2 — Preprocessing
  └── Missing values, encoding, scaling, feature selection
  └── Building sklearn pipelines

Stage 3 — Supervised Learning
  └── Classification algorithms (Logistic → Tree → Ensemble → SVM)
  └── Regression algorithms (Linear → Regularized → Tree-based)
  └── Model evaluation and cross-validation

Stage 4 — Unsupervised Learning
  └── Clustering (K-Means, DBSCAN, Hierarchical)
  └── Dimensionality reduction (PCA)

Stage 5 — Model Optimization
  └── Hyperparameter tuning (GridSearch, RandomSearch)
  └── Learning curve analysis
  └── Performance benchmarking across algorithms

Roadmap

 Add XGBoost and LightGBM implementations with SHAP-based interpretability
 Imbalanced classification — SMOTE, class weighting, threshold optimization
 Time-series forecasting fundamentals — ARIMA, feature engineering for temporal data
 Anomaly detection — Isolation Forest, One-Class SVM, Local Outlier Factor
 Model persistence — joblib serialization and deployment-ready export
 Automated ML (AutoML) exploration with TPOT or auto-sklearn
 Interactive Jupyter widgets for real-time hyperparameter experimentation


Author
Rajan M Upadhyay
MS Computer Science — Roosevelt University
LinkedIn · GitHub · rajanupadhyay2121@gmail.com
