
# MachineLearning

## 🧠 Feature Engineering and Exploratory Data Analysis (EDA)
- Missing Values Imputation  
- Handling Imbalanced Dataset  
- SMOTE  
- Handling Outliers  
- Encoding (Nominal, One-Hot, Label, Ordinal)

---

## 🔢 Simple and Multiple Linear Regression

| Model                         | Type              | Learning Technique        | Notes                                                                 |
|------------------------------|-------------------|---------------------------|-----------------------------------------------------------------------|
| Linear Regression (Custom)   | Simple Regression | Analytical (OLS)          | Fits line using closed-form solution; interpretable coefficients      |
| Multiple Linear Regression    | Multivariate       | Analytical (OLS)          | Supports multiple features; intercept & beta via matrix algebra       |
| Polynomial Linear Regression | Univariate/Multivariate | Analytical (OLS) | Transforms features into polynomial basis; then applies OLS          |
| Batch Gradient Descent       | Multivariate                | Iterative Optimization     | Uses all data per step; stable but slower on large datasets           |
| Stochastic Gradient Descent  | Multivariate                | Online Optimization        | Updates weights per sample; faster but noisy                         |
| Mini-batch Gradient Descent  | Multivariate                | Hybrid Optimization        | Compromise between BGD and SGD; faster convergence + smoother updates |

---

## 📉 Regularized Regression (L1, L2, Elastic Net)

| Model                  | Penalty Type | Learning Technique     | Notes                                                                 |
|------------------------|--------------|------------------------|-----------------------------------------------------------------------|
| Ridge Regression       | L2           | Analytical / Gradient  | Penalizes large weights; shrinks coefficients; doesn't zero them out |
| Lasso Regression       | L1           | Subgradient / Coordinate Descent | Encourages sparsity; can set coefficients exactly to zero (feature selection) |
| Elastic Net Regression | L1 + L2      | Coordinate Descent     | Combines benefits of Ridge and Lasso; balances sparsity and stability|

✅ **Use Cases**:
- Ridge: When all features are useful 
- Lasso: When only a few features are important (feature selection)  
- Elastic Net: When groups of correlated features are present or hybrid behavior is desired  
-------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧮 Logistic Regression

| Model                           | Activation | Learning Technique                | Notes                                                           |
|--------------------------------|------------|-----------------------------------|-----------------------------------------------------------------|
| Perceptron (Basic)             | Step       | Randomized update                 | Binary output (0 or 1), no probabilities                        |
| Perceptron (Sigmoid-based)     | Sigmoid    | Gradient-like updates             | Adds non-linearity, smooth decision making                      |
| Logistic Regression (Custom)   | Sigmoid    | MLE + Gradient Descent            | Trained using log loss; comparable to scikit-learn              |
| Multiclass Logistic Regression | Softmax    | Cross-Entropy + Gradient Descent  | Load_Digits dataset, NDVI Land Cover Classification task        |

## Decision Tree
## SVM
## Boosting and Bagging
## Random-Forest
## Beroulli and Gaussian Naive Bayes
## KNN 
# 🧠 UNSUPERVISED LEARNING
---

## Unsupervised Learning
- **Dimensionality Reduction**
  - Principal Component Analysis (PCA)
- **Clustering**
  - K-Means Clustering  
  - Hierarchical Clustering  
  - DBSCAN Clustering  
  - Silhouette Score for Cluster Evaluation
- **Anomaly Detection**
  - Isolation Forest  
  - DBSCAN Outlier Detection  
  - Local Outlier Factor (LOF)

---
