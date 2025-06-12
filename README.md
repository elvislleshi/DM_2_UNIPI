# Data Mining II Project - IMDb Movie Analysis  
Elvis Lleshi  
Academic Year: 2024/2025  

## 📚 Project Overview

This project is part of the *Data Mining II* course and implements a complete pipeline of advanced data mining techniques on IMDb movie datasets.  

It covers all required modules:

- **Module 0:** Data Understanding & Preparation  
- **Module 1:** Outlier Detection & Imbalanced Learning  
- **Module 2:** Advanced Classification, Advanced Regression & Explainability  
- **Module 3:** Time Series Analysis (Clustering, Motifs/Discords, Shapelets, Classification)

---

## 📂 Datasets

- `imdb.csv` — tabular metadata about movies (~150k records, 32 features)  
- `imdb_ts.csv` — time series dataset (daily revenue of movies, 100 days)

---

## ⚙️ Implemented Modules

### Module 0 — Data Understanding & Preparation
- Data cleaning, missing value treatment, duplicate removal
- Feature engineering (durationYears, main genre extraction)
- Correlation analysis & dimensionality reduction
- Time series normalization (MinMaxScaler)

### Module 1 — Outlier Detection & Imbalanced Learning
- **Outlier Detection:**  
  - Methods: Isolation Forest, Local Outlier Factor (LOF), KNN-based  
  - PCA visualization of outliers  
  - Before/After outlier removal comparison  
- **Imbalanced Learning:**  
  - Classification task: `isAdult`  
  - Techniques used: Random Oversampling, SMOTE, KNN-based Oversampling, Random Undersampling, Tomek Links, Cluster Centroids  
  - Models: Decision Tree, KNN  
  - Full evaluation of impact on class distribution and model performance

### Module 2 — Advanced Classification & Regression, Explainability
- **Advanced Classification:**  
  - Multi-class target: `rating`  
  - Classifiers used:
    - Logistic Regression
    - SVM (linear & non-linear)
    - Random Forest
    - Gradient Boosting, LightGBM, CatBoost, XGBoost
    - MLP Neural Network  
  - Hyperparameter tuning for all models  
  - Evaluation: Accuracy, Precision, Recall, F1-score, ROC curves  

- **Advanced Regression:**  
  - Target: `averageRating`  
  - Models used:
    - Random Forest Regressor
    - Gradient Boosting Regressor  
  - Metrics: MAE, RMSE, R²

- **Explainability:**  
  - Global: Surrogate Tree on LightGBM predictions  
  - Local: SHAP & LIME analysis on selected instance

### Module 3 — Time Series Analysis
- **Motifs/Discords Detection:** using STUMPY (Matrix Profile)  
- **Shapelets Discovery:** Shapelet-based classification  
- **Time Series Clustering:**  
  - Algorithms: K-Means, Hierarchical  
  - Distances: Euclidean, Dynamic Time Warping (DTW)  
  - Visualization: PCA, t-SNE  
- **Time Series Classification:**  
  - Models: KNN with Euclidean and DTW, Shapelet-based Classifier, RNN  

---

## 🏆 Key Takeaways
- Non-linear SVM outperformed other classifiers for multi-class classification.
- RNN performed well for time series classification, capturing sequential patterns.
- Gradient Boosting was the best regression model for predicting `averageRating`.
- SHAP & LIME offered valuable model interpretability.
- Time series motifs, discords, and shapelets provided insights into movie revenue dynamics.

---

## 🚀 Tech Stack
- Python (Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, Keras/TensorFlow, STUMPY, SHAP, LIME, t-SNE, PCA)

---

## 📜 Conclusion
This project demonstrates the full application of advanced data mining techniques on complex real-world datasets, combining tabular and time series data, and integrates model interpretability and performance optimization.

---

**Author:** Elvis Lleshi  
**Course:** Data Mining II (2024/2025)  


