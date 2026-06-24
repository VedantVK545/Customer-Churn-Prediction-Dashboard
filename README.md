# 📊 Customer Churn Prediction Dashboard

  An interactive machine learning dashboard that predicts customer churn and
  provides actionable business insights.

  ## ✨ Features

  ### 📈 Overview
  - Dataset statistics, churn distribution visualization, feature summary

  ### 🔍 Data Explorer
  - Interactive filters (Churn, Contract, Internet Service)
  - Download filtered data as CSV
  - Box plots, distribution charts, correlation heatmaps

  ### 🤖 Model Training
  - **5 ML Models**: Logistic Regression, Random Forest, Gradient Boosting,
  XGBoost, SVM
  - **SMOTE** handling for imbalanced data
  - **Hyperparameter Tuning** via GridSearchCV

  ### 📊 Model Comparison
  - Side-by-side comparison across 5 metrics (Accuracy, Precision, Recall,
  F1, ROC-AUC)
  - Automatic best model selection

  ### 📈 Model Evaluation
  - Confusion Matrix, ROC Curve, Precision-Recall Curve
  - Feature Importance, Classification Report
  - **Export charts as PNG**

  ### 🔮 Predict
  - Single customer churn prediction with confidence scoring
  - **Batch Predictions**: Upload CSV for bulk predictions
  - Download prediction results

  ### 💼 Business ROI
  - Retention campaign ROI calculator
  - Cost-benefit analysis (Revenue saved vs lost)
  - Business recommendations

  ### 📂 Custom Dataset
  - Upload any CSV with binary target
  - Auto-preprocessing, auto-encoding
  - Train all models on custom data

  ## 🛠️ Tech Stack

  | Category | Technologies |
  |----------|--------------|
  | Language | Python 3.12 |
  | ML Framework | scikit-learn, XGBoost, imbalanced-learn |
  | Visualization | matplotlib, seaborn |
  | Dashboard | Streamlit |
  | Data | pandas, numpy |

  ## 🚀 Quick Start

  ```bash
  # Clone
  git clone <repo-url>
  cd churn-prediction-v2

  # Install dependencies
  pip install -r requirements.txt

  # Run
  streamlit run app.py
```

  📊 Dataset

  Uses the IBM Telco Customer Churn dataset (7043 customers, 20 features).

  🎯 Performance

  Best model achieves 85%+ ROC-AUC on test set.

  📄 License

  MIT
