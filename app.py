"""
Customer Churn Prediction Dashboard
Features: Multiple ML Models, ROC Curves, Business ROI, Advanced Filters, XGBoost, Batch Predictions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📊")

# ==================== CACHE FUNCTIONS ====================
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

@st.cache_data
def preprocess(df):
    df = df.copy()
    # Handle TotalCharges - convert to numeric and fill missing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill TotalCharges with 0 for rows with empty string, then median for rest
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    # Encode binary columns
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    # Encode service columns
    df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    # Encode ordinal columns
    df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0, 'Mailed check': 1,
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
    })
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    # Drop any remaining rows with NaN values
    df = df.dropna()
    return df

# ==================== PLOT HELPERS ====================
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_roc_curve(y_true, y_proba, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', color='blue', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_precision_recall(y_true, y_proba, model_name="Model"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = np.mean(precision[:-1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', color='green', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(feat_imp)), feat_imp['importance'], color='steelblue')
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features')
    return fig

def plot_model_comparison(results_df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics)))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        model_names = results_df['Model'].str[:15]
        bars = ax.bar(model_names, results_df[metric], color=colors[idx])
        ax.set_ylabel('Score')
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ==================== MODEL TRAINING ====================
MODELS = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(probability=True, random_state=42, kernel='rbf')
}

# Hyperparameter grids for tuning
PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
}

def tune_model(X_train, y_train, model_name):
    """Run GridSearchCV for hyperparameter tuning."""
    if model_name not in PARAM_GRIDS:
        return None
    model = MODELS[model_name]
    param_grid = PARAM_GRIDS[model_name]
    grid = GridSearchCV(
        model, param_grid, cv=3,
        scoring='roc_auc', n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ==================== SUMMARY REPORT ====================
def generate_summary_report(df_stats, best_model_name, best_auc, top_features, roi):
    """Generate a text summary report of the analysis."""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║           CUSTOMER CHURN ANALYSIS SUMMARY REPORT         ║
╚══════════════════════════════════════════════════════════╝

══════════════ DATASET OVERVIEW ══════════════
• Total Customers:     {df_stats.get('total', 'N/A'):,}
• Churn Rate:          {df_stats.get('churn_rate', 'N/A')}%
• Churned Customers:   {df_stats.get('churned', 'N/A'):,}
• Retained Customers:  {df_stats.get('retained', 'N/A'):,}

══════════════ MODEL PERFORMANCE ══════════════
• Best Model:          {best_model_name}
• ROC-AUC Score:       {best_auc}

══════════════ TOP FEATURES ══════════════
{chr(10).join(f'  {i+1}. {feat}' for i, feat in enumerate(top_features))}

══════════════ BUSINESS IMPACT ══════════════
• ROI:                 {roi}%
• Analysis Period:     {pd.Timestamp.now().strftime('%Y-%m-%d')}

Generated by Customer Churn Prediction Dashboard
    """
    return report

# ==================== MAIN APP ====================
def main():
    # ==================== MODERN CSS ====================
    st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', -apple-system, sans-serif; }

    /* Main container */
    .main > div { padding: 0 1rem; }

    /* Header - always dark gradient */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.15);
    }
    .app-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Tabs - clean button style */
    .stTabs [role="tab"] {
        background: var(--background-color, white);
        border-radius: 12px !important;
        padding: 0.6rem 1.2rem !important;
        margin: 0 0.3rem !important;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-color, #64748b);
        border: 1px solid var(--border-color, #e2e8f0);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    .stTabs [role="tab"]:hover {
        background: var(--secondary-background-color, #f1f5f9);
        border-color: var(--border-color, #cbd5e1);
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #1e3a5f !important;
        color: white !important;
        border-color: #1e3a5f !important;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.2);
    }
    .stTabs [role="tablist"] {
        gap: 0.5rem;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: var(--background-color, white);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color, #f1f5f9);
        transition: transform 0.15s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    [data-testid="metric-container"] label {
        font-weight: 500;
        font-size: 0.8rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.6rem !important;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color, #e2e8f0);
    }
    .stDataFrame thead tr th {
        background: var(--secondary-background-color, #f8fafc) !important;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 0.8rem 1rem !important;
    }
    .stDataFrame tbody tr td {
        padding: 0.6rem 1rem !important;
        font-size: 0.88rem;
    }
    .stDataFrame tbody tr:hover {
        background: var(--secondary-background-color, #f8fafc) !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: all 0.2s;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%) !important;
        color: white !important;
    }

    /* Download buttons */
    .stDownloadButton button {
        border-radius: 10px !important;
        font-weight: 500;
        padding: 0.4rem 1.2rem;
    }

    /* Headers inside tabs */
    h2 {
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
        letter-spacing: -0.3px;
    }
    h3 {
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* Expander */
    .st-emotion-cache-1yc7g1p {
        border: 1px solid var(--border-color, #e2e8f0);
        border-radius: 12px;
        background: var(--background-color, white);
    }

    /* Select boxes */
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    /* Success/Warning/Error messages */
    .stAlert {
        border-radius: 12px;
        border: none;
    }

    /* Progress bar */
    .stProgress > div > div {
        border-radius: 10px;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 10px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 1.5rem;
        font-size: 0.85rem;
        opacity: 0.6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>📊 Customer Churn Prediction Dashboard</h1>
        <p>Predict Customer Churn with Machine Learning &nbsp;·&nbsp; Business ROI Analysis &nbsp;·&nbsp; Model Comparison</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df_raw = load_data()
        df = preprocess(df_raw.copy())
        st.sidebar.success("✅ Dataset loaded!")
    except FileNotFoundError:
        st.error("Dataset not found. Please place the CSV file in data/ folder")
        return

    # Sidebar
    st.sidebar.header("📊 Quick Stats")
    st.sidebar.metric("Total Customers", f"{len(df):,}")
    st.sidebar.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    st.sidebar.metric("Churned", f"{int(df['Churn'].sum()):,}")
    st.sidebar.metric("Retained", f"{int(len(df) - df['Churn'].sum()):,}")

    # Tabs
    tabs = st.tabs([
        "📈 Overview",
        "🔍 Data Explorer",
        "🤖 Model Training",
        "📊 Model Comparison",
        "📈 Model Evaluation",
        "🔮 Predict",
        "💼 Business ROI",
        "📂 Custom Dataset"
    ])

    # ========== TAB 1: OVERVIEW ==========
    with tabs[0]:
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Churned", f"{int(df['Churn'].sum()):,}")
        col3.metric("Retention Rate", f"{(1 - df['Churn'].mean())*100:.1f}%")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(df['Churn'].value_counts(), labels=['Not Churn', 'Churn'],
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0.05))
            ax.set_title('Customer Churn Distribution')
            st.image(plot_to_img(fig))
        with col2:
            st.subheader("Dataset Summary")
            st.write(f"**Total Rows:** {len(df):,}")
            st.write(f"**Features:** {len(df.columns)}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")
            st.markdown("**Feature Categories:**")
            st.write("- **Demographics:** gender, SeniorCitizen, Partner, Dependents")
            st.write("- **Services:** PhoneService, InternetService, OnlineSecurity, etc.")
            st.write("- **Financial:** MonthlyCharges, TotalCharges, Contract, PaymentMethod")

            # Download Summary Report (only if models are trained)
            if st.session_state.get('results_df') is not None:
                st.markdown("---")
                st.subheader("📋 Summary Report")
                best = st.session_state['results_df'].iloc[0]
                imp_df = st.session_state.get('importance_df', pd.DataFrame())
                top_feats = imp_df.head(5)['feature'].tolist() if 'feature' in imp_df.columns else ['N/A']
                roi_val = st.session_state.get('roi_value', 0)
                report_text = generate_summary_report(
                    {'total': len(df), 'churn_rate': f"{df['Churn'].mean()*100:.1f}",
                     'churned': int(df['Churn'].sum()), 'retained': int(len(df) - df['Churn'].sum())},
                    best['Model'], best['ROC-AUC'], top_feats, roi_val
                )
                st.download_button("📥 Download Summary Report (.txt)", report_text, "churn_report.txt", "text/plain")

    # ========== TAB 2: DATA EXPLORER ==========
    with tabs[1]:
        st.header("🔍 Data Explorer")
        st.markdown("Explore the data with interactive filters and visualizations")

        # Advanced Filters
        st.subheader("🎯 Advanced Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_churn = st.multiselect("Filter by Churn", ["Yes", "No"], default=["Yes", "No"])
        with col2:
            filter_contract = st.multiselect("Filter by Contract", df_raw['Contract'].unique().tolist(), default=df_raw['Contract'].unique().tolist())
        with col3:
            filter_internet = st.multiselect("Filter by Internet Service", df_raw['InternetService'].unique().tolist(), default=df_raw['InternetService'].unique().tolist())

        # Apply filters
        mask = df_raw['Churn'].isin(filter_churn) & df_raw['Contract'].isin(filter_contract) & df_raw['InternetService'].isin(filter_internet)
        df_filtered = df_raw[mask].copy()
        st.info(f"📊 Showing {len(df_filtered):,} of {len(df):,} customers ({len(df_filtered)/len(df)*100:.1f}%)")
        st.dataframe(df_filtered)

        # Download button
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Data as CSV", csv, "filtered_churn_data.csv", "text/csv")

        st.markdown("---")
        st.subheader("📊 Churn Rate by Category")
        cat_cols = ['Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract', 'PaymentMethod', 'PaperlessBilling']
        selected_col = st.selectbox("Select category to analyze", cat_cols)
        if selected_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            churn_by_col = df.groupby(selected_col)['Churn'].mean().sort_values(ascending=False)
            bars = ax.bar(range(len(churn_by_col)), churn_by_col.values, color='steelblue')
            ax.set_xticks(range(len(churn_by_col)))
            ax.set_xticklabels(churn_by_col.index, rotation=45, ha='right')
            ax.set_ylabel("Churn Rate")
            ax.set_title(f"Churn Rate by {selected_col}")
            ax.set_ylim(0, 1)
            for bar, rate in zip(bars, churn_by_col.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            st.image(plot_to_img(fig))

        st.markdown("---")
        st.subheader("📈 Numerical Feature Analysis")
        analysis_type = st.radio("Select plot type", ["Box Plot", "Distribution"], horizontal=True)
        if analysis_type == "Box Plot":
            fig, ax = plt.subplots(figsize=(10, 6))
            # Use df (preprocessed) for numerical columns, but map Churn back to Yes/No for display
            df_box = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
            df_box['Churn'] = df['Churn'].map({1: 'Churned', 0: 'Not Churned'})
            df_box_melted = df_box.melt(id_vars='Churn', value_vars=['tenure', 'MonthlyCharges', 'TotalCharges'])
            # Convert value to numeric, coercing errors
            df_box_melted['value'] = pd.to_numeric(df_box_melted['value'], errors='coerce')
            df_box_melted = df_box_melted.dropna()
            sns.boxplot(data=df_box_melted, x='variable', y='value', hue='Churn', ax=ax)
            ax.set_title('Feature Distribution by Churn Status')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Value')
            plt.tight_layout()
            st.image(plot_to_img(fig))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax = axes[0]
            churned = pd.to_numeric(df_raw[df_raw['Churn'] == 'Yes']['MonthlyCharges'], errors='coerce').dropna()
            not_churned = pd.to_numeric(df_raw[df_raw['Churn'] == 'No']['MonthlyCharges'], errors='coerce').dropna()
            ax.hist(not_churned, bins=30, alpha=0.6, color='#2ecc71', label='Not Churned', edgecolor='black')
            ax.hist(churned, bins=30, alpha=0.6, color='#e74c3c', label='Churned', edgecolor='black')
            ax.set_xlabel('Monthly Charges ($)')
            ax.set_ylabel('Count')
            ax.set_title('Monthly Charges Distribution by Churn')
            ax.legend()
            ax = axes[1]
            churned = df_raw[df_raw['Churn'] == 'Yes']['tenure'].dropna()
            not_churned = df_raw[df_raw['Churn'] == 'No']['tenure'].dropna()
            ax.hist(not_churned, bins=30, alpha=0.6, color='#2ecc71', label='Not Churned', edgecolor='black')
            ax.hist(churned, bins=30, alpha=0.6, color='#e74c3c', label='Churned', edgecolor='black')
            ax.set_xlabel('Tenure (months)')
            ax.set_ylabel('Count')
            ax.set_title('Tenure Distribution by Churn')
            ax.legend()
            plt.tight_layout()
            st.image(plot_to_img(fig))

        st.markdown("---")
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.image(plot_to_img(fig))

    # ========== TAB 3: MODEL TRAINING ==========
    with tabs[2]:
        st.header("🤖 Model Training")
        st.markdown("Train multiple machine learning models and compare performance")

        # Model selection
        st.subheader("Training Options")
        use_smote = st.checkbox("Apply SMOTE for class imbalance", value=True, help="SMOTE creates synthetic samples to balance classes")

        # Hyperparameter tuning option
        col1, col2 = st.columns(2)
        with col1:
            tune_models = st.checkbox("🔧 Enable Hyperparameter Tuning", value=False,
                                      help="Uses GridSearchCV to find optimal parameters (takes longer)")
        with col2:
            if tune_models:
                tune_model_select = st.multiselect(
                    "Select models to tune",
                    ['Random Forest', 'Gradient Boosting', 'XGBoost'],
                    default=['XGBoost']
                )

        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a minute."):
                X = df.drop('Churn', axis=1)
                y = df['Churn']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                # Apply SMOTE if enabled
                if use_smote:
                    try:
                        smote = SMOTE(random_state=42)
                        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                        st.success("✅ SMOTE applied!")
                    except Exception as e:
                        st.warning(f"⚠️ SMOTE failed, training without it: {e}")
                        X_train_res, y_train_res = X_train, y_train
                else:
                    X_train_res, y_train_res = X_train, y_train

                # Fill NaN
                X_train_res = X_train_res.fillna(X_train_res.median())
                X_test = X_test.fillna(X_train.median())

                results = []
                tune_results = []

                for name, model in MODELS.items():
                    try:
                        # Hyperparameter tuning for selected models
                        if tune_models and name in tune_model_select:
                            with st.status(f"🔧 Tuning {name}...", expanded=False):
                                grid_model, best_params, best_score = tune_model(X_train_res, y_train_res, name)
                                model = grid_model
                                tune_results.append({
                                    'Model': name,
                                    'Best Params': str(best_params),
                                    'Best CV Score': round(best_score, 4)
                                })
                                st.write(f"Best params for {name}: {best_params}")

                        # Train
                        model.fit(X_train_res, y_train_res)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]

                        results.append({
                            'Model': name,
                            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                            'Precision': round(precision_score(y_test, y_pred), 4),
                            'Recall': round(recall_score(y_test, y_pred), 4),
                            'F1-Score': round(f1_score(y_test, y_pred), 4),
                            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
                            'model': model,
                            'y_pred': y_pred,
                            'y_proba': y_proba
                        })
                    except Exception as e:
                        st.error(f"Error training {name}: {e}")

                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('ROC-AUC', ascending=False)
                    st.session_state['results_df'] = results_df
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['feature_names'] = X.columns.tolist()
                    best_model = results_df.iloc[0]
                    st.success(f"🏆 Best Model: **{best_model['Model']}** with ROC-AUC: {best_model['ROC-AUC']:.4f}")
                    st.dataframe(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']], use_container_width=True)

                    # Show tuning results
                    if tune_results:
                        st.subheader("🔧 Hyperparameter Tuning Results")
                        st.dataframe(pd.DataFrame(tune_results), use_container_width=True)

                    # Save all models
                    models_dict = {row['Model']: row['model'] for _, row in results_df.iterrows()}
                    st.session_state['models_dict'] = models_dict
                    st.session_state['trained'] = True

                    # Save feature importance for best model
                    best_model_obj = models_dict[best_model['Model']]
                    imp_fig = plot_feature_importance(best_model_obj, st.session_state['feature_names'])
                    if imp_fig:
                        plt.close(imp_fig)

        # Save/Load Models Section - Always visible
        st.markdown("---")
        st.subheader("💾 Save/Load Models")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Save Trained Models**")
            if st.session_state.get('models_dict'):
                # Save all models
                models_data = io.BytesIO()
                joblib.dump(st.session_state['models_dict'], models_data)
                st.download_button(
                    label="📥 Download All Models (.pkl)",
                    data=models_data.getvalue(),
                    file_name="churn_models.pkl",
                    mime="application/octet-stream"
                )
                # Save individual model
                model_to_save = st.selectbox("Select model to save individually", list(st.session_state['models_dict'].keys()))
                if model_to_save:
                    model_data = io.BytesIO()
                    joblib.dump(st.session_state['models_dict'][model_to_save], model_data)
                    st.download_button(
                        label=f"📥 Download {model_to_save} (.pkl)",
                        data=model_data.getvalue(),
                        file_name=f"{model_to_save.replace(' ', '_')}.pkl",
                        mime="application/octet-stream",
                        key=f"save_{model_to_save}"
                    )
            else:
                st.info("Train models first to enable saving")

        with col2:
            st.markdown("**Load Previously Saved Models**")
            uploaded_model = st.file_uploader("Upload .pkl model file", type=['pkl'], key='load_model')
            if uploaded_model is not None:
                try:
                    loaded_models = joblib.load(uploaded_model)
                    if isinstance(loaded_models, dict):
                        st.session_state['models_dict'] = loaded_models
                        st.success(f"✅ Loaded {len(loaded_models)} models!")
                        st.session_state['trained'] = True
                    else:
                        st.session_state['models_dict'] = {'Loaded Model': loaded_models}
                        st.success("✅ Loaded single model!")
                        st.session_state['trained'] = True
                except Exception as e:
                    st.error(f"Error loading model: {e}")

    # ========== TAB 4: MODEL COMPARISON ==========
    with tabs[3]:
        st.header("📊 Model Comparison")
        if 'results_df' not in st.session_state:
            st.info("👈 Train models first in the 'Model Training' tab")
        else:
            results_df = st.session_state['results_df']
            st.subheader("Performance Comparison")
            fig = plot_model_comparison(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']])
            st.image(plot_to_img(fig))
            st.subheader("Detailed Metrics")
            st.dataframe(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']], use_container_width=True)

    # ========== TAB 5: MODEL EVALUATION ==========
    with tabs[4]:
        st.header("📈 Model Evaluation")
        if 'results_df' not in st.session_state:
            st.info("👈 Train models first in the 'Model Training' tab")
        else:
            results_df = st.session_state['results_df']
            y_test = st.session_state['y_test']
            feature_names = st.session_state['feature_names']

            # Select model to evaluate
            model_names = results_df['Model'].tolist()
            selected_model = st.selectbox("Select model to evaluate", model_names)
            model_row = results_df[results_df['Model'] == selected_model].iloc[0]
            model = model_row['model']
            y_pred = model_row['y_pred']
            y_proba = model_row['y_proba']

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                fig = plot_confusion_matrix(y_test, y_pred)
                st.image(plot_to_img(fig))
            with col2:
                st.subheader("ROC Curve")
                fig = plot_roc_curve(y_test, y_proba, selected_model)
                st.image(plot_to_img(fig))

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Precision-Recall Curve")
                fig = plot_precision_recall(y_test, y_proba, selected_model)
                st.image(plot_to_img(fig))
            with col2:
                st.subheader("Feature Importance")
                fig = plot_feature_importance(model, feature_names)
                if fig:
                    st.image(plot_to_img(fig))

            st.markdown("---")
            st.subheader("Classification Report")

            # Build a clean DataFrame from sklearn report
            from sklearn.metrics import precision_recall_fscore_support
            labels = ['Not Churn', 'Churn']
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

            report_data = []
            for i, label in enumerate(labels):
                report_data.append({
                    'Class': label,
                    'Precision': f'{precision[i]:.3f}',
                    'Recall': f'{recall[i]:.3f}',
                    'F1-Score': f'{f1[i]:.3f}',
                    'Support': f'{int(support[i])}'
                })

            # Overall accuracy
            acc = accuracy_score(y_test, y_pred)
            report_data.append({
                'Class': '**Accuracy**',
                'Precision': f'{acc:.3f}',
                'Recall': f'{acc:.3f}',
                'F1-Score': f'{acc:.3f}',
                'Support': f'{len(y_test)}'
            })

            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🖼️ Export Charts")
            with st.expander("Click to export charts as PNG"):
                st.info("Right-click any chart above and select 'Save image as...' or use the button below:")
                # Generate a combined export
                fig_cm = plot_confusion_matrix(y_test, y_pred)
                fig_roc = plot_roc_curve(y_test, y_proba, selected_model)
                fig_pr = plot_precision_recall(y_test, y_proba, selected_model)

                col1, col2, col3 = st.columns(3)
                with col1:
                    buf_cm = plot_to_img(fig_cm)
                    st.download_button("📥 Confusion Matrix", buf_cm, "confusion_matrix.png", "image/png", key="export_cm")
                with col2:
                    buf_roc = plot_to_img(fig_roc)
                    st.download_button("📥 ROC Curve", buf_roc, "roc_curve.png", "image/png", key="export_roc")
                with col3:
                    buf_pr = plot_to_img(fig_pr)
                    st.download_button("📥 PR Curve", buf_pr, "pr_curve.png", "image/png", key="export_pr")

    # ========== TAB 6: PREDICT ==========
    with tabs[5]:
        st.header("🔮 Customer Churn Prediction")
        st.markdown("Enter customer details to predict churn probability")

        # Check if we have models available (either trained or loaded)
        has_models = st.session_state.get('trained', False) or 'models_dict' in st.session_state

        if not has_models:
            st.warning("⚠️ Train a model first in the 'Model Training' tab OR load a saved model")
        else:
            # Get best model from either trained or loaded models
            models_dict = st.session_state.get('models_dict', {})
            if models_dict:
                # Use the first model if no results_df
                if 'results_df' in st.session_state:
                    best_model_name = st.session_state['results_df'].iloc[0]['Model']
                    best_model = models_dict.get(best_model_name, list(models_dict.values())[0])
                else:
                    best_model_name = st.selectbox("Select model to use", list(models_dict.keys()))
                    best_model = models_dict[best_model_name]
                feature_names = list(best_model.feature_names_in_) if hasattr(best_model, 'feature_names_in_') else None
                if feature_names is None:
                    feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                                    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

            st.subheader("Customer Details")
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            with col2:
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)

            if st.button("🔮 Predict Churn", type="primary"):
                input_data = pd.DataFrame([{
                    'gender': 1 if gender == "Male" else 0,
                    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                    'Partner': 1 if partner == "Yes" else 0,
                    'Dependents': 1 if dependents == "Yes" else 0,
                    'tenure': tenure,
                    'PhoneService': 1 if phone_service == "Yes" else 0,
                    'MultipleLines': 1 if multiple_lines == "Yes" else 0,
                    'InternetService': 2 if internet_service == "Fiber optic" else (1 if internet_service == "DSL" else 0),
                    'OnlineSecurity': 1 if online_security == "Yes" else 0,
                    'OnlineBackup': 1 if online_backup == "Yes" else 0,
                    'DeviceProtection': 1 if device_protection == "Yes" else 0,
                    'TechSupport': 1 if tech_support == "Yes" else 0,
                    'StreamingTV': 1 if streaming_tv == "Yes" else 0,
                    'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
                    'Contract': 2 if contract == "Two year" else (1 if contract == "One year" else 0),
                    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                    'PaymentMethod': ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(payment_method),
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges
                }])
                # Ensure all feature columns exist
                for col in feature_names:
                    if col not in input_data.columns:
                        input_data[col] = 0
                # Reorder columns to match training data
                input_data = input_data[feature_names]
                proba = best_model.predict_proba(input_data)[0][1]
                pred = best_model.predict(input_data)[0]

                # Confidence calculation
                if proba > 0.8 or proba < 0.2:
                    confidence = "🔵 High Confidence"
                    conf_color = "blue"
                elif proba > 0.6 or proba < 0.4:
                    confidence = "🟡 Medium Confidence"
                    conf_color = "orange"
                else:
                    confidence = "🔴 Low Confidence (borderline)"
                    conf_color = "red"

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if pred == 1:
                        st.error(f"⚠️ **HIGH CHURN RISK: {proba*100:.1f}%**")
                        st.info("💡 Recommended action: Target this customer with a retention campaign")
                    else:
                        st.success(f"✅ **LOW CHURN RISK: {proba*100:.1f}%** - Likely to stay")
                with col2:
                    st.metric("Prediction Reliability", confidence)
                    st.progress(proba if pred == 1 else 1 - proba)
                    st.caption(f"Model confidence score: {(proba*100 if pred == 1 else (1-proba)*100):.0f}%")

            # ========== BATCH PREDICTIONS SECTION ==========
            st.markdown("---")
            st.subheader("📊 Batch Predictions")
            st.markdown("Upload a CSV file with customer data to get predictions for multiple customers at once.")

            # Sample template download
            st.markdown("**Step 1: Download template (optional)**")
            template_df = pd.DataFrame([{
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 12,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': 'Yes',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 70.0,
                'TotalCharges': 1000.0
            }])
            csv_template = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Template",
                data=csv_template,
                file_name="customer_template.csv",
                mime="text/csv",
                key="download_template"
            )

            # File upload
            st.markdown("**Step 2: Upload your CSV file**")
            uploaded_csv = st.file_uploader("Upload CSV with customer data", type=['csv'], key='batch_upload')

            if uploaded_csv is not None:
                try:
                    # Read uploaded file
                    batch_df = pd.read_csv(uploaded_csv)
                    st.info(f"📊 Loaded {len(batch_df)} customers")

                    # Show preview
                    st.markdown("**Data Preview:**")
                    st.dataframe(batch_df.head())

                    # Preprocess the batch data
                    st.markdown("**Step 3: Run predictions**")
                    if st.button("🚀 Run Batch Predictions", type="primary"):
                        with st.spinner("Processing..."):
                            # Prepare batch data
                            batch_processed = batch_df.copy()

                            # Encode columns (same as main preprocessing)
                            batch_processed['gender'] = batch_processed['gender'].map({'Male': 1, 'Female': 0})
                            batch_processed['Partner'] = batch_processed['Partner'].map({'Yes': 1, 'No': 0})
                            batch_processed['Dependents'] = batch_processed['Dependents'].map({'Yes': 1, 'No': 0})
                            batch_processed['PhoneService'] = batch_processed['PhoneService'].map({'Yes': 1, 'No': 0})
                            batch_processed['PaperlessBilling'] = batch_processed['PaperlessBilling'].map({'Yes': 1, 'No': 0})
                            batch_processed['MultipleLines'] = batch_processed['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
                            batch_processed['OnlineSecurity'] = batch_processed['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
                            batch_processed['OnlineBackup'] = batch_processed['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
                            batch_processed['DeviceProtection'] = batch_processed['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
                            batch_processed['TechSupport'] = batch_processed['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
                            batch_processed['StreamingTV'] = batch_processed['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
                            batch_processed['StreamingMovies'] = batch_processed['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
                            batch_processed['InternetService'] = batch_processed['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
                            batch_processed['Contract'] = batch_processed['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
                            batch_processed['PaymentMethod'] = batch_processed['PaymentMethod'].map({
                                'Electronic check': 0, 'Mailed check': 1,
                                'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
                            })
                            batch_processed['TotalCharges'] = pd.to_numeric(batch_processed['TotalCharges'], errors='coerce').fillna(0)
                            batch_processed['SeniorCitizen'] = batch_processed['SeniorCitizen'].fillna(0)

                            # Select features
                            batch_input = batch_processed[feature_names].fillna(0)

                            # Get predictions
                            predictions = best_model.predict(batch_input)
                            probabilities = best_model.predict_proba(batch_input)[:, 1]

                            # Create results dataframe
                            results_df = batch_df.copy()
                            results_df['Predicted_Churn'] = predictions
                            results_df['Churn_Probability'] = (probabilities * 100).round(2)
                            results_df['Risk_Level'] = results_df['Churn_Probability'].apply(
                                lambda x: 'High' if x > 60 else ('Medium' if x > 30 else 'Low')
                            )

                            # Show summary
                            st.success(f"✅ Predictions complete for {len(results_df)} customers!")

                            col1, col2, col3 = st.columns(3)
                            high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                            medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium'])
                            low_risk = len(results_df[results_df['Risk_Level'] == 'Low'])

                            col1.metric("🔴 High Risk", f"{high_risk} ({high_risk/len(results_df)*100:.1f}%)")
                            col2.metric("🟡 Medium Risk", f"{medium_risk} ({medium_risk/len(results_df)*100:.1f}%)")
                            col3.metric("🟢 Low Risk", f"{low_risk} ({low_risk/len(results_df)*100:.1f}%)")

                            # Show results
                            st.markdown("**Prediction Results:**")
                            st.dataframe(results_df[['CustomerID' if 'CustomerID' in results_df.columns else results_df.columns[0],
                                                   'Predicted_Churn', 'Churn_Probability', 'Risk_Level']].head(10))

                            # Download results
                            results_csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=results_csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                                key="download_results"
                            )

                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    st.info("Make sure your CSV has the required columns: gender, SeniorCitizen, Partner, Dependents, tenure, etc.")

    # ========== TAB 7: BUSINESS ROI ==========
    with tabs[6]:
        st.header("💼 Business Impact & ROI Analysis")
        st.markdown("Calculate the financial impact of churn prediction and retention strategies")

        # Check if we have models (trained or loaded)
        has_results = 'results_df' in st.session_state
        has_models = st.session_state.get('trained', False) or 'models_dict' in st.session_state

        if not has_results and not has_models:
            st.warning("⚠️ Train a model first OR load a saved model")
        elif has_results:
            # Full ROI analysis with trained models
            model_row = st.session_state['results_df'].iloc[0]
            y_test = st.session_state['y_test']
            y_pred = model_row['y_pred']
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            st.subheader("💰 Cost Parameters")
            col1, col2 = st.columns(2)
            with col1:
                retention_cost = st.number_input("Cost per Retention Campaign ($)", min_value=0, max_value=500, value=50, help="Average cost to run a retention campaign per customer")
            with col2:
                clv = st.number_input("Customer Lifetime Value ($)", min_value=0, max_value=5000, value=500, help="Average revenue from a retained customer")

            # Calculate metrics
            total_cost = (tp + fp) * retention_cost
            saved_revenue = tp * clv
            lost_revenue = fn * clv
            net_benefit = saved_revenue - total_cost
            roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0

            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retention Cost", f"${total_cost:,.0f}")
            col2.metric("Revenue Saved", f"${saved_revenue:,.0f}")
            col3.metric("Revenue Lost", f"${lost_revenue:,.0f}")
            col4.metric("Net Benefit", f"${net_benefit:,.0f}")
            st.session_state['roi_value'] = roi

            st.markdown("---")
            st.subheader("📈 ROI Analysis")
            roi_color = "green" if roi > 0 else "red"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {roi_color}, #1e3c72); padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                <h2 style='margin: 0; font-size: 3em;'>ROI: {roi:.1f}%</h2>
                <p style='margin: 10px 0 0 0; font-size: 1.2em;'>For every $1 spent on retention, you get ${roi/100 + 1:.2f} back</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📊 Key Insights")
            st.write(f"- ✅ **True Positives:** {tp} churners correctly identified (can be targeted for retention)")
            st.write(f"- ⚠️ **False Positives:** {fp} customers falsely flagged (unnecessary retention cost)")
            st.write(f"- ❌ **False Negatives:** {fn} churners missed (lost revenue opportunity)")
            st.write(f"- ✅ **True Negatives:** {tn} non-churners correctly identified (no action needed)")

            st.markdown("---")
            st.subheader("💡 Business Recommendations")
            if roi > 100:
                st.success("🎉 Excellent ROI! The model is highly profitable. Consider scaling the retention program.")
            elif roi > 0:
                st.info("✅ Positive ROI. The model is profitable. Consider tuning to reduce false positives.")
            else:
                st.error("⚠️ Negative ROI. Consider adjusting the model threshold or reducing retention costs.")
        else:
            # Loaded model - show simplified ROI estimator
            st.info("📌 Loaded model detected. Full ROI analysis requires test set from training. Showing estimator based on model accuracy.")

            models_dict = st.session_state.get('models_dict', {})
            model_name = st.selectbox("Select model for estimation", list(models_dict.keys()))
            model = models_dict[model_name]

            # Estimate based on typical performance
            estimated_accuracy = st.slider("Estimated Model Accuracy (%)", 60, 95, 80)
            estimated_recall = st.slider("Estimated Recall (%)", 50, 90, 70)

            st.markdown("---")
            st.subheader("💰 Cost Parameters")
            col1, col2 = st.columns(2)
            with col1:
                retention_cost = st.number_input("Cost per Retention Campaign ($)", min_value=0, max_value=500, value=50)
            with col2:
                clv = st.number_input("Customer Lifetime Value ($)", min_value=0, max_value=5000, value=500)

            # Simulate with 1000 customers and 26.5% churn rate
            total_customers = 1000
            churned = int(total_customers * 0.265)
            not_churned = total_customers - churned

            # Based on estimated recall
            tp = int(churned * estimated_recall / 100)
            fn = churned - tp
            fp = int(not_churned * (1 - estimated_accuracy / 100))
            tn = not_churned - fp

            total_cost = (tp + fp) * retention_cost
            saved_revenue = tp * clv
            lost_revenue = fn * clv
            net_benefit = saved_revenue - total_cost
            roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0

            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retention Cost", f"${total_cost:,.0f}")
            col2.metric("Revenue Saved", f"${saved_revenue:,.0f}")
            col3.metric("Revenue Lost", f"${lost_revenue:,.0f}")
            col4.metric("Net Benefit", f"${net_benefit:,.0f}")

            st.markdown("---")
            roi_color = "green" if roi > 0 else "red"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {roi_color}, #1e3c72); padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                <h2 style='margin: 0; font-size: 3em;'>Estimated ROI: {roi:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

    # ========== TAB 8: CUSTOM DATASET ==========
    with tabs[7]:
        st.header("📂 Custom Dataset Analysis")
        st.markdown("Upload any CSV dataset with a binary target column to train models and get predictions.")

        uploaded_custom = st.file_uploader("Upload your CSV dataset", type=['csv'], key='custom_data')

        if uploaded_custom is not None:
            try:
                custom_df = pd.read_csv(uploaded_custom)
                st.success(f"✅ Loaded dataset: {len(custom_df)} rows, {len(custom_df.columns)} columns")
                st.dataframe(custom_df.head())

                # Select target column
                st.markdown("---")
                st.subheader("Configure Target Variable")

                col1, col2 = st.columns(2)
                with col1:
                    target_col = st.selectbox("Select the target column (what to predict)", custom_df.columns.tolist())
                with col2:
                    if target_col:
                        target_unique = custom_df[target_col].unique()
                        st.write(f"**Unique values:** {len(target_unique)}")
                        st.write(f"**Values:** {target_unique[:10]}")

                if target_col:
                    # Check if binary
                    n_unique = custom_df[target_col].nunique()

                    if n_unique == 1:
                        st.error("Target column must have at least 2 unique values")
                    elif n_unique > 10:
                        st.warning("Target has many unique values (>10). Binary classification may not be suitable.")
                        proceed = st.checkbox("I understand, proceed anyway")
                        if not proceed:
                            proceed = False
                    else:
                        proceed = True

                    if proceed:
                        st.markdown("---")
                        st.subheader("⚙️ Preprocessing Options")

                        drop_cols = st.multiselect("Drop unnecessary columns", [c for c in custom_df.columns if c != target_col])

                        with st.spinner("Preprocessing dataset..."):
                            # Prepare data
                            df_custom = custom_df.copy()
                            if drop_cols:
                                df_custom = df_custom.drop(columns=drop_cols)

                            # Separate target
                            y_custom = df_custom[target_col]
                            X_custom = df_custom.drop(columns=[target_col])

                            # Auto-process
                            from sklearn.preprocessing import LabelEncoder

                            # Handle missing values
                            for col in X_custom.columns:
                                if X_custom[col].dtype == 'object':
                                    X_custom[col] = X_custom[col].fillna('MISSING')
                                else:
                                    X_custom[col] = X_custom[col].fillna(X_custom[col].median())

                            # Encode categorical features
                            label_encoders = {}
                            for col in X_custom.columns:
                                if X_custom[col].dtype == 'object':
                                    le = LabelEncoder()
                                    X_custom[col] = le.fit_transform(X_custom[col].astype(str))
                                    label_encoders[col] = le

                            # Encode target if needed
                            if y_custom.dtype == 'object':
                                y_custom = LabelEncoder().fit_transform(y_custom)

                            # Ensure numeric
                            X_custom = X_custom.apply(pd.to_numeric, errors='coerce').fillna(0)

                        st.success("✅ Data ready!")
                        st.write(f"**Features:** {len(X_custom.columns)}, **Samples:** {len(X_custom)}")
                        st.write(f"**Target distribution:** {pd.Series(y_custom).value_counts().to_dict()}")

                        # Train button
                        st.markdown("---")
                        st.subheader("🤖 Train on Custom Data")

                        use_smote_custom = st.checkbox("Apply SMOTE (balance classes)", value=True, key='smote_custom')
                        tune_custom = st.checkbox("🔧 Hyperparameter Tuning (slower)", value=False, key='tune_custom')

                        if st.button("🚀 Train Models on Custom Data", type="primary"):
                            with st.spinner("Training..."):
                                X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42, stratify=y_custom)

                                # Handle NaN
                                X_train = X_train.fillna(X_train.median())
                                X_test = X_test.fillna(X_train.median())

                                # SMOTE
                                if use_smote_custom:
                                    try:
                                        smote = SMOTE(random_state=42)
                                        X_train, y_train = smote.fit_resample(X_train, y_train)
                                        st.success("✅ SMOTE applied!")
                                    except Exception as e:
                                        st.warning(f"SMOTE failed: {e}")

                                results_custom = []
                                for name, model in MODELS.items():
                                    try:
                                        if tune_custom and name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                                            tuned, params, score = tune_model(X_train, y_train, name)
                                            model = tuned
                                            st.write(f"**{name}** tuned: {params}")

                                        model.fit(X_train, y_train)
                                        y_pred = model.predict(X_test)
                                        y_proba = model.predict_proba(X_test)[:, 1]

                                        results_custom.append({
                                            'Model': name,
                                            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                                            'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
                                            'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
                                            'F1-Score': round(f1_score(y_test, y_pred, average='weighted'), 4),
                                            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4)
                                        })
                                    except Exception as e:
                                        st.error(f"{name} error: {e}")

                                if results_custom:
                                    rc_df = pd.DataFrame(results_custom).sort_values('ROC-AUC', ascending=False)
                                    st.subheader("📊 Results")
                                    st.dataframe(rc_df, use_container_width=True)

                                    # Feature importance for best model
                                    st.subheader("🔍 Feature Importance")
                                    best_model_name = rc_df.iloc[0]['Model']
                                    best_model = [v for k, v in MODELS.items() if k == best_model_name][0]
                                    best_model.fit(X_train, y_train)
                                    fig = plot_feature_importance(best_model, X_custom.columns.tolist())
                                    if fig:
                                        st.image(plot_to_img(fig))

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Make sure your CSV file is valid and has readable data.")

    # Footer
    st.markdown("""
    <div class="app-footer">
        <p style="margin: 0;">Built with ❤️ using Python, Streamlit, and Machine Learning</p>
        <p style="margin: 0.3rem 0 0 0;">Made by Vedant Karmankar &nbsp;·&nbsp; Customer Churn Prediction Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
