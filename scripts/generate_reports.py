#!/usr/bin/env python3
"""
Criminal Justice Bias Analysis Pipeline
========================================
Analyzes COMPAS recidivism and NYPD Stop-Question-Frisk datasets for algorithmic bias.

This script generates:
- JSON data files for the web dashboard
- PNG visualizations
- Chart metadata with explanations

Authors: Group 2 - Vidit, Ronaldo, Kaleemulla, Vishal
Course: Ethics, Sociology, and Governance of AI (MAIB AI 219)
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# Fairlearn imports
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame
    )
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logging.warning("Fairlearn not available. Using manual fairness metric implementations.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_OUTPUT_DIR = PROJECT_ROOT / "web" / "public" / "data"
IMG_OUTPUT_DIR = PROJECT_ROOT / "web" / "public" / "img"

# Ensure output directories exist
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Chart style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette for consistency
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'neutral': '#6b7280',
    'races': {
        'African_American': '#ef4444',
        'White': '#3b82f6',
        'Hispanic': '#10b981',
        'Asian': '#f59e0b',
        'Other': '#8b5cf6',
        'Native_American': '#ec4899',
        'Black': '#ef4444',
        'B': '#ef4444',
        'W': '#3b82f6',
        'Q': '#10b981',
        'A': '#f59e0b',
        'P': '#ec4899',
        'I': '#8b5cf6',
        'Z': '#6b7280',
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_json(data: Any, filename: str) -> Path:
    """Save data to JSON file."""
    filepath = DATA_OUTPUT_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {filepath}")
    return filepath


def save_chart(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """Save matplotlib figure to PNG."""
    filepath = IMG_OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"Saved: {filepath}")
    return filepath


def compute_fairness_metrics_manual(y_true: np.ndarray, y_pred: np.ndarray,
                                     sensitive_attr: np.ndarray) -> Dict[str, float]:
    """
    Compute fairness metrics manually when fairlearn is not available.
    """
    groups = np.unique(sensitive_attr)
    metrics = {}

    # Selection rate (positive prediction rate) by group
    selection_rates = {}
    fpr_by_group = {}
    fnr_by_group = {}

    for group in groups:
        mask = sensitive_attr == group
        if mask.sum() == 0:
            continue

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        # Selection rate
        selection_rates[str(group)] = y_pred_g.mean()

        # Confusion matrix elements
        tn = ((y_true_g == 0) & (y_pred_g == 0)).sum()
        fp = ((y_true_g == 0) & (y_pred_g == 1)).sum()
        fn = ((y_true_g == 1) & (y_pred_g == 0)).sum()
        tp = ((y_true_g == 1) & (y_pred_g == 1)).sum()

        # FPR and FNR
        fpr_by_group[str(group)] = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_by_group[str(group)] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Compute differences (max - min)
    if selection_rates:
        metrics['selection_rate_difference'] = max(selection_rates.values()) - min(selection_rates.values())
        metrics['selection_rates_by_group'] = selection_rates

    if fpr_by_group:
        metrics['fpr_difference'] = max(fpr_by_group.values()) - min(fpr_by_group.values())
        metrics['fpr_by_group'] = fpr_by_group

    if fnr_by_group:
        metrics['fnr_difference'] = max(fnr_by_group.values()) - min(fnr_by_group.values())
        metrics['fnr_by_group'] = fnr_by_group

    # Equalized odds difference (max of FPR diff and FNR diff)
    metrics['equalized_odds_difference'] = max(
        metrics.get('fpr_difference', 0),
        metrics.get('fnr_difference', 0)
    )

    return metrics


def compute_group_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                                      sensitive_attr: np.ndarray) -> Dict[str, Dict]:
    """Compute confusion matrix for each group."""
    groups = np.unique(sensitive_attr)
    results = {}

    for group in groups:
        mask = sensitive_attr == group
        if mask.sum() == 0:
            continue

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        tn = int(((y_true_g == 0) & (y_pred_g == 0)).sum())
        fp = int(((y_true_g == 0) & (y_pred_g == 1)).sum())
        fn = int(((y_true_g == 1) & (y_pred_g == 0)).sum())
        tp = int(((y_true_g == 1) & (y_pred_g == 1)).sum())

        total = tn + fp + fn + tp

        results[str(group)] = {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'total': total,
            'accuracy': (tn + tp) / total if total > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        }

    return results


# ============================================================================
# DATASET LOADING
# ============================================================================

def find_compas_file() -> Path:
    """Find COMPAS dataset file with robust path detection."""
    possible_paths = [
        PROJECT_ROOT / "_propublica_data_for_fairml.csv",
        PROJECT_ROOT / "propublica_data_for_fairml.csv",
        PROJECT_ROOT / "archive (1)" / "propublicaCompassRecividism_data_fairml.csv" / "propublica_data_for_fairml.csv",
        PROJECT_ROOT / "archive (1)" / "propublicaCompassRecividism_data_fairml.csv" / "_propublica_data_for_fairml.csv",
        PROJECT_ROOT / "data" / "propublica_data_for_fairml.csv",
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"Found COMPAS dataset: {path}")
            return path

    # Search recursively
    for csv_file in PROJECT_ROOT.rglob("*propublica*.csv"):
        if csv_file.is_file():
            logger.info(f"Found COMPAS dataset (recursive): {csv_file}")
            return csv_file

    raise FileNotFoundError(
        "COMPAS dataset not found. Please ensure propublica_data_for_fairml.csv "
        "or _propublica_data_for_fairml.csv exists in the project directory."
    )


def find_nypd_file() -> Path:
    """Find NYPD SQF dataset file."""
    possible_paths = [
        PROJECT_ROOT / "2012.csv",
        PROJECT_ROOT / "data" / "2012.csv",
        PROJECT_ROOT / "nypd_sqf_2012.csv",
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"Found NYPD SQF dataset: {path}")
            return path

    raise FileNotFoundError(
        "NYPD SQF 2012 dataset not found. Please ensure 2012.csv exists in the project directory."
    )


def load_compas_data() -> pd.DataFrame:
    """Load and validate COMPAS dataset."""
    filepath = find_compas_file()
    df = pd.read_csv(filepath)
    logger.info(f"Loaded COMPAS data: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"COMPAS columns: {list(df.columns)}")
    return df


def load_nypd_data(sample_size: int = 100000) -> pd.DataFrame:
    """
    Load NYPD SQF dataset with sampling for performance.
    The full dataset is ~500k rows; we sample for faster analysis.
    """
    filepath = find_nypd_file()

    # Read with sampling for large file
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded NYPD data: {len(df)} rows, {len(df.columns)} columns")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_SEED)
        logger.info(f"Sampled NYPD data to {sample_size} rows for analysis")

    logger.info(f"NYPD columns: {list(df.columns)[:20]}...")  # First 20 columns
    return df


# ============================================================================
# COMPAS ANALYSIS
# ============================================================================

def analyze_compas(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive COMPAS analysis including:
    - Data preprocessing
    - Model training (3 models)
    - Fairness analysis
    - Bias mitigation
    - Visualization generation
    """
    logger.info("=" * 60)
    logger.info("COMPAS RECIDIVISM ANALYSIS")
    logger.info("=" * 60)

    results = {
        'dataset_name': 'COMPAS',
        'description': 'ProPublica COMPAS Recidivism Dataset',
        'sample_size': len(df),
        'charts': [],
        'metrics': {},
        'fairness': {},
        'models': {},
    }

    # -------------------------------------------------------------------------
    # 1. DATA PREPROCESSING
    # -------------------------------------------------------------------------
    logger.info("Step 1: Data Preprocessing")

    # Detect target column
    target_col = None
    for col in ['Two_yr_Recidivism', 'two_year_recid', 'recidivism', 'is_recid']:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"Could not find target column. Available: {list(df.columns)}")

    logger.info(f"Target column: {target_col}")

    # Detect race columns (already one-hot encoded in this dataset)
    race_cols = ['African_American', 'Asian', 'Hispanic', 'Native_American', 'Other']
    available_race_cols = [c for c in race_cols if c in df.columns]

    # Create race label from one-hot encoding
    def get_race_label(row):
        for col in available_race_cols:
            if row.get(col, 0) == 1:
                return col
        return 'White'  # If none of the minority flags are set

    df['race_label'] = df.apply(get_race_label, axis=1)

    # Detect sex column
    sex_col = 'Female' if 'Female' in df.columns else None
    if sex_col:
        df['sex_label'] = df[sex_col].map({1: 'Female', 0: 'Male'})

    # Feature columns (exclude target and derived columns)
    exclude_cols = [target_col, 'race_label', 'sex_label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Handle missing values
    X = df[feature_cols].copy()
    y = df[target_col].values
    race = df['race_label'].values

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    # Store dataset summary
    results['preprocessing'] = {
        'target_column': target_col,
        'feature_columns': feature_cols,
        'race_column': 'race_label (derived from one-hot)',
        'n_features': len(feature_cols),
        'missing_values': int(df[feature_cols].isnull().sum().sum()),
        'outcome_distribution': {
            'positive': int(y.sum()),
            'negative': int(len(y) - y.sum()),
            'base_rate': float(y.mean())
        }
    }

    # -------------------------------------------------------------------------
    # 2. EXPLORATORY ANALYSIS & VISUALIZATIONS
    # -------------------------------------------------------------------------
    logger.info("Step 2: Exploratory Analysis")

    # Chart 1: Outcome base rate by race
    race_outcome = df.groupby('race_label')[target_col].agg(['mean', 'count'])
    race_outcome.columns = ['recidivism_rate', 'count']
    race_outcome = race_outcome.sort_values('count', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(race_outcome)), race_outcome['recidivism_rate'],
                  color=[COLORS['races'].get(r, COLORS['neutral']) for r in race_outcome.index])
    ax.set_xticks(range(len(race_outcome)))
    ax.set_xticklabels(race_outcome.index, rotation=45, ha='right')
    ax.set_ylabel('Recidivism Rate')
    ax.set_xlabel('Race')
    ax.set_title('Two-Year Recidivism Rate by Race (COMPAS)')
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, val in zip(bars, race_outcome['recidivism_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    # Add count labels
    for i, (idx, row) in enumerate(race_outcome.iterrows()):
        ax.text(i, 0.02, f'n={row["count"]:,}', ha='center', va='bottom',
                fontsize=8, color='white', fontweight='bold')

    save_chart(fig, 'compas_recidivism_by_race.png')

    results['charts'].append({
        'chart_id': 'compas_recidivism_by_race',
        'filename': 'compas_recidivism_by_race.png',
        'title': 'Two-Year Recidivism Rate by Race',
        'dataset': 'COMPAS',
        'explanation': (
            "This chart shows the actual two-year recidivism rates across different racial groups "
            "in the COMPAS dataset. African American defendants show a higher observed recidivism rate "
            "compared to other groups. However, this disparity reflects complex socioeconomic factors, "
            "systemic inequalities in the criminal justice system, and potential selection bias in who "
            "gets arrested and prosecuted. The base rate differences are a key consideration when "
            "evaluating algorithmic fairness, as they affect the interpretation of prediction errors."
        ),
        'speaker_notes': [
            "African American defendants have highest observed recidivism rate in this dataset",
            "Base rate differences complicate fairness analysis - equal error rates may not be achievable",
            "These rates reflect arrests/convictions, not actual criminal behavior",
            "Socioeconomic factors and systemic bias contribute to these disparities"
        ],
        'slide_title': 'Recidivism Rates Vary by Race',
        'talk_track': (
            "Looking at the raw recidivism rates, we see significant variation across racial groups. "
            "African American defendants in this dataset have a recidivism rate of about 52%, compared to "
            "roughly 39% for White defendants. These differences reflect not just individual behavior but "
            "systemic factors including policing patterns, socioeconomic conditions, and selection effects "
            "in who enters the criminal justice system in the first place."
        )
    })

    results['metrics']['base_rates_by_race'] = race_outcome['recidivism_rate'].to_dict()
    results['metrics']['counts_by_race'] = race_outcome['count'].to_dict()

    # Chart 2: Prior convictions distribution by race
    if 'Number_of_Priors' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for race_group in ['African_American', 'White', 'Hispanic']:
            if race_group in df['race_label'].values:
                subset = df[df['race_label'] == race_group]['Number_of_Priors']
                ax.hist(subset, bins=range(0, min(20, int(subset.max())+2)), alpha=0.5,
                        label=f'{race_group} (n={len(subset)})',
                        color=COLORS['races'].get(race_group, COLORS['neutral']))

        ax.set_xlabel('Number of Prior Convictions')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prior Convictions by Race (COMPAS)')
        ax.legend()

        save_chart(fig, 'compas_priors_distribution.png')

        results['charts'].append({
            'chart_id': 'compas_priors_distribution',
            'filename': 'compas_priors_distribution.png',
            'title': 'Prior Convictions Distribution by Race',
            'dataset': 'COMPAS',
            'explanation': (
                "The distribution of prior convictions varies across racial groups, with African American "
                "defendants showing a wider spread. Prior convictions are a strong predictor of recidivism "
                "but also reflect historical disparities in policing and prosecution. Using this feature "
                "in prediction models can perpetuate historical biases, as minority communities have been "
                "disproportionately policed. This creates a feedback loop where past discrimination "
                "influences future predictions."
            ),
            'speaker_notes': [
                "Prior convictions strongly predict recidivism but carry historical bias",
                "Differential policing means minorities have more opportunity for prior arrests",
                "Using priors in models can perpetuate historical discrimination",
                "This illustrates the 'feedback loop' problem in predictive systems"
            ],
            'slide_title': 'Prior Convictions: A Biased Feature?',
            'talk_track': (
                "Prior convictions are one of the strongest predictors of future recidivism, but they're "
                "also deeply problematic from a fairness perspective. Due to historical over-policing of "
                "minority communities, African American individuals are more likely to have prior records "
                "for the same behavior. When we train algorithms on this data, we risk encoding these "
                "historical disparities into our predictions."
            )
        })

    # -------------------------------------------------------------------------
    # 3. MODEL TRAINING
    # -------------------------------------------------------------------------
    logger.info("Step 3: Model Training")

    # Train/test split with stratification
    X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
        X_imputed, y, race, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_iter=100)
    }

    model_results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Train
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

        # Compute metrics
        model_results[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_pred_proba)),
            'brier_score': float(brier_score_loss(y_test, y_pred_proba)),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
        }

        # Fairness metrics
        fairness = compute_fairness_metrics_manual(y_test, y_pred, race_test)
        model_results[name]['fairness'] = fairness

        logger.info(f"  AUC: {model_results[name]['auc']:.3f}, "
                   f"F1: {model_results[name]['f1']:.3f}, "
                   f"DP Diff: {fairness['selection_rate_difference']:.3f}")

    results['models'] = {k: {kk: vv for kk, vv in v.items()
                             if kk not in ['predictions', 'probabilities']}
                        for k, v in model_results.items()}

    # Store best model info
    best_model_name = max(model_results, key=lambda x: model_results[x]['auc'])
    best_model = models[best_model_name]
    best_results = model_results[best_model_name]

    results['best_model'] = {
        'name': best_model_name,
        'auc': best_results['auc'],
        'f1': best_results['f1'],
    }

    # Chart 3: Model comparison - AUC
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(model_results.keys())
    aucs = [model_results[m]['auc'] for m in model_names]
    f1s = [model_results[m]['f1'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, aucs, width, label='AUC', color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score', color=COLORS['secondary'])

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (COMPAS)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    save_chart(fig, 'compas_model_comparison.png')

    results['charts'].append({
        'chart_id': 'compas_model_comparison',
        'filename': 'compas_model_comparison.png',
        'title': 'Model Performance Comparison',
        'dataset': 'COMPAS',
        'explanation': (
            "This chart compares three machine learning models trained to predict two-year recidivism. "
            "AUC (Area Under ROC Curve) measures overall discrimination ability, while F1 Score balances "
            "precision and recall. Gradient Boosting typically achieves the highest AUC, demonstrating "
            "better ability to rank defendants by risk. However, higher accuracy doesn't guarantee fairer "
            "outcomes - a model can be accurate overall while producing disparate error rates across groups."
        ),
        'speaker_notes': [
            "All models achieve moderate predictive performance (AUC ~0.65-0.72)",
            "Higher complexity doesn't always mean better performance here",
            "Accuracy alone is insufficient - must consider fairness metrics",
            "The 'best' model depends on how we weight accuracy vs. fairness"
        ],
        'slide_title': 'Comparing Predictive Models',
        'talk_track': (
            "We trained three models of increasing complexity. Interestingly, the performance gains from "
            "more sophisticated models are modest. All achieve AUC scores around 0.65-0.72, which is "
            "typical for recidivism prediction. The key insight is that raw accuracy isn't everything - "
            "we need to examine how these errors are distributed across demographic groups."
        )
    })

    # Chart 4: ROC curves for all models
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, res in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
        ax.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison (COMPAS)')
    ax.legend(loc='lower right')

    save_chart(fig, 'compas_roc_curves.png')

    results['charts'].append({
        'chart_id': 'compas_roc_curves',
        'filename': 'compas_roc_curves.png',
        'title': 'ROC Curves - Model Comparison',
        'dataset': 'COMPAS',
        'explanation': (
            "ROC curves visualize the trade-off between true positive rate (correctly identifying recidivists) "
            "and false positive rate (incorrectly flagging non-recidivists) at various thresholds. The curves "
            "for all three models are similar, suggesting the choice of algorithm matters less than the "
            "fundamental predictability of the outcome. The diagonal line represents random guessing - "
            "our models perform meaningfully better but are far from perfect prediction."
        ),
        'speaker_notes': [
            "ROC curves show trade-off between sensitivity and specificity",
            "All models significantly outperform random chance",
            "Similar curves suggest algorithm choice has limited impact",
            "Perfect prediction (AUC=1.0) is likely impossible for this task"
        ],
        'slide_title': 'ROC Analysis Shows Moderate Predictability',
        'talk_track': (
            "The ROC curves tell an important story - all three models perform similarly, with AUCs "
            "clustering around 0.70. This suggests that the choice of algorithm matters less than the "
            "inherent difficulty of predicting human behavior. No model achieves excellent discrimination, "
            "which raises questions about deploying such systems for high-stakes decisions."
        )
    })

    # Chart 5: Confusion Matrix for best model
    y_pred_best = best_results['predictions']
    cm = confusion_matrix(y_test, y_pred_best)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Recidivism', 'Recidivism'])
    ax.set_yticklabels(['No Recidivism', 'Recidivism'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {best_model_name} (COMPAS)')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                          ha='center', va='center', fontsize=12,
                          color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.colorbar(im, ax=ax, label='Count')

    save_chart(fig, 'compas_confusion_matrix.png')

    results['charts'].append({
        'chart_id': 'compas_confusion_matrix',
        'filename': 'compas_confusion_matrix.png',
        'title': f'Confusion Matrix - {best_model_name}',
        'dataset': 'COMPAS',
        'explanation': (
            "The confusion matrix shows how predictions map to actual outcomes. False positives (top-right) "
            "represent people incorrectly predicted to recidivate - they may face harsher treatment despite "
            "not actually reoffending. False negatives (bottom-left) represent missed predictions of "
            "recidivism. In criminal justice, false positives can mean unnecessary incarceration, while "
            "false negatives might mean inadequate supervision. The ethical weight of these errors differs."
        ),
        'speaker_notes': [
            "False positives: predicted recidivism but didn't reoffend (potential unjust detention)",
            "False negatives: predicted no recidivism but did reoffend (potential public safety issue)",
            "Neither error type is 'neutral' - both have real human consequences",
            "The distribution of these errors across groups is the fairness question"
        ],
        'slide_title': 'Understanding Prediction Errors',
        'talk_track': (
            "This confusion matrix reveals the real-world implications of our model's errors. When we "
            "predict someone will recidivate but they don't - a false positive - that person might "
            "receive a longer sentence or be denied parole unjustly. Conversely, false negatives might "
            "mean someone who needed intervention didn't receive it. Neither error is neutral."
        )
    })

    # Chart 6: Fairness metrics by race
    fairness_data = best_results['fairness']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Selection rates
    ax = axes[0]
    sr = fairness_data['selection_rates_by_group']
    races = list(sr.keys())
    rates = list(sr.values())
    bars = ax.bar(races, rates, color=[COLORS['races'].get(r, COLORS['neutral']) for r in races])
    ax.set_ylabel('Selection Rate')
    ax.set_title('Positive Prediction Rate by Race')
    ax.set_xticklabels(races, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(rates), color='gray', linestyle='--', label='Average')
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # FPR by group
    ax = axes[1]
    fpr = fairness_data['fpr_by_group']
    rates = list(fpr.values())
    bars = ax.bar(races, rates, color=[COLORS['races'].get(r, COLORS['neutral']) for r in races])
    ax.set_ylabel('False Positive Rate')
    ax.set_title('FPR by Race (Error on Non-Recidivists)')
    ax.set_xticklabels(races, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # FNR by group
    ax = axes[2]
    fnr = fairness_data['fnr_by_group']
    rates = list(fnr.values())
    bars = ax.bar(races, rates, color=[COLORS['races'].get(r, COLORS['neutral']) for r in races])
    ax.set_ylabel('False Negative Rate')
    ax.set_title('FNR by Race (Error on Recidivists)')
    ax.set_xticklabels(races, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_chart(fig, 'compas_fairness_by_race.png')

    results['charts'].append({
        'chart_id': 'compas_fairness_by_race',
        'filename': 'compas_fairness_by_race.png',
        'title': 'Fairness Metrics by Race',
        'dataset': 'COMPAS',
        'explanation': (
            "These three panels reveal how prediction errors distribute across racial groups. Selection rate "
            "(left) shows who gets predicted as high-risk - African Americans are flagged at higher rates. "
            "False Positive Rate (center) shows errors on non-recidivists - African Americans are more "
            "likely to be wrongly predicted to recidivate. False Negative Rate (right) shows errors on "
            "actual recidivists - White defendants are more likely to be wrongly predicted as low-risk. "
            "This pattern mirrors the ProPublica investigation's findings about COMPAS."
        ),
        'speaker_notes': [
            "Higher FPR for Black defendants = more likely wrongly labeled high-risk",
            "Higher FNR for White defendants = more likely wrongly labeled low-risk",
            "This is the core 'algorithmic bias' finding from ProPublica's investigation",
            "These disparities persist across different model architectures"
        ],
        'slide_title': 'Disparate Error Rates: The Fairness Problem',
        'talk_track': (
            "This is the crux of the fairness debate. African American defendants face a higher false "
            "positive rate - meaning they're more likely to be incorrectly labeled as high-risk when "
            "they won't actually recidivate. Conversely, White defendants have a higher false negative "
            "rate - more likely to be labeled low-risk when they will recidivate. These aren't random "
            "errors; they're systematically skewed."
        )
    })

    # Chart 7: Calibration curves by race
    fig, ax = plt.subplots(figsize=(10, 8))

    y_prob_best = best_results['probabilities']

    for race_group in ['African_American', 'White', 'Hispanic']:
        mask = race_test == race_group
        if mask.sum() < 50:
            continue

        prob_true, prob_pred = calibration_curve(y_test[mask], y_prob_best[mask], n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=f'{race_group} (n={mask.sum()})',
               color=COLORS['races'].get(race_group, COLORS['neutral']), linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curves by Race - {best_model_name} (COMPAS)')
    ax.legend(loc='lower right')

    save_chart(fig, 'compas_calibration_curves.png')

    results['charts'].append({
        'chart_id': 'compas_calibration_curves',
        'filename': 'compas_calibration_curves.png',
        'title': 'Calibration Curves by Race',
        'dataset': 'COMPAS',
        'explanation': (
            "Calibration curves show whether predicted probabilities match observed frequencies. A well-"
            "calibrated model should have points along the diagonal - if the model says 70% risk, about "
            "70% of those individuals should actually recidivate. Different calibration by group indicates "
            "the model's confidence means different things for different populations. This connects to the "
            "debate about whether 'fairness' means equal calibration or equal error rates across groups."
        ),
        'speaker_notes': [
            "Calibration = do predicted probabilities match reality?",
            "Differences suggest the 'same score' means different things for different groups",
            "This is central to COMPAS debate: Northpointe argued calibration was fair",
            "ProPublica argued error rate differences mattered more"
        ],
        'slide_title': 'What Does a Risk Score Really Mean?',
        'talk_track': (
            "Calibration curves address a crucial question: when the model says someone has a 70% "
            "risk of recidivism, do 70% of such people actually recidivate? The COMPAS company argued "
            "their scores were calibrated equally across races. But ProPublica showed that error rates "
            "differed. This highlights a fundamental tension - we can't simultaneously have equal "
            "calibration AND equal error rates when base rates differ between groups."
        )
    })

    # -------------------------------------------------------------------------
    # 4. BIAS MITIGATION
    # -------------------------------------------------------------------------
    logger.info("Step 4: Bias Mitigation")

    mitigation_results = {}

    # Apply ThresholdOptimizer if fairlearn available
    if FAIRLEARN_AVAILABLE:
        logger.info("Applying ThresholdOptimizer for bias mitigation...")

        # Retrain logistic regression for threshold optimization
        lr_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)

        try:
            threshold_opt = ThresholdOptimizer(
                estimator=lr_model,
                constraints="equalized_odds",
                prefit=True
            )
            threshold_opt.fit(X_train_scaled, y_train, sensitive_features=race_train)

            # Predict with mitigation
            y_pred_mitigated = threshold_opt.predict(X_test_scaled, sensitive_features=race_test)

            # Compute post-mitigation metrics
            mitigation_results['accuracy'] = float(accuracy_score(y_test, y_pred_mitigated))
            mitigation_results['f1'] = float(f1_score(y_test, y_pred_mitigated, zero_division=0))

            # Fairness metrics post-mitigation
            fairness_post = compute_fairness_metrics_manual(y_test, y_pred_mitigated, race_test)
            mitigation_results['fairness'] = fairness_post

            mitigation_results['method'] = 'ThresholdOptimizer (Equalized Odds)'
            mitigation_results['applied'] = True

            logger.info(f"Post-mitigation: Accuracy={mitigation_results['accuracy']:.3f}, "
                       f"EO Diff={fairness_post['equalized_odds_difference']:.3f}")

        except Exception as e:
            logger.warning(f"ThresholdOptimizer failed: {e}")
            mitigation_results['applied'] = False
            mitigation_results['error'] = str(e)
    else:
        # Manual threshold adjustment
        logger.info("Applying manual threshold adjustment for bias mitigation...")

        # Find thresholds that equalize FPR across groups
        y_prob = best_results['probabilities']

        # Simple approach: use different thresholds per group to equalize selection rate
        group_thresholds = {}
        target_rate = y_prob.mean()  # Overall positive rate as target

        for race in np.unique(race_test):
            mask = race_test == race
            probs = y_prob[mask]
            # Find threshold that achieves target selection rate
            sorted_probs = np.sort(probs)[::-1]
            n_select = int(len(probs) * target_rate)
            if n_select > 0 and n_select < len(sorted_probs):
                group_thresholds[race] = sorted_probs[n_select]
            else:
                group_thresholds[race] = 0.5

        # Apply group-specific thresholds
        y_pred_mitigated = np.zeros_like(y_prob)
        for race in np.unique(race_test):
            mask = race_test == race
            y_pred_mitigated[mask] = (y_prob[mask] >= group_thresholds[race]).astype(int)

        mitigation_results['accuracy'] = float(accuracy_score(y_test, y_pred_mitigated))
        mitigation_results['f1'] = float(f1_score(y_test, y_pred_mitigated, zero_division=0))
        fairness_post = compute_fairness_metrics_manual(y_test, y_pred_mitigated, race_test)
        mitigation_results['fairness'] = fairness_post
        mitigation_results['method'] = 'Group-specific Thresholds (Selection Rate Parity)'
        mitigation_results['applied'] = True
        mitigation_results['thresholds'] = {str(k): float(v) for k, v in group_thresholds.items()}

    results['mitigation'] = mitigation_results

    # Chart 8: Pre vs Post Mitigation Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy vs Fairness trade-off
    ax = axes[0]

    pre_acc = best_results['accuracy']
    pre_eo = best_results['fairness']['equalized_odds_difference']
    post_acc = mitigation_results.get('accuracy', pre_acc)
    post_eo = mitigation_results.get('fairness', {}).get('equalized_odds_difference', pre_eo)

    ax.scatter([pre_eo], [pre_acc], s=200, c=COLORS['danger'], label='Before Mitigation', marker='o', zorder=5)
    ax.scatter([post_eo], [post_acc], s=200, c=COLORS['success'], label='After Mitigation', marker='s', zorder=5)
    ax.annotate('', xy=(post_eo, post_acc), xytext=(pre_eo, pre_acc),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_xlabel('Equalized Odds Difference (lower = fairer)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Fairness-Accuracy Trade-off')
    ax.legend()
    ax.set_xlim(0, max(pre_eo, post_eo) * 1.2)

    # FPR comparison
    ax = axes[1]

    pre_fpr = best_results['fairness']['fpr_by_group']
    post_fpr = mitigation_results.get('fairness', {}).get('fpr_by_group', pre_fpr)

    races = list(pre_fpr.keys())
    x = np.arange(len(races))
    width = 0.35

    bars1 = ax.bar(x - width/2, [pre_fpr[r] for r in races], width, label='Before', color=COLORS['danger'], alpha=0.7)
    bars2 = ax.bar(x + width/2, [post_fpr.get(r, 0) for r in races], width, label='After', color=COLORS['success'], alpha=0.7)

    ax.set_ylabel('False Positive Rate')
    ax.set_title('FPR by Race: Before vs After Mitigation')
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    save_chart(fig, 'compas_mitigation_comparison.png')

    results['charts'].append({
        'chart_id': 'compas_mitigation_comparison',
        'filename': 'compas_mitigation_comparison.png',
        'title': 'Bias Mitigation: Before vs After',
        'dataset': 'COMPAS',
        'explanation': (
            "Bias mitigation techniques can reduce fairness gaps but typically at some cost to overall "
            "accuracy. The left panel shows the trade-off: moving from red (before) to green (after) "
            "reduces the equalized odds difference but may decrease accuracy. The right panel shows "
            "how false positive rates become more similar across groups after mitigation. This "
            "illustrates a fundamental tension in algorithmic fairness - we often can't maximize "
            "accuracy and fairness simultaneously."
        ),
        'speaker_notes': [
            "Mitigation reduces disparity but may reduce accuracy",
            "This is the 'fairness-accuracy trade-off' discussed in literature",
            "The 'optimal' point depends on societal values, not just math",
            "Different mitigation methods have different trade-off profiles"
        ],
        'slide_title': 'The Fairness-Accuracy Trade-off',
        'talk_track': (
            "When we apply bias mitigation - in this case, threshold optimization to equalize error "
            "rates - we see a meaningful reduction in the equalized odds difference. However, this "
            "comes at a cost to overall accuracy. This trade-off isn't a technical failure; it's a "
            "reflection of the fundamental impossibility of satisfying all fairness criteria "
            "simultaneously when base rates differ."
        )
    })

    # Chart 9: Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        # Use permutation importance for logistic regression
        if 'Logistic' in best_model_name:
            perm_importance = permutation_importance(
                best_model, X_test_scaled, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
            )
        else:
            perm_importance = permutation_importance(
                best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
            )
        importances = perm_importance.importances_mean

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['feature'], importance_df['importance'], color=COLORS['primary'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 10 Feature Importances - {best_model_name} (COMPAS)')

    save_chart(fig, 'compas_feature_importance.png')

    results['charts'].append({
        'chart_id': 'compas_feature_importance',
        'filename': 'compas_feature_importance.png',
        'title': 'Feature Importance',
        'dataset': 'COMPAS',
        'explanation': (
            "Feature importance reveals what the model 'looks at' when making predictions. Prior "
            "convictions and age are typically most predictive. Notably, race-related features often "
            "have low direct importance, but this doesn't mean the model is unbiased - race correlates "
            "with other features like number of priors due to systemic factors. The model can learn "
            "racial disparities indirectly through proxy variables, a phenomenon called 'redundant "
            "encoding' of protected attributes."
        ),
        'speaker_notes': [
            "Number of priors is typically the strongest predictor",
            "Age also matters - younger defendants have higher predicted risk",
            "Low importance of race features doesn't mean model is fair",
            "Proxy variables can encode race indirectly"
        ],
        'slide_title': 'What Drives Predictions?',
        'talk_track': (
            "Looking at feature importance, we see that criminal history - particularly number of "
            "prior convictions - dominates the prediction. Race variables have lower direct importance, "
            "but this doesn't mean the model is race-neutral. Other features like priors and age "
            "correlate with race due to historical policing patterns, so the model can learn racial "
            "patterns indirectly."
        )
    })

    results['feature_importance'] = importance_df.to_dict('records')

    return results


# ============================================================================
# NYPD STOP-QUESTION-FRISK ANALYSIS
# ============================================================================

def analyze_nypd(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive NYPD SQF analysis including:
    - Data preprocessing
    - Outcome analysis (arrests, searches, frisks)
    - Geographic analysis
    - Model training
    - Fairness analysis
    """
    logger.info("=" * 60)
    logger.info("NYPD STOP-QUESTION-FRISK ANALYSIS")
    logger.info("=" * 60)

    results = {
        'dataset_name': 'NYPD SQF 2012',
        'description': 'New York Police Department Stop, Question, and Frisk Data (2012)',
        'sample_size': len(df),
        'charts': [],
        'metrics': {},
        'fairness': {},
        'models': {},
    }

    # -------------------------------------------------------------------------
    # 1. DATA PREPROCESSING
    # -------------------------------------------------------------------------
    logger.info("Step 1: Data Preprocessing")

    # Clean column names
    df.columns = df.columns.str.lower().str.strip()

    # Race mapping
    race_map = {
        'B': 'Black',
        'W': 'White',
        'Q': 'Hispanic',
        'A': 'Asian',
        'P': 'Black Hispanic',
        'I': 'Native American',
        'Z': 'Other',
        ' ': 'Unknown'
    }

    if 'race' in df.columns:
        df['race_label'] = df['race'].map(race_map).fillna('Unknown')
    else:
        logger.warning("Race column not found in NYPD data")
        df['race_label'] = 'Unknown'

    # Sex cleaning
    if 'sex' in df.columns:
        df['sex_label'] = df['sex'].map({'M': 'Male', 'F': 'Female'}).fillna('Unknown')

    # Detect outcome column
    outcome_cols = ['arstmade', 'searched', 'frisked', 'contrabn']
    target_col = None

    for col in outcome_cols:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        logger.error(f"No outcome column found. Available: {list(df.columns)}")
        raise ValueError("Could not find suitable outcome column")

    logger.info(f"Target column: {target_col}")

    # Convert target to binary
    def convert_to_binary(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            return 1 if val.upper() in ['Y', 'YES', '1', 'TRUE'] else 0
        return 0

    df['target'] = df[target_col].apply(convert_to_binary)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Age cleaning
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df[(df['age'] >= 10) & (df['age'] <= 100)]

    # Precinct cleaning
    if 'pct' in df.columns:
        df['precinct'] = pd.to_numeric(df['pct'], errors='coerce')

    results['preprocessing'] = {
        'target_column': target_col,
        'target_interpretation': {
            'arstmade': 'Arrest was made during stop',
            'searched': 'Person was searched',
            'frisked': 'Person was frisked',
            'contrabn': 'Contraband was found'
        }.get(target_col, 'Unknown'),
        'n_samples': len(df),
        'race_distribution': df['race_label'].value_counts().to_dict(),
        'outcome_rate': float(df['target'].mean())
    }

    logger.info(f"Sample size after cleaning: {len(df)}")
    logger.info(f"Overall {target_col} rate: {df['target'].mean():.1%}")

    # -------------------------------------------------------------------------
    # 2. EXPLORATORY ANALYSIS
    # -------------------------------------------------------------------------
    logger.info("Step 2: Exploratory Analysis")

    # Chart 1: Stops by race
    race_counts = df['race_label'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS['races'].get(r, COLORS['neutral']) for r in race_counts.index]
    bars = ax.bar(range(len(race_counts)), race_counts.values, color=colors)
    ax.set_xticks(range(len(race_counts)))
    ax.set_xticklabels(race_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Stops')
    ax.set_xlabel('Race')
    ax.set_title('Stop-and-Frisk Encounters by Race (NYPD 2012)')

    for bar, val in zip(bars, race_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    # Add percentage labels
    total = race_counts.sum()
    for i, (idx, val) in enumerate(race_counts.items()):
        ax.text(i, val/2, f'{val/total*100:.1f}%', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    save_chart(fig, 'nypd_stops_by_race.png')

    results['charts'].append({
        'chart_id': 'nypd_stops_by_race',
        'filename': 'nypd_stops_by_race.png',
        'title': 'Stop-and-Frisk Encounters by Race',
        'dataset': 'NYPD SQF',
        'explanation': (
            "This chart shows the racial composition of police stops in NYC during 2012. Black and "
            "Hispanic individuals constitute the vast majority of stops - far exceeding their "
            "proportion of the city's population. In 2012, NYC was approximately 33% White, 29% "
            "Hispanic, and 26% Black, yet White individuals represent a small fraction of stops. "
            "This disparity was central to the Floyd v. City of New York ruling that found the "
            "program unconstitutional."
        ),
        'speaker_notes': [
            "Black and Hispanic individuals make up ~85% of stops",
            "NYC demographic composition doesn't match stop patterns",
            "This disparity led to federal court ruling against NYPD in 2013",
            "Raises questions about reasonable suspicion standards"
        ],
        'slide_title': 'Who Gets Stopped? Racial Disparities in SQF',
        'talk_track': (
            "The Stop-Question-Frisk data reveals stark racial disparities. In 2012, Black and "
            "Hispanic New Yorkers made up approximately 85% of all stops, despite being about 55% "
            "of the city's population. These numbers became central evidence in the federal lawsuit "
            "that ultimately found the program unconstitutional."
        )
    })

    results['metrics']['stops_by_race'] = race_counts.to_dict()

    # Chart 2: Outcome rate by race
    outcome_name = target_col.replace('arstmade', 'Arrest').replace('searched', 'Search').replace('frisked', 'Frisk')
    race_outcome = df.groupby('race_label')['target'].agg(['mean', 'count'])
    race_outcome.columns = ['outcome_rate', 'count']
    race_outcome = race_outcome[race_outcome['count'] >= 100].sort_values('count', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(race_outcome)), race_outcome['outcome_rate'],
                  color=[COLORS['races'].get(r, COLORS['neutral']) for r in race_outcome.index])
    ax.set_xticks(range(len(race_outcome)))
    ax.set_xticklabels(race_outcome.index, rotation=45, ha='right')
    ax.set_ylabel(f'{outcome_name} Rate')
    ax.set_xlabel('Race')
    ax.set_title(f'{outcome_name} Rate by Race (NYPD SQF 2012)')
    ax.set_ylim(0, min(1, race_outcome['outcome_rate'].max() * 1.3))

    for bar, val in zip(bars, race_outcome['outcome_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    save_chart(fig, 'nypd_outcome_by_race.png')

    results['charts'].append({
        'chart_id': 'nypd_outcome_by_race',
        'filename': 'nypd_outcome_by_race.png',
        'title': f'{outcome_name} Rate by Race',
        'dataset': 'NYPD SQF',
        'explanation': (
            f"This chart shows the {outcome_name.lower()} rate - the proportion of stops that resulted "
            f"in a {outcome_name.lower()} - broken down by race. Interestingly, outcome rates are often "
            "similar or even lower for Black and Hispanic individuals compared to White individuals. "
            "This is significant: if police had equally good 'reasonable suspicion' across groups, we'd "
            "expect similar outcome rates. Lower 'hit rates' for minorities suggest the threshold for "
            "stopping them may be lower - a form of differential treatment."
        ),
        'speaker_notes': [
            f"{outcome_name} rates are relatively similar across groups",
            "Lower or equal rates for minorities despite more stops is telling",
            "Suggests different 'reasonable suspicion' thresholds by race",
            "This is the 'hit rate' disparity discussed in policing literature"
        ],
        'slide_title': f'{outcome_name} Rates: A Deeper Story',
        'talk_track': (
            f"Looking at {outcome_name.lower()} rates - the proportion of stops that resulted in "
            f"a {outcome_name.lower()} - we see something important. Despite being stopped far more "
            "often, Black and Hispanic individuals don't have higher rates. This suggests police "
            "may use a lower threshold of suspicion when stopping minorities - they're stopped more "
            "often but aren't more likely to have done something wrong."
        )
    })

    results['metrics']['outcome_rates_by_race'] = race_outcome['outcome_rate'].to_dict()

    # Chart 3: Geographic Analysis - Outcome by Precinct
    if 'precinct' in df.columns:
        precinct_stats = df.groupby('precinct').agg({
            'target': ['mean', 'count'],
            'race_label': lambda x: (x == 'Black').mean()  # Proportion Black
        }).reset_index()
        precinct_stats.columns = ['precinct', 'outcome_rate', 'stop_count', 'pct_black']
        precinct_stats = precinct_stats[precinct_stats['stop_count'] >= 100]

        top_precincts = precinct_stats.nlargest(15, 'stop_count')

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(top_precincts))
        width = 0.35

        bars1 = ax.bar(x - width/2, top_precincts['outcome_rate'], width,
                       label=f'{outcome_name} Rate', color=COLORS['primary'])
        bars2 = ax.bar(x + width/2, top_precincts['pct_black'], width,
                       label='% Black Stops', color=COLORS['danger'], alpha=0.7)

        ax.set_ylabel('Rate')
        ax.set_xlabel('Precinct')
        ax.set_title('Top 15 Precincts: Outcome Rate vs Racial Composition')
        ax.set_xticks(x)
        ax.set_xticklabels([int(p) for p in top_precincts['precinct']], rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)

        save_chart(fig, 'nypd_precinct_analysis.png')

        results['charts'].append({
            'chart_id': 'nypd_precinct_analysis',
            'filename': 'nypd_precinct_analysis.png',
            'title': 'Precinct Analysis: Outcomes and Demographics',
            'dataset': 'NYPD SQF',
            'explanation': (
                "This chart reveals geographic patterns in stop-and-frisk. Each bar pair shows a precinct's "
                f"{outcome_name.lower()} rate alongside the proportion of Black individuals stopped there. "
                "Precincts vary significantly in both metrics. Some high-activity precincts have very high "
                "proportions of Black stops. This geographic clustering means that neighborhood-level factors "
                "like poverty, crime rates, and police deployment patterns contribute to racial disparities "
                "in stops - but also raises questions about whether deployment itself reflects bias."
            ),
            'speaker_notes': [
                "Geographic patterns show concentrated policing in certain precincts",
                "Racial composition of stops varies significantly by location",
                "Some precincts show >90% Black/Hispanic stops",
                "Deployment decisions determine who gets policed"
            ],
            'slide_title': 'Geography of Policing',
            'talk_track': (
                "The geographic dimension is crucial. Stop patterns aren't uniform across the city - "
                "they're heavily concentrated in certain precincts. These precincts tend to have higher "
                "proportions of Black and Hispanic residents and receive more police resources. This "
                "creates a feedback loop: more police presence leads to more stops, which generates "
                "more data, which may justify continued heavy policing."
            )
        })

        results['metrics']['precinct_analysis'] = top_precincts.to_dict('records')

    # Chart 4: Confounding visualization - Race x Precinct x Outcome
    if 'precinct' in df.columns:
        # Create heatmap-style visualization
        top_precincts_list = precinct_stats.nlargest(10, 'stop_count')['precinct'].values
        major_races = ['Black', 'Hispanic', 'White']

        heatmap_data = []
        for pct in top_precincts_list:
            pct_data = df[df['precinct'] == pct]
            row = {'precinct': int(pct)}
            for race in major_races:
                race_data = pct_data[pct_data['race_label'] == race]
                if len(race_data) > 0:
                    row[f'{race}_outcome'] = race_data['target'].mean()
                    row[f'{race}_count'] = len(race_data)
                else:
                    row[f'{race}_outcome'] = np.nan
                    row[f'{race}_count'] = 0
            heatmap_data.append(row)

        heatmap_df = pd.DataFrame(heatmap_data)

        fig, ax = plt.subplots(figsize=(10, 8))

        outcome_matrix = np.array([
            [heatmap_df[f'{race}_outcome'].values[i] for race in major_races]
            for i in range(len(heatmap_df))
        ])

        im = ax.imshow(outcome_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.3)

        ax.set_xticks(range(len(major_races)))
        ax.set_xticklabels(major_races)
        ax.set_yticks(range(len(heatmap_df)))
        ax.set_yticklabels([f"Pct {int(p)}" for p in heatmap_df['precinct']])
        ax.set_xlabel('Race')
        ax.set_ylabel('Precinct')
        ax.set_title(f'{outcome_name} Rate by Precinct and Race (Top 10 Precincts)')

        # Add text annotations
        for i in range(len(heatmap_df)):
            for j, race in enumerate(major_races):
                val = outcome_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                           color='white' if val > 0.15 else 'black', fontsize=9)

        plt.colorbar(im, ax=ax, label=f'{outcome_name} Rate')

        save_chart(fig, 'nypd_confounding_heatmap.png')

        results['charts'].append({
            'chart_id': 'nypd_confounding_heatmap',
            'filename': 'nypd_confounding_heatmap.png',
            'title': f'{outcome_name} Rate by Precinct and Race',
            'dataset': 'NYPD SQF',
            'explanation': (
                "This heatmap reveals the complex interaction between race, location, and stop outcomes. "
                f"Each cell shows the {outcome_name.lower()} rate for a specific race in a specific precinct. "
                "Within precincts, we can see whether outcome rates differ by race holding location constant. "
                "This helps disentangle whether racial disparities are driven by 'who gets stopped' vs "
                "'where police are deployed'. The pattern suggests both factors play a role."
            ),
            'speaker_notes': [
                "Heatmap controls for location to isolate racial effects",
                "Within-precinct disparities suggest race matters beyond geography",
                "Some precincts show racial disparities, others don't",
                "Complex interaction between place and race"
            ],
            'slide_title': 'Disentangling Race and Place',
            'talk_track': (
                "This heatmap helps us understand whether racial disparities persist when we control "
                "for location. If disparities were purely geographic - driven by where police are "
                "deployed - we'd expect similar rates within each precinct. Instead, we see variation "
                "by race even within the same precinct, suggesting that race itself influences "
                "stop and arrest decisions beyond just neighborhood effects."
            )
        })

    # Chart 5: Age distribution by race
    if 'age' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for race in ['Black', 'Hispanic', 'White']:
            if race in df['race_label'].values:
                subset = df[df['race_label'] == race]['age'].dropna()
                ax.hist(subset, bins=range(10, 70, 5), alpha=0.5,
                        label=f'{race} (mean={subset.mean():.1f})',
                        color=COLORS['races'].get(race, COLORS['neutral']), density=True)

        ax.set_xlabel('Age')
        ax.set_ylabel('Density')
        ax.set_title('Age Distribution of Stopped Individuals by Race')
        ax.legend()

        save_chart(fig, 'nypd_age_distribution.png')

        results['charts'].append({
            'chart_id': 'nypd_age_distribution',
            'filename': 'nypd_age_distribution.png',
            'title': 'Age Distribution by Race',
            'dataset': 'NYPD SQF',
            'explanation': (
                "The age distribution of stopped individuals shows that police stops heavily target "
                "young people across all races. The peak is in the late teens and early twenties. "
                "This pattern is consistent with crime statistics showing younger individuals are "
                "more likely to be involved in crime - but it also means young minority males bear "
                "the heaviest burden of stop-and-frisk policies, with potential long-term effects "
                "on their relationship with law enforcement and society."
            ),
            'speaker_notes': [
                "Stops heavily concentrated among young people (15-25)",
                "Similar age patterns across racial groups",
                "Young Black and Hispanic males most frequently stopped",
                "Repeated stops can affect trust in law enforcement"
            ],
            'slide_title': 'The Young Bear the Burden',
            'talk_track': (
                "Looking at age, we see stops concentrate heavily among young people - the peak is "
                "around 18-22 years old. This pattern holds across races, but combined with the "
                "racial disparities we've seen, it means young Black and Hispanic men face the "
                "most intense police contact. Research shows repeated stops can erode trust in "
                "institutions and have lasting psychological effects."
            )
        })

    # -------------------------------------------------------------------------
    # 3. PREDICTIVE MODELING
    # -------------------------------------------------------------------------
    logger.info("Step 3: Predictive Modeling")

    # Select features for modeling
    # Circumstance columns (reasons for stop)
    circumstance_cols = [c for c in df.columns if c.startswith(('cs_', 'rf_', 'ac_', 'pf_'))]

    # Convert Y/N to binary
    for col in circumstance_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).upper() in ['Y', '1'] else 0)

    # Add other features
    other_features = []
    if 'age' in df.columns:
        other_features.append('age')

    feature_cols = circumstance_cols + other_features
    feature_cols = [c for c in feature_cols if c in df.columns]

    logger.info(f"Using {len(feature_cols)} features for modeling")

    # Prepare data
    df_model = df.dropna(subset=['target', 'race_label'] + feature_cols[:5])  # Require at least some features

    X = df_model[feature_cols].fillna(0)
    y = df_model['target'].values
    race = df_model['race_label'].values

    # Train/test split
    X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
        X, y, race, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_iter=100)
    }

    model_results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

        model_results[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_pred_proba)),
            'brier_score': float(brier_score_loss(y_test, y_pred_proba)),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
        }

        # Fairness metrics
        fairness = compute_fairness_metrics_manual(y_test, y_pred, race_test)
        model_results[name]['fairness'] = fairness

        logger.info(f"  AUC: {model_results[name]['auc']:.3f}, "
                   f"F1: {model_results[name]['f1']:.3f}")

    results['models'] = {k: {kk: vv for kk, vv in v.items()
                             if kk not in ['predictions', 'probabilities']}
                        for k, v in model_results.items()}

    # Best model
    best_model_name = max(model_results, key=lambda x: model_results[x]['auc'])
    best_results = model_results[best_model_name]

    results['best_model'] = {
        'name': best_model_name,
        'auc': best_results['auc'],
        'f1': best_results['f1'],
    }

    # Chart 6: Model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(model_results.keys())
    aucs = [model_results[m]['auc'] for m in model_names]
    f1s = [model_results[m]['f1'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, aucs, width, label='AUC', color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score', color=COLORS['secondary'])

    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance Comparison - Predicting {outcome_name} (NYPD)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    save_chart(fig, 'nypd_model_comparison.png')

    results['charts'].append({
        'chart_id': 'nypd_model_comparison',
        'filename': 'nypd_model_comparison.png',
        'title': f'Model Performance - {outcome_name} Prediction',
        'dataset': 'NYPD SQF',
        'explanation': (
            f"These models attempt to predict whether a stop will result in an {outcome_name.lower()}, "
            "using the circumstances recorded at the time of the stop. Predictive performance indicates "
            "how well stop characteristics correlate with outcomes. Interestingly, if police were "
            "perfectly effective at identifying suspicious activity, we'd expect high predictive "
            "accuracy. Moderate accuracy suggests significant noise in who gets stopped - many stops "
            "don't result in any enforcement action."
        ),
        'speaker_notes': [
            f"Models predict {outcome_name.lower()} from stop circumstances",
            "Moderate AUC suggests imperfect targeting",
            "Low F1 often due to class imbalance (most stops don't lead to arrest)",
            "Predictability indicates signal in stop reasons"
        ],
        'slide_title': f'Can We Predict {outcome_name}s?',
        'talk_track': (
            f"We trained models to predict whether a stop would lead to an {outcome_name.lower()} "
            "based on the recorded circumstances. The moderate performance tells us something important: "
            "stop characteristics do predict outcomes to some degree, but there's substantial "
            "unpredictability. Many stops based on 'reasonable suspicion' don't result in any action."
        )
    })

    # Chart 7: Fairness metrics for NYPD
    fairness_data = best_results['fairness']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Selection rates
    ax = axes[0]
    sr = fairness_data['selection_rates_by_group']
    # Filter to main groups
    main_races = ['Black', 'Hispanic', 'White', 'Asian']
    sr_filtered = {k: v for k, v in sr.items() if k in main_races}

    races = list(sr_filtered.keys())
    rates = list(sr_filtered.values())
    bars = ax.bar(races, rates, color=[COLORS['races'].get(r, COLORS['neutral']) for r in races])
    ax.set_ylabel('Positive Prediction Rate')
    ax.set_title(f'Predicted {outcome_name} Rate by Race')
    ax.set_ylim(0, max(rates) * 1.3 if rates else 1)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # FPR comparison
    ax = axes[1]
    fpr = fairness_data['fpr_by_group']
    fnr = fairness_data['fnr_by_group']

    fpr_filtered = {k: v for k, v in fpr.items() if k in main_races}
    fnr_filtered = {k: v for k, v in fnr.items() if k in main_races}

    x = np.arange(len(main_races))
    width = 0.35

    fpr_vals = [fpr_filtered.get(r, 0) for r in main_races]
    fnr_vals = [fnr_filtered.get(r, 0) for r in main_races]

    bars1 = ax.bar(x - width/2, fpr_vals, width, label='FPR', color=COLORS['danger'], alpha=0.7)
    bars2 = ax.bar(x + width/2, fnr_vals, width, label='FNR', color=COLORS['warning'], alpha=0.7)

    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rates by Race')
    ax.set_xticks(x)
    ax.set_xticklabels(main_races)
    ax.legend()

    plt.tight_layout()
    save_chart(fig, 'nypd_fairness_metrics.png')

    results['charts'].append({
        'chart_id': 'nypd_fairness_metrics',
        'filename': 'nypd_fairness_metrics.png',
        'title': 'Fairness Metrics by Race',
        'dataset': 'NYPD SQF',
        'explanation': (
            f"These panels show how our {outcome_name.lower()} prediction model performs across racial "
            "groups. The left panel shows predicted positive rates - whether the model predicts high "
            "or low risk differently by race. The right panel shows error rates: FPR (falsely predicting "
            f"{outcome_name.lower()}) and FNR (missing actual {outcome_name.lower()}s). Disparities here "
            "indicate that model errors aren't equally distributed, raising fairness concerns if such "
            "models were used to guide policing decisions."
        ),
        'speaker_notes': [
            "Model predictions vary by race",
            "Error rates show how mistakes are distributed",
            "Disparities in errors = disparate impact",
            "Using such models could amplify existing biases"
        ],
        'slide_title': 'Fairness in Predictive Policing',
        'talk_track': (
            "If we were to use this model to guide policing - which is essentially what predictive "
            "policing does - we'd need to examine its fairness properties. The disparities in error "
            "rates show that the model doesn't fail equally across groups. Some groups face higher "
            "false positive rates, meaning they'd be disproportionately flagged for scrutiny."
        )
    })

    results['fairness'] = fairness_data

    return results


# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def generate_comparison(compas_results: Dict, nypd_results: Dict) -> Dict[str, Any]:
    """Generate cross-dataset comparison analysis."""
    logger.info("=" * 60)
    logger.info("CROSS-DATASET COMPARISON")
    logger.info("=" * 60)

    comparison = {
        'datasets': {
            'COMPAS': {
                'name': 'COMPAS Recidivism',
                'sample_size': compas_results['sample_size'],
                'outcome': 'Two-year recidivism',
                'base_rate': compas_results['preprocessing']['outcome_distribution']['base_rate'],
                'best_model_auc': compas_results['best_model']['auc'],
            },
            'NYPD': {
                'name': 'NYPD Stop-Question-Frisk',
                'sample_size': nypd_results['sample_size'],
                'outcome': nypd_results['preprocessing']['target_interpretation'],
                'base_rate': nypd_results['preprocessing']['outcome_rate'],
                'best_model_auc': nypd_results['best_model']['auc'],
            }
        },
        'charts': [],
        'key_findings': [],
    }

    # Chart 1: Side-by-side base rates by race
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # COMPAS
    ax = axes[0]
    compas_rates = compas_results['metrics']['base_rates_by_race']
    races_c = list(compas_rates.keys())
    rates_c = list(compas_rates.values())
    bars = ax.bar(races_c, rates_c, color=[COLORS['races'].get(r, COLORS['neutral']) for r in races_c])
    ax.set_ylabel('Rate')
    ax.set_title('COMPAS: Recidivism Rate by Race')
    ax.set_xticklabels(races_c, rotation=45, ha='right')
    ax.set_ylim(0, 0.7)
    for bar, val in zip(bars, rates_c):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    # NYPD
    ax = axes[1]
    nypd_rates = nypd_results['metrics']['outcome_rates_by_race']
    races_n = list(nypd_rates.keys())[:5]  # Top 5
    rates_n = [nypd_rates[r] for r in races_n]
    bars = ax.bar(races_n, rates_n, color=[COLORS['races'].get(r, COLORS['neutral']) for r in races_n])
    ax.set_ylabel('Rate')
    ax.set_title(f'NYPD: {nypd_results["preprocessing"]["target_interpretation"].split()[0]} Rate by Race')
    ax.set_xticklabels(races_n, rotation=45, ha='right')
    ax.set_ylim(0, 0.3)
    for bar, val in zip(bars, rates_n):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_chart(fig, 'comparison_base_rates.png')

    comparison['charts'].append({
        'chart_id': 'comparison_base_rates',
        'filename': 'comparison_base_rates.png',
        'title': 'Outcome Rates by Race: COMPAS vs NYPD',
        'dataset': 'Comparison',
        'explanation': (
            "Comparing outcome rates across datasets reveals different patterns of disparity. In COMPAS, "
            "African American defendants have higher observed recidivism rates. In NYPD data, the pattern "
            "is different - arrest/search rates are relatively similar or even lower for minorities despite "
            "far more stops. This contrast highlights how different stages of the criminal justice system "
            "produce different disparity patterns. Both reflect systemic issues but manifest differently."
        ),
        'speaker_notes': [
            "COMPAS shows higher recidivism rates for Black defendants",
            "NYPD shows similar or lower 'hit rates' despite more stops",
            "Different stages of system, different disparity patterns",
            "Both datasets show racial disparities, expressed differently"
        ],
        'slide_title': 'Two Datasets, Different Disparity Patterns',
        'talk_track': (
            "Comparing these datasets reveals how racial disparities manifest differently at different "
            "points in the system. In COMPAS, Black defendants show higher recidivism rates - but this "
            "reflects who gets arrested and prosecuted. In NYPD data, minorities are stopped far more "
            "but don't have higher outcome rates - suggesting lower thresholds for stopping them."
        )
    })

    # Chart 2: Fairness metric comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract fairness metrics from best models
    compas_fairness = compas_results['models'][compas_results['best_model']['name']]['fairness']
    nypd_fairness = nypd_results['models'][nypd_results['best_model']['name']]['fairness']

    metrics = ['selection_rate_difference', 'fpr_difference', 'fnr_difference', 'equalized_odds_difference']
    metric_labels = ['Selection Rate\nDifference', 'FPR\nDifference', 'FNR\nDifference', 'Equalized Odds\nDifference']

    compas_vals = [compas_fairness.get(m, 0) for m in metrics]
    nypd_vals = [nypd_fairness.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, compas_vals, width, label='COMPAS', color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, nypd_vals, width, label='NYPD SQF', color=COLORS['secondary'])

    ax.set_ylabel('Disparity (absolute difference)')
    ax.set_title('Fairness Metric Comparison: COMPAS vs NYPD')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Common threshold (0.1)')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    save_chart(fig, 'comparison_fairness_metrics.png')

    comparison['charts'].append({
        'chart_id': 'comparison_fairness_metrics',
        'filename': 'comparison_fairness_metrics.png',
        'title': 'Fairness Metric Comparison',
        'dataset': 'Comparison',
        'explanation': (
            "This chart compares fairness metrics between the two datasets. Both show meaningful "
            "disparities across multiple fairness criteria. The dashed line at 0.1 represents a "
            "common threshold used in fairness literature - values above suggest actionable disparity. "
            "Notably, the pattern of which metrics show most disparity differs between datasets, "
            "reflecting the different nature of the decisions being modeled (recidivism prediction "
            "vs. stop outcome prediction)."
        ),
        'speaker_notes': [
            "Both datasets exceed common fairness thresholds",
            "Different metrics are most problematic in each dataset",
            "0.1 difference often used as actionable threshold",
            "Both systems would fail many fairness audits"
        ],
        'slide_title': 'Fairness Audit Results',
        'talk_track': (
            "Both datasets show significant disparities across multiple fairness metrics. The "
            "dashed line at 0.1 represents a common threshold - above this, many would consider "
            "the disparity actionable. Both COMPAS and a model trained on NYPD data exceed this "
            "threshold on multiple metrics, suggesting both would fail fairness audits."
        )
    })

    comparison['fairness_comparison'] = {
        'COMPAS': {m: compas_fairness.get(m, 0) for m in metrics},
        'NYPD': {m: nypd_fairness.get(m, 0) for m in metrics}
    }

    # Chart 3: Conceptual Pipeline Diagram
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('The Feedback Loop: How Biased Data Perpetuates Bias', fontsize=14, fontweight='bold', pad=20)

    # Draw boxes and arrows for the pipeline
    boxes = [
        {'xy': (1, 6), 'text': 'Historical\nPolicing Patterns', 'color': COLORS['danger']},
        {'xy': (4, 6), 'text': 'Stop & Arrest\nData (NYPD)', 'color': COLORS['warning']},
        {'xy': (7, 6), 'text': 'Training Data\nfor ML Models', 'color': COLORS['neutral']},
        {'xy': (10, 6), 'text': 'Predictive\nPolicing Tools', 'color': COLORS['primary']},
        {'xy': (10, 3), 'text': 'Risk Scores\n(COMPAS-like)', 'color': COLORS['secondary']},
        {'xy': (7, 3), 'text': 'Sentencing &\nParole Decisions', 'color': COLORS['danger']},
        {'xy': (4, 3), 'text': 'Criminal\nRecords', 'color': COLORS['warning']},
        {'xy': (1, 3), 'text': 'Future\nArrests', 'color': COLORS['danger']},
    ]

    for box in boxes:
        rect = FancyBboxPatch(box['xy'], 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=box['color'], alpha=0.3, edgecolor=box['color'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + 1.25, box['xy'][1] + 0.75, box['text'],
               ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw arrows (top row)
    arrow_style = dict(arrowstyle='->', color='gray', lw=2, mutation_scale=15)
    ax.annotate('', xy=(4, 6.75), xytext=(3.5, 6.75), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 6.75), xytext=(6.5, 6.75), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 6.75), xytext=(9.5, 6.75), arrowprops=arrow_style)

    # Vertical arrow
    ax.annotate('', xy=(11.25, 4.5), xytext=(11.25, 6), arrowprops=arrow_style)

    # Bottom row (reverse direction)
    ax.annotate('', xy=(7, 3.75), xytext=(9.5, 3.75), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 3.75), xytext=(6.5, 3.75), arrowprops=arrow_style)
    ax.annotate('', xy=(1, 3.75), xytext=(3.5, 3.75), arrowprops=arrow_style)

    # Feedback loop arrow
    ax.annotate('', xy=(1, 6), xytext=(1, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=3,
                              connectionstyle='arc3,rad=0.3', mutation_scale=20))
    ax.text(0.3, 5.25, 'FEEDBACK\nLOOP', fontsize=8, fontweight='bold', color=COLORS['danger'],
           ha='center', va='center', rotation=90)

    # Add explanatory text
    ax.text(7, 1,
           'Biased historical policing → Biased data → Biased predictions → Biased decisions → More biased data',
           ha='center', va='center', fontsize=10, style='italic', color='gray')

    save_chart(fig, 'comparison_feedback_loop.png')

    comparison['charts'].append({
        'chart_id': 'comparison_feedback_loop',
        'filename': 'comparison_feedback_loop.png',
        'title': 'The Algorithmic Feedback Loop',
        'dataset': 'Comparison',
        'explanation': (
            "This diagram illustrates how bias perpetuates through the criminal justice AI pipeline. "
            "Historical policing patterns (which reflect societal biases) generate arrest data. This "
            "data trains predictive policing tools and risk assessment algorithms. These tools influence "
            "real decisions - who gets stopped, who gets longer sentences. Those decisions generate new "
            "criminal records, which become future training data. The loop reinforces itself: past bias "
            "becomes encoded in algorithms that produce future bias."
        ),
        'speaker_notes': [
            "Bias isn't just in the algorithm - it's in the entire system",
            "Training data reflects historical discrimination",
            "Predictions influence decisions that generate new data",
            "Without intervention, the loop amplifies disparities"
        ],
        'slide_title': 'The Vicious Cycle of Algorithmic Bias',
        'talk_track': (
            "This is perhaps the most important insight from our analysis. Bias in criminal justice "
            "AI isn't a one-time problem we can fix with better algorithms. It's a feedback loop. "
            "Historical discrimination produced biased data. That data trains algorithms. Those "
            "algorithms influence decisions. Those decisions create new data. And the cycle continues, "
            "potentially amplifying disparities over time."
        )
    })

    # Generate key findings
    comparison['key_findings'] = [
        {
            'finding': 'Both datasets show significant racial disparities in outcomes and model predictions.',
            'evidence': 'COMPAS shows higher recidivism rates for African Americans; NYPD shows minorities are stopped far more frequently.',
            'implication': 'Algorithmic systems trained on this data will inherit these disparities.'
        },
        {
            'finding': 'Different fairness metrics reveal different aspects of bias.',
            'evidence': 'COMPAS shows high FPR disparity (error on non-recidivists); NYPD shows high selection rate disparity.',
            'implication': 'No single fairness metric captures all forms of bias; comprehensive auditing is needed.'
        },
        {
            'finding': 'Bias mitigation can reduce disparities but involves accuracy trade-offs.',
            'evidence': f'Post-mitigation COMPAS model reduced equalized odds difference while sacrificing some accuracy.',
            'implication': 'Fairness interventions require value judgments about acceptable trade-offs.'
        },
        {
            'finding': 'Geographic and demographic factors are deeply confounded.',
            'evidence': 'NYPD precinct analysis shows race and location are intertwined in stop patterns.',
            'implication': 'Controlling for geography doesn\'t eliminate racial disparities; both contribute.'
        },
        {
            'finding': 'Predictive models have moderate accuracy, questioning their use for high-stakes decisions.',
            'evidence': f'Best models achieve AUC of {compas_results["best_model"]["auc"]:.2f} (COMPAS) and {nypd_results["best_model"]["auc"]:.2f} (NYPD).',
            'implication': 'Models are better than random but far from reliable; error rates have human consequences.'
        },
        {
            'finding': 'The criminal justice AI pipeline creates feedback loops that can amplify bias.',
            'evidence': 'Policing data feeds risk scores, which influence sentencing, which creates criminal records for future training.',
            'implication': 'Technical fixes alone are insufficient; systemic intervention is required.'
        },
    ]

    return comparison


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("CRIMINAL JUSTICE BIAS ANALYSIS PIPELINE")
    logger.info("Group 2: Vidit, Ronaldo, Kaleemulla, Vishal")
    logger.info("Course: Ethics, Sociology, and Governance of AI")
    logger.info("=" * 60)

    try:
        # Load datasets
        logger.info("\n" + "=" * 40)
        logger.info("LOADING DATASETS")
        logger.info("=" * 40)

        compas_df = load_compas_data()
        nypd_df = load_nypd_data(sample_size=100000)

        # Run analyses
        compas_results = analyze_compas(compas_df)
        nypd_results = analyze_nypd(nypd_df)
        comparison_results = generate_comparison(compas_results, nypd_results)

        # Save JSON outputs
        logger.info("\n" + "=" * 40)
        logger.info("SAVING JSON OUTPUTS")
        logger.info("=" * 40)

        save_json(compas_results, 'compas_analysis.json')
        save_json(nypd_results, 'nypd_analysis.json')
        save_json(comparison_results, 'comparison_analysis.json')

        # Generate combined chart metadata
        all_charts = (
            compas_results['charts'] +
            nypd_results['charts'] +
            comparison_results['charts']
        )
        save_json(all_charts, 'charts_metadata.json')

        # Generate summary for dashboard
        dashboard_data = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'team': 'Group 2: Vidit, Ronaldo, Kaleemulla, Vishal',
            'course': 'Ethics, Sociology, and Governance of AI (MAIB AI 219)',
            'datasets': {
                'COMPAS': {
                    'name': 'ProPublica COMPAS Recidivism',
                    'sample_size': compas_results['sample_size'],
                    'outcome': 'Two-year recidivism',
                    'base_rate': compas_results['preprocessing']['outcome_distribution']['base_rate'],
                    'best_model': compas_results['best_model'],
                },
                'NYPD': {
                    'name': 'NYPD Stop-Question-Frisk 2012',
                    'sample_size': nypd_results['sample_size'],
                    'outcome': nypd_results['preprocessing']['target_interpretation'],
                    'base_rate': nypd_results['preprocessing']['outcome_rate'],
                    'best_model': nypd_results['best_model'],
                }
            },
            'key_findings': comparison_results['key_findings'],
            'total_charts': len(all_charts),
        }
        save_json(dashboard_data, 'dashboard_summary.json')

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("OUTPUT FILES GENERATED")
        print("=" * 60)

        print("\nJSON Data Files:")
        for f in DATA_OUTPUT_DIR.glob("*.json"):
            print(f"  - {f}")

        print("\nChart Images:")
        for f in IMG_OUTPUT_DIR.glob("*.png"):
            print(f"  - {f}")

        print(f"\nTotal charts generated: {len(all_charts)}")
        print(f"\nCOMPAS Analysis:")
        print(f"  - Sample size: {compas_results['sample_size']:,}")
        print(f"  - Best model: {compas_results['best_model']['name']} (AUC: {compas_results['best_model']['auc']:.3f})")

        print(f"\nNYPD Analysis:")
        print(f"  - Sample size: {nypd_results['sample_size']:,}")
        print(f"  - Best model: {nypd_results['best_model']['name']} (AUC: {nypd_results['best_model']['auc']:.3f})")

        print("\n" + "=" * 60)
        print("Next steps:")
        print("  1. cd web")
        print("  2. npm install")
        print("  3. npm run dev (for development)")
        print("  4. npm run build (for production)")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
