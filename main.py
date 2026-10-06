import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import VotingClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Academic plot styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def load_data(filepath):
    df_raw = pd.read_excel(filepath, sheet_name='Gut microbiota data')
    headers = df_raw.iloc[0].values
    df = df_raw.iloc[1:].copy()
    df.columns = headers

    X = df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_raw = df['AIDs'].astype(str)

    # Sanitize feature names for tree algorithms
    X.columns = [f"taxa_{i}_" + "".join([c if c.isalnum() else "_" for c in col[-25:]]) for i, col in enumerate(X.columns)]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    return X, y, le.classes_


def tune_hyperparameters(X_tr, y_tr):
    print("[+] Optimizing model hyperparameters (Max Depth, Learning Rate, Estimators, Subsample)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. XGBoost Optimization
    def obj_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }
        clf = xgb.XGBClassifier(**params)
        scores = [roc_auc_score(y_tr[v], clf.fit(X_tr.iloc[t], y_tr[t]).predict_proba(X_tr.iloc[v]), multi_class='ovr') 
                  for t, v in cv.split(X_tr, y_tr)]
        return np.mean(scores)

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(obj_xgb, n_trials=10)
    p_xgb = study_xgb.best_params
    p_xgb.update({'eval_metric': 'mlogloss', 'random_state': 42, 'n_jobs': -1})

    # 2. LightGBM Optimization
    def obj_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'subsample_freq': 1,
            'objective': 'multiclass',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        clf = lgb.LGBMClassifier(**params)
        scores = [roc_auc_score(y_tr[v], clf.fit(X_tr.iloc[t], y_tr[t]).predict_proba(X_tr.iloc[v]), multi_class='ovr') 
                  for t, v in cv.split(X_tr, y_tr)]
        return np.mean(scores)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(obj_lgb, n_trials=10)
    p_lgb = study_lgb.best_params
    p_lgb.update({'objective': 'multiclass', 'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'subsample_freq': 1})

    # 3. CatBoost Optimization (Updated with Bernoulli bootstrap)
    def obj_cat(trial):
        params = {
            'iterations': trial.suggest_int('n_estimators', 100, 300, step=50),
            'depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'bootstrap_type': 'Bernoulli',
            'loss_function': 'MultiClass',
            'random_seed': 42,
            'verbose': 0
        }
        clf = CatBoostClassifier(**params)
        scores = [roc_auc_score(y_tr[v], clf.fit(X_tr.iloc[t], y_tr[t]).predict_proba(X_tr.iloc[v]), multi_class='ovr') 
                  for t, v in cv.split(X_tr, y_tr)]
        return np.mean(scores)

    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(obj_cat, n_trials=10)
    p_cat = study_cat.best_params
    p_cat = {
        'iterations': p_cat['n_estimators'],
        'depth': p_cat['max_depth'],
        'learning_rate': p_cat['learning_rate'],
        'subsample': p_cat['subsample'],
        'bootstrap_type': 'Bernoulli',
        'loss_function': 'MultiClass',
        'random_seed': 42,
        'verbose': 0
    }

    return p_xgb, p_lgb, p_cat


def plot_results(res_df, class_names, y_te, ens_probs, ensemble_m, xgb_m, X_te):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel A: Model Metrics
    res_df.plot(kind='bar', ax=axes[0, 0], colormap='tab10', width=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('A. Classification Performance Metrics', fontsize=12, fontweight='bold', loc='left')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0.6, 1.0)
    axes[0, 0].set_xticklabels(res_df.index, rotation=0)
    axes[0, 0].legend(frameon=True, fontsize=9)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

    # Panel B: Multi-class ROC Curves
    for i, c in enumerate(class_names):
        y_bin = (y_te == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, ens_probs[:, i])
        axes[0, 1].plot(fpr, tpr, lw=1.5, label=f'{c} (AUC = {auc(fpr, tpr):.2f})')

    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=0.8)
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.02])
    axes[0, 1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0, 1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0, 1].set_title('B. Multi-Class ROC Curves (Soft Ensemble)', fontsize=12, fontweight='bold', loc='left')
    axes[0, 1].legend(loc="lower right", fontsize=8, ncol=2)
    axes[0, 1].grid(linestyle='--', alpha=0.5)

    # Panel C: Confusion Matrix
    cm = confusion_matrix(y_te, ensemble_m.predict(X_te))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title('C. Ensemble Confusion Matrix', fontsize=12, fontweight='bold', loc='left')
    axes[1, 0].set_xlabel('Predicted Phenotype')
    axes[1, 0].set_ylabel('True Phenotype')

    # Panel D: Top 15 Discriminative Features
    imp = xgb_m.feature_importances_
    top15 = np.argsort(imp)[-15:]
    top_names = [X_te.columns[i].split('_')[-1] for i in top15]
    axes[1, 1].barh(range(15), imp[top15], color='#2b5c8f', edgecolor='black', linewidth=0.5)
    axes[1, 1].set_yticks(range(15))
    axes[1, 1].set_yticklabels(top_names, fontsize=9)
    axes[1, 1].set_title('D. Top 15 Discriminative Genera (XGBoost)', fontsize=12, fontweight='bold', loc='left')
    axes[1, 1].set_xlabel('Gini Importance Score')
    axes[1, 1].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    X, y, class_names = load_data('Dataset.xlsx')
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    p_xgb, p_lgb, p_cat = tune_hyperparameters(X_tr, y_tr)

    m_xgb = xgb.XGBClassifier(**p_xgb)
    m_lgb = lgb.LGBMClassifier(**p_lgb)
    m_cat = CatBoostClassifier(**p_cat)

    m_ens = VotingClassifier(
        estimators=[('xgb', m_xgb), ('lgb', m_lgb), ('cat', m_cat)],
        voting='soft'
    )

    models = {
        'LightGBM': m_lgb,
        'CatBoost': m_cat,
        'XGBoost': m_xgb,
        'Ensemble': m_ens
    }

    res = {}
    probs = {}

    print("\n[+] Training final models and evaluating performance...")
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        pb = model.predict_proba(X_te)
        
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average='weighted')
        auc_val = roc_auc_score(y_te, pb, multi_class='ovr', average='weighted')
        
        res[name] = {'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': auc_val}
        probs[name] = pb

    res_df = pd.DataFrame(res).T
    print("\nSummary of Test Set Performance:")
    print(res_df.to_string())

    plot_results(res_df, class_names, y_te, probs['Ensemble'], m_ens, m_xgb, X_te)


if __name__ == '__main__':
    main()