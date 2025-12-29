"""Train per-dataset models, compute importances, SHAP and produce ensemble meta-model.

This script trains a RandomForest and a baseline LogisticRegression per available dataset
found in `src.data_loader` (uci / kaggle). It computes permutation importance, global
SHAP importance and saves artifacts under `models/`.

Usage: PYTHONPATH=. python -m src.train_ensemble
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import shap

from src.data_loader import load_uci, load_kaggle, generate_synthetic
from src.preprocess import build_preprocessor, canonicalize_columns

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
MODEL_DIR = MODEL_DIR.resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FEATURES = ['age','systolic_blood_pressure','diastolic_blood_pressure','blood_sugar',
                     'hemoglobin','heart_rate','bmi','parity','anc_visits','location']


def prepare(df):
    df = canonicalize_columns(df)
    # map risk to label if present
    if 'risk_label' not in df.columns:
        if 'risk' in df.columns:
            df['risk_label'] = df['risk'].map({'low':0,'mid':1,'medium':1,'high':2,'high risk':2}).fillna(0).astype(int)
        else:
            df = generate_synthetic(n=len(df))
    # ensure expected features exist
    for f in EXPECTED_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    X = df[EXPECTED_FEATURES].copy()
    y = df['risk_label'].astype(int)
    return X, y


def train_on_df(name, df):
    print('Training on dataset:', name, 'rows:', len(df))
    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)

    preprocessor, num_feats, cat_feats = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X_train_t, y_train)

    # Baseline logistic (interpretable)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    lr.fit(X_res, y_res)

    # Random forest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED)
    rf.fit(X_res, y_res)

    # Evaluate on test
    models = {'logistic': lr, 'random_forest': rf}
    results = {}
    for k,m in models.items():
        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(X_test_t)
            # pick prob for class 2
            classes = list(m.classes_) if hasattr(m, 'classes_') else None
            idx = classes.index(2) if classes and 2 in classes else -1
            prob_high = proba[:, idx] if idx >= 0 else proba[:, -1]
        else:
            prob_high = None
        y_pred = m.predict(X_test_t)
        res = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'recall_high': float(recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0) if 2 in set(y_test) else 0.0),
            'precision_high': float(precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0) if 2 in set(y_test) else 0.0),
            'f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        }
        try:
            res['auroc'] = float(roc_auc_score((y_test==2).astype(int), prob_high)) if prob_high is not None else None
        except Exception:
            res['auroc'] = None
        results[k] = res

    # Permutation importance on rf
    perm = permutation_importance(rf, X_test_t, y_test, n_repeats=10, random_state=SEED, n_jobs=1)
    feat_names = num_feats + (preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_feats).tolist() if len(cat_feats) > 0 else [])
    perm_df = pd.DataFrame({'feature': feat_names, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std})
    perm_df = perm_df.sort_values('importance_mean', ascending=False)

    # SHAP global importance for rf (TreeExplainer)
    try:
        explainer = shap.TreeExplainer(rf)
        # compute on a subset of train for speed
        bg = X_train_t[: min(200, X_train_t.shape[0])]
        shap_vals = explainer.shap_values(bg)
        # shap_vals shape varies; compute mean absolute per feature for class 2 if present
        arr = np.asarray(shap_vals)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) or (1,n_features,n_classes)
            # try to reduce to (n_samples, n_features)
            if arr.shape[2] > 1:
                cls_idx = 2 if 2 in list(rf.classes_) else -1
                vals = arr[:,:,cls_idx] if arr.shape[0] > 1 else arr[0,:,:][:,cls_idx]
            else:
                vals = arr.reshape(arr.shape[0], arr.shape[1])
        elif arr.ndim == 2:
            vals = arr
        else:
            vals = arr
        shap_mean_abs = np.mean(np.abs(vals), axis=0)
        shap_df = pd.DataFrame({'feature': feat_names, 'shap_mean_abs': shap_mean_abs}).sort_values('shap_mean_abs', ascending=False)
    except Exception as e:
        print('SHAP global failed:', e)
        shap_df = pd.DataFrame({'feature': feat_names, 'shap_mean_abs': np.zeros(len(feat_names))})

    # save schema for this dataset (canonicalized column list)
    schema = {'columns': list(df.columns)}
    with open(MODEL_DIR / f'schema_{name}.json', 'w') as f:
        json.dump(schema, f)

    # save artifacts
    joblib.dump(preprocessor, MODEL_DIR / f'preprocessor_{name}.joblib')
    joblib.dump(lr, MODEL_DIR / f'logistic_{name}.joblib')
    joblib.dump(rf, MODEL_DIR / f'rf_{name}.joblib')
    perm_df.to_csv(MODEL_DIR / f'perm_importance_{name}.csv', index=False)
    shap_df.to_csv(MODEL_DIR / f'shap_global_{name}.csv', index=False)
    with open(MODEL_DIR / f'metrics_{name}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved artifacts for', name)
    # return test set and base-model high-risk probabilities for stacking
    # compute rf predict_proba on X_test_t
    if hasattr(rf, 'predict_proba'):
        proba = rf.predict_proba(X_test_t)
        classes = list(rf.classes_)
        idx = classes.index(2) if 2 in classes else -1
        prob_high = proba[:, idx] if idx >= 0 else proba[:, -1]
    else:
        prob_high = rf.predict(X_test_t)

    return {
        'name': name,
        'preprocessor': str(MODEL_DIR / f'preprocessor_{name}.joblib'),
        'logistic': str(MODEL_DIR / f'logistic_{name}.joblib'),
        'rf': str(MODEL_DIR / f'rf_{name}.joblib'),
        'perm_csv': str(MODEL_DIR / f'perm_importance_{name}.csv'),
        'shap_csv': str(MODEL_DIR / f'shap_global_{name}.csv'),
        'metrics_json': str(MODEL_DIR / f'metrics_{name}.json')
        ,
        'X_test_t_shape': X_test_t.shape,
        'y_test': y_test.tolist(),
        'prob_high': prob_high.tolist()
    }


def main():
    datasets = []
    u = load_uci()
    if u is not None:
        datasets.append(('uci', u))
    k = load_kaggle()
    if k is not None:
        datasets.append(('kaggle', k))
    # fallback: if none, use synthetic
    if not datasets:
        print('No real datasets found, generating synthetic dataset')
        datasets.append(('synthetic', generate_synthetic(n=1200)))

    summary = []
    # collect per-dataset test probs for stacking
    meta_X_parts = []
    meta_y_parts = []
    names = []
    for name, df in datasets:
        info = train_on_df(name, df)
        summary.append(info)
        names.append(info['name'])
        # load prob_high and y_test
        ph = np.asarray(info['prob_high'])
        yt = np.asarray(info['y_test'])
        meta_X_parts.append(ph.reshape(-1,1))
        meta_y_parts.append((yt==2).astype(int))

    # build meta dataset (concatenate rows across datasets)
    if len(meta_X_parts) > 0:
        meta_X = np.vstack(meta_X_parts)
        meta_y = np.concatenate(meta_y_parts)
        # if multiple base models (one per dataset), meta_X columns correspond to single base model each
        # train a simple logistic regression meta-classifier for high-risk detection
        from sklearn.linear_model import LogisticRegression
        meta_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
        try:
            meta_clf.fit(meta_X, meta_y)
            joblib.dump(meta_clf, MODEL_DIR / 'meta_model.joblib')
            print('Saved meta_model.joblib')
        except Exception as e:
            print('Meta training failed:', e)

    with open(MODEL_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Training complete. Summary written to models/training_summary.json')


if __name__ == '__main__':
    main()
