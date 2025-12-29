from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import json
import glob

from src.preprocess import canonicalize_columns

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
MODEL_DIR = MODEL_DIR.resolve()


def load_preprocessor():
    return joblib.load(MODEL_DIR / 'preprocessor.joblib')


def load_model(model_name: str = 'final_model.joblib'):
    return joblib.load(MODEL_DIR / model_name)


EXPECTED_FEATURES = ['age','systolic_blood_pressure','diastolic_blood_pressure','blood_sugar',
                     'hemoglobin','heart_rate','bmi','parity','anc_visits','location']


def proba_to_category(prob_high, thresholds=(0.4, 0.7)):
    if prob_high >= thresholds[1]:
        return 'High'
    elif prob_high >= thresholds[0]:
        return 'Medium'
    else:
        return 'Low'


def load_all_models(models_dir: str = None):
    """Load per-dataset preprocessors and models plus optional meta model.
    Returns (models_map, meta_model)
    models_map: {name: {'pre': Preprocessor or None, 'rf': model or None, 'log': model or None, 'schema': [...]} }
    """
    MODEL_DIR_LOCAL = Path(models_dir) if models_dir else MODEL_DIR
    models = {}
    for pre_path in glob.glob(str(MODEL_DIR_LOCAL / 'preprocessor_*.joblib')):
        name = Path(pre_path).name.replace('preprocessor_','').replace('.joblib','')
        try:
            pre = joblib.load(pre_path)
        except Exception:
            pre = None
        rf = None
        log = None
        try:
            rf = joblib.load(MODEL_DIR_LOCAL / f'rf_{name}.joblib')
        except Exception:
            pass
        try:
            log = joblib.load(MODEL_DIR_LOCAL / f'logistic_{name}.joblib')
        except Exception:
            pass
        schema = []
        try:
            with open(MODEL_DIR_LOCAL / f'schema_{name}.json') as f:
                schema = json.load(f).get('columns', [])
        except Exception:
            schema = []
        models[name] = {'pre': pre, 'rf': rf, 'log': log, 'schema': schema}

    meta = None
    try:
        meta = joblib.load(MODEL_DIR_LOCAL / 'meta_model.joblib')
    except Exception:
        meta = None
    return models, meta


def predict_single(patient_dict: dict, model=None, preprocessor=None):
    # If explicit model provided, use it; otherwise attempt schema selection and meta-ensemble
    models_map = None
    meta_model = None
    if preprocessor is None or model is None:
        models_map, meta_model = load_all_models()

    if preprocessor is None and model is None:
        # schema-based selection: pick model with largest column overlap
        df_tmp = pd.DataFrame([patient_dict])
        df_tmp = df_tmp.rename(columns={c: c.strip().lower().replace(' ', '_') for c in df_tmp.columns})
        df_tmp = canonicalize_columns(df_tmp)
        best_name = None
        best_score = -1
        for name, info in models_map.items():
            sch = [c.strip().lower().replace(' ', '_') for c in info.get('schema') or []]
            if not sch:
                continue
            overlap = len(set(df_tmp.columns).intersection(set(sch)))
            score = overlap / max(1, len(sch))
            if score > best_score:
                best_score = score
                best_name = name
        if best_name and best_score > 0:
            chosen = models_map[best_name]
            model = chosen.get('rf') or chosen.get('log')
            preprocessor = chosen.get('pre')
        else:
            # fallback to single-model functions
            try:
                preprocessor = load_preprocessor()
                model = load_model()
            except Exception:
                raise RuntimeError('No model available for prediction')

    df = pd.DataFrame([patient_dict])
    df = df.rename(columns={c: c.strip().lower().replace(' ', '_') for c in df.columns})
    df = canonicalize_columns(df)
    for f in EXPECTED_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df = df[EXPECTED_FEATURES]

    X_t = preprocessor.transform(df)

    # base model prediction
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_t)[0]
        if hasattr(model, 'classes_'):
            classes = list(model.classes_)
            if 2 in classes:
                idx = classes.index(2)
            else:
                idx = -1
        else:
            idx = -1
        prob_high = proba[idx] if idx >= 0 else proba[-1]
    else:
        prob_high = float(model.predict(X_t)[0])

    # If meta model exists, compute base probs across all loaded models and run meta
    if meta_model is not None and models_map is not None:
        meta_feats = []
        for name, info in models_map.items():
            pre_i = info.get('pre')
            m_i = info.get('rf') or info.get('log')
            if pre_i is None or m_i is None:
                continue
            dfi = df.copy()
            try:
                Xi = pre_i.transform(dfi)
                if hasattr(m_i, 'predict_proba'):
                    p = m_i.predict_proba(Xi)[0]
                    classes = list(m_i.classes_) if hasattr(m_i, 'classes_') else None
                    idx = classes.index(2) if classes and 2 in classes else -1
                    ph = p[idx] if idx >= 0 else p[-1]
                else:
                    ph = float(m_i.predict(Xi)[0])
            except Exception:
                ph = 0.0
            meta_feats.append(ph)
        try:
            meta_arr = np.asarray(meta_feats).reshape(1, -1)
            meta_pred_proba = meta_model.predict_proba(meta_arr)[0]
            prob_high_meta = meta_pred_proba[1]
            score = int(prob_high_meta * 100)
            cat = proba_to_category(prob_high_meta)
            return {'score': score, 'category': cat, 'meta_proba_high': float(prob_high_meta), 'base_proba_high': float(prob_high)}
        except Exception:
            pass

    score = int(prob_high * 100)
    cat = proba_to_category(prob_high)
    return {'score': score, 'category': cat, 'proba': proba.tolist() if 'proba' in locals() else None}


def predict_batch(df: pd.DataFrame, model=None, preprocessor=None):
    if preprocessor is None:
        preprocessor = load_preprocessor()
    if model is None:
        model = load_model()
    # Accept common alternate column names, canonicalize, and add missing expected features
    df = df.rename(columns={c: c.strip().lower().replace(' ', '_') for c in df.columns})
    df = canonicalize_columns(df)

    alt_map = {
        'age': ['age'],
        'systolic_blood_pressure': ['systolicbp','systolic_bp','systolic_blood_pressure'],
        'diastolic_blood_pressure': ['diastolicbp','diastolic_bp','diastolic_blood_pressure'],
        'blood_sugar': ['bs','blood_sugar','bloodsugar'],
        'hemoglobin': ['hemoglobin','hb'],
        'heart_rate': ['heartrate','heart_rate'],
        'bmi': ['bmi','body_mass_index','weight'],
        'parity': ['parity','previous_pregnancies'],
        'anc_visits': ['anc_visits','ancvisits','anc_visits'],
        'location': ['location','place']
    }
    cols_lower = {c.lower(): c for c in df.columns}
    for target, alts in alt_map.items():
        for a in alts:
            if a in cols_lower and target not in df.columns:
                df[target] = df[cols_lower[a]]
                break

    for f in EXPECTED_FEATURES:
        if f not in df.columns:
            df[f] = np.nan

    df = df[EXPECTED_FEATURES]
    X_t = preprocessor.transform(df)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_t)
        classes = list(model.classes_)
        idx = classes.index(2) if 2 in classes else -1
        prob_high = proba[:, idx]
        scores = (prob_high * 100).astype(int)
        categories = [proba_to_category(p) for p in prob_high]
        res_df = df.copy()
        res_df['risk_score'] = scores
        res_df['risk_category'] = categories
        return res_df
    else:
        preds = model.predict(X_t)
        res_df = df.copy()
        res_df['risk_category'] = preds
        return res_df
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.preprocess import canonicalize_columns

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / '..' / 'models'
MODEL_DIR = MODEL_DIR.resolve()

def load_preprocessor():
    return joblib.load(MODEL_DIR / 'preprocessor.joblib')

def load_model(model_name: str = 'final_model.joblib'):
    return joblib.load(MODEL_DIR / model_name)

EXPECTED_FEATURES = ['age','systolic_blood_pressure','diastolic_blood_pressure','blood_sugar',
                     'hemoglobin','heart_rate','bmi','parity','anc_visits','location']


def proba_to_category(prob_high, thresholds=(0.4, 0.7)):
    if prob_high >= thresholds[1]:
        return 'High'
    elif prob_high >= thresholds[0]:
        return 'Medium'
    else:
        return 'Low'

def predict_single(patient_dict: dict, model=None, preprocessor=None):
    if preprocessor is None:
        preprocessor = load_preprocessor()
    if model is None:
        model = load_model()

    df = pd.DataFrame([patient_dict])
    # canonicalize uploaded keys and ensure all expected features present
    df = df.rename(columns={c: c.strip().lower().replace(' ', '_') for c in df.columns})
    df = canonicalize_columns(df)
    for f in EXPECTED_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df = df[EXPECTED_FEATURES]

    X_t = preprocessor.transform(df)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_t)[0]
        if hasattr(model, 'classes_'):
            classes = list(model.classes_)
            if 2 in classes:
                idx = classes.index(2)
            else:
                idx = -1
        else:
            idx = -1
        prob_high = proba[idx]
        score = int(prob_high * 100)
        cat = proba_to_category(prob_high)
        return {'score': score, 'category': cat, 'proba': proba.tolist()}
    else:
        pred = model.predict(X_t)[0]
        return {'score': int(pred), 'category': str(pred), 'proba': None}

def predict_batch(df: pd.DataFrame, model=None, preprocessor=None):
    if preprocessor is None:
        preprocessor = load_preprocessor()
    if model is None:
        model = load_model()
    # Accept common alternate column names, canonicalize, and add missing expected features
    df = df.rename(columns={c: c.strip().lower().replace(' ', '_') for c in df.columns})
    df = canonicalize_columns(df)

    # handle some typical alternate names
    alt_map = {
        'age': ['age'],
        'systolic_blood_pressure': ['systolicbp','systolic_bp','systolic_blood_pressure'],
        'diastolic_blood_pressure': ['diastolicbp','diastolic_bp','diastolic_blood_pressure'],
        'blood_sugar': ['bs','blood_sugar','bloodsugar'],
        'hemoglobin': ['hemoglobin','hb'],
        'heart_rate': ['heartrate','heart_rate'],
        'bmi': ['bmi','body_mass_index','weight'],
        'parity': ['parity','previous_pregnancies'],
        'anc_visits': ['anc_visits','ancvisits','anc_visits'],
        'location': ['location','place']
    }
    # map alternate names onto expected if present
    cols_lower = {c.lower(): c for c in df.columns}
    for target, alts in alt_map.items():
        for a in alts:
            if a in cols_lower and target not in df.columns:
                df[target] = df[cols_lower[a]]
                break

    for f in EXPECTED_FEATURES:
        if f not in df.columns:
            df[f] = np.nan

    df = df[EXPECTED_FEATURES]
    X_t = preprocessor.transform(df)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_t)
        classes = list(model.classes_)
        idx = classes.index(2) if 2 in classes else -1
        prob_high = proba[:, idx]
        scores = (prob_high * 100).astype(int)
        categories = [proba_to_category(p) for p in prob_high]
        res_df = df.copy()
        res_df['risk_score'] = scores
        res_df['risk_category'] = categories
        return res_df
    else:
        preds = model.predict(X_t)
        res_df = df.copy()
        res_df['risk_category'] = preds
        return res_df
