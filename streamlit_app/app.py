import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt

from src.predict import predict_single, predict_batch, load_model, load_preprocessor

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models'

st.set_page_config(page_title='Maternal Risk Predictor', layout='wide')
st.title('Maternal Risk Predictor — Prototype with SHAP explanations')

st.markdown('''
Input patient parameters in the sidebar for a single prediction, or upload a CSV for batch prediction.
Make sure you have trained models (run `PYTHONPATH=. python -m src.train`) so `models/preprocessor.joblib` and `models/final_model.joblib` exist.
''')

st.sidebar.header('Single patient input (for demo)')
age = st.sidebar.number_input('Age', min_value=12, max_value=60, value=25)
systolic = st.sidebar.number_input('Systolic BP', min_value=60, max_value=220, value=110)
diastolic = st.sidebar.number_input('Diastolic BP', min_value=40, max_value=140, value=70)
hemoglobin = st.sidebar.number_input('Hemoglobin (g/dL)', min_value=3.0, max_value=20.0, value=11.0, step=0.1)
blood_sugar = st.sidebar.number_input('Blood sugar (mg/dL)', min_value=40, max_value=500, value=90)
heart_rate = st.sidebar.number_input('Heart rate (bpm)', min_value=40, max_value=200, value=80)
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=60.0, value=23.0, step=0.1)
parity = st.sidebar.number_input('Parity (previous pregnancies)', min_value=0, max_value=10, value=0)
anc_visits = st.sidebar.number_input('ANC visits', min_value=0, max_value=20, value=3)
location = st.sidebar.selectbox('Location', ['rural', 'urban'])

single_input = {
    'age': age,
    'systolic_blood_pressure': systolic,
    'diastolic_blood_pressure': diastolic,
    'hemoglobin': hemoglobin,
    'blood_sugar': blood_sugar,
    'heart_rate': heart_rate,
    'bmi': bmi,
    'parity': parity,
    'anc_visits': anc_visits,
    'location': location
}

col1, col2 = st.columns([1,2])

with col1:
    st.subheader('Single prediction')
    if st.button('Predict (single)'):
        try:
            model = load_model()
            pre = load_preprocessor()
        except Exception as e:
            st.error('Model or preprocessor not found. Run training script first (`PYTHONPATH=. python -m src.train`).')
            st.stop()
        out = predict_single(single_input, model=model, preprocessor=pre)
        st.metric('Risk score', f"{out['score']}/100")
        st.write('**Category:**', out['category'])
        # show top contributors summary if SHAP values were computed
        try:
            # run the same SHAP workflow used below to compute local shap vector
            df0 = preprocessor.transform(pd.DataFrame([single_input]))
            X_row = df0
            background = None
            sample_path = MODEL_DIR / 'processed_train_sample.csv'
            if sample_path.exists():
                bg = pd.read_csv(sample_path).drop(columns=['risk_label'], errors='ignore')
                background = bg.sample(min(50, len(bg)), random_state=42).values
            else:
                background = np.zeros((10, X_row.shape[1]))

            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model, data=background)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_row)
            # normalize shap vector to 1D for chosen class
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # try to pick class 2
                classes = list(model.classes_) if hasattr(model, 'classes_') else None
                idx = classes.index(2) if classes and 2 in classes else -1
                if idx >= 0 and idx < arr.shape[2]:
                    sv = arr[0,:,idx]
                else:
                    sv = arr[0,:, -1]
            elif arr.ndim == 2:
                sv = arr.reshape(-1)
            else:
                sv = arr.reshape(-1)

            # feature names
            try:
                feat_names = preprocessor.get_feature_names_out()
                feat_names = list(feat_names)
            except Exception:
                try:
                    num_names = preprocessor.transformers_[0][2]
                    cat_src = preprocessor.transformers_[1][2]
                    cat_onehot = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_src).tolist()
                    feat_names = list(num_names) + list(cat_onehot)
                except Exception:
                    feat_names = [f'feat_{i}' for i in range(len(sv))]

            sv = np.asarray(sv).reshape(-1)
            top_idx = np.argsort(np.abs(sv))[-3:][::-1]
            st.write('**Top contributors:**')
            for i in top_idx:
                st.write(f"- {feat_names[i]}: {sv[i]:+.3f}")
        except Exception:
            pass
        if out.get('proba') is not None:
            st.write('Probability vector (classes order):', out['proba'])

        # SHAP explanation for single input
        try:
            preprocessor = pre
            model_local = model
            background = None
            sample_path = MODEL_DIR / 'processed_train_sample.csv'
            if sample_path.exists():
                bg = pd.read_csv(sample_path).drop(columns=['risk_label'], errors='ignore')
                background = bg.sample(min(50, len(bg)), random_state=42).values
            else:
                df0 = pd.DataFrame([single_input])
                X0 = preprocessor.transform(df0)
                background = np.zeros((10, X0.shape[1]))

            X_row = preprocessor.transform(pd.DataFrame([single_input]))

            # ensure numpy arrays
            if not isinstance(background, (list, tuple)):
                background = np.asarray(background)
            X_row = np.asarray(X_row)

            if hasattr(model_local, 'feature_importances_'):
                explainer = shap.TreeExplainer(model_local, data=background)
            else:
                # KernelExplainer expects a function that accepts 2D-array
                pred_fn = lambda x: model_local.predict_proba(x)
                explainer = shap.KernelExplainer(pred_fn, background)
            shap_values = explainer.shap_values(X_row)

            if isinstance(shap_values, list) and len(shap_values) >= 1:
                if hasattr(model_local, 'classes_') and 2 in list(model_local.classes_):
                    class_idx = list(model_local.classes_).index(2)
                else:
                    class_idx = -1
                sv = shap_values[class_idx] if class_idx >= 0 and class_idx < len(shap_values) else shap_values[-1]
            else:
                sv = shap_values

            try:
                # attempt to recover feature names from preprocessor
                num_names = preprocessor.transformers_[0][2]
                cat_names_src = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
                cat_onehot = preprocessor.named_transformers_.get('cat')
                if cat_onehot is not None:
                    try:
                        cat_onehot = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_names_src).tolist()
                    except Exception:
                        cat_onehot = []
                else:
                    cat_onehot = []
                feat_names = list(num_names) + list(cat_onehot)
            except Exception:
                feat_names = [f'feat_{i}' for i in range(sv.shape[1])]

            shap_abs = np.abs(sv).flatten()
            top_idx = np.argsort(shap_abs)[-6:][::-1]
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x=shap_abs[top_idx], y=[feat_names[i] for i in top_idx], ax=ax)
            ax.set_xlabel('Absolute SHAP value')
            ax.set_title('Top contributing features (SHAP)')
            st.pyplot(fig)
        except Exception as e:
            st.write('SHAP explanation unavailable:', e)

with col2:
    st.subheader('Batch CSV prediction')
    uploaded = st.file_uploader('Upload CSV for batch predictions (columns must match features)', type=['csv'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write('Uploaded file preview')
            st.dataframe(df.head())
            if st.button('Run batch prediction'):
                model = load_model()
                pre = load_preprocessor()
                res = predict_batch(df, model=model, preprocessor=pre)
                st.write('Results preview:')
                st.dataframe(res.head())
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f'Error processing uploaded file: {e}')

st.markdown('---')
st.write('Note: This is a demonstration prototype. For production use, validate and retrain on real clinical data, add security, logging, and offline-mode support for low-connectivity environments.')
