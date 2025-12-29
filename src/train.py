# Train script for maternal-risk models
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_any, generate_synthetic
from src.preprocess import build_preprocessor

SEED = 42

def get_dataset():
    df = load_any()
    if df is None:
        print('No CSV found under data/raw/. Using synthetic dataset for demo.')
        df = generate_synthetic(n=1200)
    df = df.rename(columns={c: c.strip().lower().replace(' ', '_') for c in df.columns})
    if 'risk_label' not in df.columns:
        if 'risk' in df.columns:
            df['risk_label'] = df['risk'].map({'low':0,'mid':1,'high':2}).fillna(0).astype(int)
        else:
            print('No target column found; regenerating synthetic dataset with labels.')
            df = generate_synthetic(n=len(df))
    return df

def main():
    ROOT = Path(__file__).resolve().parents[1]
    MODEL_DIR = ROOT / 'models'
    MODEL_DIR = MODEL_DIR.resolve()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = get_dataset()
    expected_features = ['age','systolic_blood_pressure','diastolic_blood_pressure','blood_sugar',
                         'hemoglobin','heart_rate','bmi','parity','anc_visits','location']
    features = [f for f in expected_features if f in df.columns]
    if len(features) < 5:
        raise RuntimeError('Not enough matching features found. Check dataset columns or use the synthetic option.')
    X = df[features].copy()
    y = df['risk_label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)

    preprocessor, numeric_feats, cat_feats = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X_train_t, y_train)
    print('Resampled train distribution:', dict(zip(*np.unique(y_res, return_counts=True))))

    # Initialize base models
    print('\n' + '='*50)
    print('TRAINING BASE MODELS')
    print('='*50)
    
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=SEED)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )
    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        verbosity=-1
    )

    # Train base models
    print('Training Logistic Regression...')
    lr.fit(X_res, y_res)
    print('Training Random Forest...')
    rf.fit(X_res, y_res)
    print('Training XGBoost...')
    xgb.fit(X_res, y_res)
    print('Training LightGBM...')
    lgbm.fit(X_res, y_res)

    # Create and train ensemble model (Voting Classifier)
    print('\n' + '='*50)
    print('TRAINING ENSEMBLE MODEL')
    print('='*50)
    
    # Soft voting uses predicted probabilities for better accuracy
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=SEED)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=SEED)),
            ('xgb', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                  random_state=SEED, use_label_encoder=False, 
                                  eval_metric='mlogloss', verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    random_state=SEED, verbosity=-1))
        ],
        voting='soft'  # Use probability-based voting for better results
    )
    
    print('Training Ensemble (Soft Voting: LR + RF + XGBoost + LightGBM)...')
    ensemble.fit(X_res, y_res)

    # All models including ensemble
    models = {
        'logistic_regression': lr, 
        'random_forest': rf, 
        'xgboost': xgb, 
        'lightgbm': lgbm,
        'ensemble': ensemble
    }
    
    # Track accuracies to find best model
    accuracies = {}
    
    print('\n' + '='*50)
    print('MODEL EVALUATION RESULTS')
    print('='*50)
    
    for name, m in models.items():
        y_pred = m.predict(X_test_t)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        print(f"\n{name.upper()} accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=3))

        cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_xticklabels(['Low','Medium','High']); ax.set_yticklabels(['Low','Medium','High'])
        ax.set_title(f'{name}')
        fig.savefig(MODEL_DIR / f'confusion_{name}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

    # Find best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    best_accuracy = accuracies[best_model_name]
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model_name.upper()}")
    print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # Save all models
    joblib.dump(preprocessor, MODEL_DIR / 'preprocessor.joblib')
    joblib.dump(lr, MODEL_DIR / 'logistic_model.joblib')
    joblib.dump(rf, MODEL_DIR / 'rf_model.joblib')
    joblib.dump(xgb, MODEL_DIR / 'xgboost_model.joblib')
    joblib.dump(lgbm, MODEL_DIR / 'lightgbm_model.joblib')
    joblib.dump(ensemble, MODEL_DIR / 'ensemble_model.joblib')
    
    # Save the best model as final_model
    joblib.dump(best_model, MODEL_DIR / 'final_model.joblib')
    
    # Save model comparison results
    comparison_df = pd.DataFrame([
        {'model': name, 'accuracy': acc, 'accuracy_pct': f"{acc*100:.2f}%"} 
        for name, acc in accuracies.items()
    ]).sort_values('accuracy', ascending=False)
    comparison_df.to_csv(MODEL_DIR / 'model_comparison.csv', index=False)
    
    print("\n" + "="*50)
    print("MODEL COMPARISON LEADERBOARD")
    print("="*50)
    for i, row in comparison_df.iterrows():
        marker = ">>>" if row['model'] == best_model_name else "   "
        print(f"{marker} {row['model']:20s} {row['accuracy_pct']:>8s}")

    try:
        num_names = numeric_feats
        cat_onehot = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_feats).tolist() if len(cat_feats) > 0 else []
        feat_names = num_names + cat_onehot
        processed = pd.DataFrame(X_res, columns=feat_names)
        processed['risk_label'] = y_res
        processed.to_csv(MODEL_DIR / 'processed_train_sample.csv', index=False)
    except Exception:
        pass

    print(f"\nSaved preprocessor and all models to {MODEL_DIR}")
    print(f"Final model ({best_model_name}) saved as final_model.joblib")


if __name__ == '__main__':
    main()
