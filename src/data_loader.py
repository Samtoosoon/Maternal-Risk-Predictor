from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None

def load_uci():
    return load_csv_if_exists(RAW_DIR / "uci_maternal_health.csv")

def load_kaggle():
    return load_csv_if_exists(RAW_DIR / "kaggle_maternal_india.csv")

def load_any():
    df = load_uci()
    if df is not None:
        print("Loaded UCI dataset from:", RAW_DIR / "uci_maternal_health.csv")
        return df
    df = load_kaggle()
    if df is not None:
        print("Loaded Kaggle dataset from:", RAW_DIR / "kaggle_maternal_india.csv")
        return df
    return None

def generate_synthetic(n=1200, random_state=42):
    np.random.seed(random_state)
    age = np.random.normal(26, 5, n).clip(15,50)
    systolic = np.random.normal(110, 12, n).clip(80,200)
    diastolic = (systolic - np.random.normal(40,7,n)).clip(40,120)
    blood_sugar = np.random.normal(95,30,n).clip(60,400)
    hemoglobin = np.random.normal(11.5,2, n).clip(4,18)
    heart_rate = np.random.normal(78,10,n).clip(50,150)
    bmi = np.random.normal(23,4,n).clip(14,45)
    parity = np.random.choice([0,1,2,3,4], size=n, p=[0.45,0.3,0.15,0.07,0.03])
    anc_visits = np.random.poisson(3, n).clip(0,10)
    location = np.random.choice(['rural','urban'], size=n, p=[0.6,0.4])

    df = pd.DataFrame({
        'age': age,
        'systolic_blood_pressure': systolic,
        'diastolic_blood_pressure': diastolic,
        'blood_sugar': blood_sugar,
        'hemoglobin': hemoglobin,
        'heart_rate': heart_rate,
        'bmi': bmi,
        'parity': parity,
        'anc_visits': anc_visits,
        'location': location
    })

    def compute_risk_prob(r):
        s = 0.0
        if not pd.isna(r['hemoglobin']):
            s += max(0, (11 - r['hemoglobin']) / 6.0) * 1.5
        s += max(0, (r['systolic_blood_pressure'] - 120) / 80.0) * 1.2
        s += max(0, (r['blood_sugar'] - 100) / 300.0) * 1.0
        if r['age'] > 35:
            s += 0.8
        if not pd.isna(r['bmi']):
            if r['bmi'] < 18.5:
                s += 0.6
            elif r['bmi'] > 30:
                s += 0.8
        s += max(0, (3 - r['anc_visits'])) * 0.15
        s += max(0, (r['parity'] - 2)) * 0.1
        return 1 / (1 + np.exp(-s + 1.0))

    df['risk_prob'] = df.apply(compute_risk_prob, axis=1)
    df['risk_label'] = df['risk_prob'].apply(lambda p: 2 if p >= 0.7 else (1 if p >= 0.4 else 0))
    return df
