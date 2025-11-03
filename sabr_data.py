import pandas as pd
from pathlib import Path

def load_csv(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(path, parse_dates=['AsOf', 'Expiry'], dayfirst=True)

def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def time_to_maturity(asof, expiry):
    return (pd.to_datetime(expiry) - pd.to_datetime(asof)).days / 365.0
