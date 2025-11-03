import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
def _mid(a, b, last=None):
    try:
        if a is not None and b is not None:
            return 0.5*(float(a)+float(b))
    except Exception:
        pass
    return last if last is not None else np.nan
def estimate_forward_from_parity(calls_df, puts_df):
    c = calls_df[['strike','bid','ask','lastPrice']].copy()
    p = puts_df[['strike','bid','ask','lastPrice']].copy()
    c.columns = ['strike','cbid','cask','clast']
    p.columns = ['strike','pbid','pask','plast']
    m = pd.merge(c, p, on='strike', how='inner')
    if m.empty:
        return np.nan, np.nan
    m['Cmid'] = m.apply(lambda r: _mid(r['cbid'], r['cask'], r['clast']), axis=1)
    m['Pmid'] = m.apply(lambda r: _mid(r['pbid'], r['pask'], r['plast']), axis=1)
    m = m.dropna(subset=['Cmid','Pmid','strike'])
    if len(m) < 5:
        return np.nan, np.nan
    y = (m['Cmid'] - m['Pmid']).values
    X = np.vstack([np.ones_like(m['strike'].values), m['strike'].values]).T
    intercept, slope = np.linalg.lstsq(X, y, rcond=None)[0]
    D = -slope
    if D <= 0 or D > 1.5:
        return np.nan, np.nan
    F = intercept / D
    return float(F), float(D)
def fetch_observed_smile(yahoo_symbol: str, target_days: int = 90):
    tk = yf.Ticker(yahoo_symbol)
    expiries = tk.options
    if not expiries:
        raise RuntimeError(f"No options for {yahoo_symbol}")
    asof = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).normalize()
    exps = pd.to_datetime(expiries)
    target = asof + pd.Timedelta(days=int(target_days))
    diffs_days = (exps - target).map(lambda x: abs(x.days))
    expiry = exps[diffs_days.argmin()]
    chain = tk.option_chain(expiry.strftime('%Y-%m-%d'))
    calls = chain.calls.copy(); puts = chain.puts.copy()
    F, D = estimate_forward_from_parity(calls, puts)
    iv_calls = calls[['strike','impliedVolatility']].rename(columns={'impliedVolatility':'iv_call'})
    iv_puts  = puts[['strike','impliedVolatility']].rename(columns={'impliedVolatility':'iv_put'})
    iv = pd.merge(iv_calls, iv_puts, on='strike', how='outer')
    iv['MarketVol'] = iv[['iv_call','iv_put']].median(axis=1)
    df = iv[['strike','MarketVol']].dropna().sort_values('strike')
    df['AsOf'] = asof; df['Expiry'] = expiry.tz_localize(None)
    df = df.rename(columns={'strike':'Strike'})[['AsOf','Expiry','Strike','MarketVol']]
    meta = {'Forward': F, 'Discount': D}
    return df, meta
