
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from pathlib import Path
from sabr_utils import hagan_implied_vol
from sabr_data import load_csv, save_csv
from sabr_yahoo import fetch_observed_smile

# ----------------------------------------------
# Index mappings
# ----------------------------------------------
INDEX_SYMBOLS = {
    'FTSE100': 'EWU',
    'S&P500': '^SPX',
    'EUROSTOXX50': '^STOXX50E'
}

CSV_PATHS = {
    'FTSE100': 'MarketData/market_smile_ftse.csv',
    'S&P500': 'MarketData/market_smile_SnP500.csv',
    'EUROSTOXX50': 'MarketData/market_smile_EurStx50.csv'
}

# ----------------------------------------------
# Core SABR computations
# ----------------------------------------------
def black_vega(F, K, T, sigma):
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    nprime = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
    return np.sqrt(T) * nprime

def compute_model_vols(F, T, alpha, beta, rho, nu, strikes):
    return np.array([hagan_implied_vol(F, float(K), T, alpha, beta, rho, nu) for K in strikes])

# ----------------------------------------------
# Plotting utilities
# ----------------------------------------------
def _ax_dark(ax):
    ax.set_facecolor("#242424")
    ax.tick_params(colors="#e0e0e0")
    ax.grid(True, color="#3c3c3c", linestyle="--", linewidth=0.5, alpha=0.5)

def plot_smile(strikes, market, model, title=""):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e1e1e")
    _ax_dark(ax)
    ax.scatter(strikes, market, label="Market Vol", color="#4fa3ff", s=25)
    ax.plot(strikes, model, label="SABR Fit", color="#ffb703", linewidth=2)
    ax.set_xlabel("Strike", color="#e0e0e0", fontsize=10)
    ax.set_ylabel("Implied Volatility", color="#e0e0e0", fontsize=10)
    ax.set_title(title, color="#f0f0f0", fontsize=11, pad=10)
    leg = ax.legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")
    for t in leg.get_texts(): t.set_color("#e0e0e0")
    fig.tight_layout(); plt.show(); plt.close(fig)

def plot_smile_moneyness(F, strikes, market, model, title):
    mny = np.array(strikes, float) / float(F)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e1e1e")
    _ax_dark(ax)
    ax.scatter(mny, market, s=18, label="Market", color="#4fa3ff")
    ax.plot(mny, model, label="SABR Fit", color="#ffb703", linewidth=2)
    ax.set_xlabel("Moneyness (K/F)", color="#e0e0e0")
    ax.set_ylabel("Implied Volatility", color="#e0e0e0")
    ax.set_title(title, color="#f0f0f0")
    ax.legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")
    plt.tight_layout(); plt.show(); plt.close(fig)

# ----------------------------------------------
# Parameter Sensitivity Plots
# ----------------------------------------------
def plot_parameter_sensitivity(F, T, strikes, calib, beta, base_vols):
    alpha, rho, nu = calib["alpha"], calib["rho"], calib["nu"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#1e1e1e")
    for ax in axes: _ax_dark(ax)
    fig.suptitle("SABR Parameter Sensitivity (±10%)", color="#f0f0f0", fontsize=12)

    for delta in [0.9, 1.0, 1.1]:
        vols = compute_model_vols(F, T, alpha * delta, beta, rho, nu, strikes)
        axes[0].plot(strikes, vols, label=f"α × {delta:.1f}")
    axes[0].scatter(strikes, base_vols, s=12, color="#4fa3ff", label="Market")
    axes[0].set_title("α Sensitivity", color="#f0f0f0")
    axes[0].legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")

    for delta in [-0.1, 0.0, 0.1]:
        vols = compute_model_vols(F, T, alpha, beta, np.clip(rho + delta, -0.999, 0.999), nu, strikes)
        axes[1].plot(strikes, vols, label=f"ρ + {delta:+.1f}")
    axes[1].scatter(strikes, base_vols, s=12, color="#4fa3ff", label="Market")
    axes[1].set_title("ρ Sensitivity", color="#f0f0f0")
    axes[1].legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")

    for delta in [0.8, 1.0, 1.2]:
        vols = compute_model_vols(F, T, alpha, beta, rho, nu * delta, strikes)
        axes[2].plot(strikes, vols, label=f"ν × {delta:.1f}")
    axes[2].scatter(strikes, base_vols, s=12, color="#4fa3ff", label="Market")
    axes[2].set_title("ν Sensitivity", color="#f0f0f0")
    axes[2].legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")

    plt.tight_layout(); plt.show(); plt.close(fig)

# ----------------------------------------------
# Kernel-weighted Local SABR Calibration
# ----------------------------------------------
def _gaussian_kernel(x, x0, h):
    z = (x - x0) / (h + 1e-12)
    return np.exp(-0.5 * z * z)

def calibrate_local_sabr_surface(
    F, T, strikes, market, beta,
    global_prior=None,
    h=0.08,
    bounds=((1e-4, -0.999, 1e-4), (2.0, 0.4, 5.0)),
    lam_prior=(1e-3, 1e-3, 1e-3),
    min_points=6,
    step=1
):
    K = np.asarray(strikes, float)
    y = np.asarray(market, float)
    mny = K / float(F)
    xm = np.log(mny + 1e-16)

    if global_prior is None:
        global_prior = calibrate_sabr(F, T, K, y, beta=beta)
    a0, r0, v0 = global_prior['alpha'], global_prior['rho'], global_prior['nu']
    la, lr, lv = lam_prior

    recs = []
    for i in range(0, len(K), step):
        x0 = xm[i]
        vegas = np.array([black_vega(F, k, T, vol) for k, vol in zip(K, y)])
        w_kernel = _gaussian_kernel(xm, x0, h)
        w = vegas * w_kernel
        keep = w > (0.02 * w.max())
        if keep.sum() < min_points:
            continue
        K_sub, y_sub, w_sub = K[keep], y[keep], w[keep]

        def resid(p):
            a, r, v = p
            m = compute_model_vols(F, T, a, beta, r, v, K_sub)
            res_data = np.clip(m - y_sub, -0.5, 0.5) * (w_sub / (np.mean(w_sub) + 1e-12))
            res_prior = np.array([la*(a - a0), lr*(r - r0), lv*(v - v0)], dtype=float)
            return np.hstack([res_data, res_prior])

        try:
            res = least_squares(resid, x0=np.array([a0, r0, v0]), bounds=bounds,
                                method='trf', loss='soft_l1', f_scale=0.05)
            a, r, v = res.x
            n_eff = float((w_sub / w_sub.max()).sum())
            sse = float(np.sum(res.fun[:-3]**2))
            recs.append({'K': K[i], 'K/F': mny[i], 'alpha': a, 'rho': r, 'nu': v, 'SSE': sse, 'n_eff': n_eff})
        except Exception:
            continue

    return pd.DataFrame.from_records(recs)

def plot_local_parameter_surface(df_params, smooth=True, sg_window=9, sg_poly=3):
    if df_params is None or df_params.empty:
        print("No local parameter data to plot."); return
    df = df_params.sort_values('K/F').reset_index(drop=True)
    x, a, r, v = df['K/F'].values, df['alpha'].values, df['rho'].values, df['nu'].values

    def smooth_vec(z):
        if not smooth or len(z) < sg_window: return None
        wl = min(len(z) - (1 - len(z) % 2), sg_window)
        if wl < 5: return None
        try: return savgol_filter(z, window_length=wl, polyorder=min(sg_poly, max(1, wl//2)))
        except Exception: return None

    a_s, r_s, v_s = smooth_vec(a), smooth_vec(r), smooth_vec(v)
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), facecolor="#1e1e1e")
    for ax in axes: _ax_dark(ax)
    fig.suptitle("Local SABR Parameters vs Moneyness (K/F)", color="#f0f0f0", fontsize=12)

    axes[0].plot(x, a, color="#ffb703", alpha=0.5, linewidth=1, label="raw")
    if a_s is not None: axes[0].plot(x, a_s, color="#ffb703", linewidth=2, label="smoothed")
    axes[0].set_ylabel("α (level)", color="#e0e0e0"); axes[0].legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")

    axes[1].plot(x, r, color="#4fa3ff", alpha=0.5, linewidth=1, label="raw")
    if r_s is not None: axes[1].plot(x, r_s, color="#4fa3ff", linewidth=2, label="smoothed")
    axes[1].set_ylabel("ρ (skew)", color="#e0e0e0"); axes[1].legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")

    axes[2].plot(x, v, color="#90ee90", alpha=0.5, linewidth=1, label="raw")
    if v_s is not None: axes[2].plot(x, v_s, color="#90ee90", linewidth=2, label="smoothed")
    axes[2].set_ylabel("ν (vol-of-vol)", color="#e0e0e0")
    axes[2].set_xlabel("Moneyness (K/F)", color="#e0e0e0")
    axes[2].legend(facecolor="#1e1e1e", edgecolor="#1e1e1e")

    plt.tight_layout(); plt.show(); plt.close(fig)

# ----------------------------------------------
# Global SABR calibration functions
# ----------------------------------------------
def model_smile_grid(F, T, alpha, beta, rho, nu, kf_min=0.7, kf_max=1.3, n=121):
    mny = np.linspace(kf_min, kf_max, n)
    K = F * mny
    vols = compute_model_vols(F, T, alpha, beta, rho, nu, K)
    return mny, K, vols

def calibrate_sabr_logmoneyness(F, T, strikes, market_vols, beta,
                                init=(0.25, -0.2, 0.6),
                                bounds=((1e-6, -0.999, 1e-6), (5.0, 0.999, 5.0))):
    K = np.asarray(strikes, float)
    y = np.asarray(market_vols, float)
    vegas = np.array([black_vega(F, k, T, s) for k, s in zip(K, y)])
    w_vega = vegas / (np.mean(vegas[vegas > 0]) if np.any(vegas > 0) else 1.0)
    x = np.log(K / float(F))
    sigma_x = 0.25
    w_x = np.exp(-0.5 * (x / sigma_x) ** 2)
    w = w_vega * w_x

    def resid(p):
        a, r, v = p
        m = compute_model_vols(F, T, a, beta, r, v, K)
        return w * np.clip(m - y, -0.5, 0.5)

    res = least_squares(resid, x0=np.array(init), bounds=bounds,
                        method='trf', loss='soft_l1', f_scale=0.05)
    a, r, v = res.x
    return {'alpha': float(a), 'beta': float(beta), 'rho': float(r), 'nu': float(v),
            'success': bool(res.success), 'cost': float(0.5 * np.sum((res.fun / (w + 1e-12)) ** 2))}

def calibrate_sabr(F, T, strikes, market_vols, beta,
                   init=(0.2, -0.1, 0.5),
                   bounds=((1e-6, -0.999, 1e-6),(5.0, 0.999, 5.0))):
    K = np.asarray(strikes, float)
    y = np.asarray(market_vols, float)
    vegas = np.array([black_vega(F, k, T, s) for k, s in zip(K, y)])
    w = vegas / (np.mean(vegas[vegas > 0]) if np.any(vegas > 0) else 1.0)

    def resid(xp):
        a, r, v = xp
        m = compute_model_vols(F, T, a, beta, r, v, K)
        return w * np.clip(m - y, -0.5, 0.5)

    res = least_squares(resid, x0=np.array(init), bounds=bounds, method='trf',
                        loss='soft_l1', f_scale=0.05)
    a, r, v = res.x
    return {'alpha': float(a), 'beta': float(beta), 'rho': float(r), 'nu': float(v),
            'success': bool(res.success),
            'cost': float(0.5 * np.sum((res.fun / (w + 1e-12)) ** 2))}

# ----------------------------------------------
# Data utilities
# ----------------------------------------------
def fetch_and_save(index_name: str, target_days: int = 90):
    sym = INDEX_SYMBOLS[index_name]
    df, meta = fetch_observed_smile(sym, target_days=target_days)
    canonical = Path(CSV_PATHS[index_name])
    canonical.parent.mkdir(parents=True, exist_ok=True)
    dated = canonical.parent / f"{df['AsOf'].iloc[0].date()}_{canonical.name}"
    df.to_csv(canonical, index=False)
    df.to_csv(dated, index=False)
    return df, meta, str(canonical), str(dated)

def load_csv_smile(index_name: str):
    path = CSV_PATHS[index_name]
    df = load_csv(path)
    if not set(['AsOf','Expiry','Strike','MarketVol']).issubset(df.columns):
        raise ValueError('CSV must have AsOf, Expiry, Strike, MarketVol.')
    asof = pd.to_datetime(df['AsOf'].iloc[0])
    expiry = pd.to_datetime(df['Expiry'].iloc[0])
    T = (expiry - asof).days / 365.0
    if 'Forward' in df.columns and pd.notna(df['Forward'].iloc[0]):
        F = float(df['Forward'].iloc[0])
    elif {'Spot','Rate','DividendYield'}.issubset(df.columns):
        S = float(df['Spot'].iloc[0]); r = float(df['Rate'].iloc[0]); q = float(df['DividendYield'].iloc[0])
        F = S * np.exp((r-q)*T)
    else:
        F = float(df['Strike'].median())
    strikes = df['Strike'].astype(float).values
    market  = df['MarketVol'].astype(float).values
    meta = {'AsOf': str(asof.date()), 'Expiry': str(expiry.date()), 'T': float(T), 'Forward': float(F)}
    return df[['AsOf','Expiry','Strike','MarketVol']], meta
