# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from pathlib import Path
import importlib
import sabr_pipeline

# --- reload & patch SABR kernel exactly as in your notebook ---
importlib.reload(sabr_pipeline)
sabr_pipeline._THRESHOLD = 0.005

def _gaussian_kernel(x, x0, h):
    z = (x - x0) / (h + 1e-12)
    return np.exp(-0.5 * z * z)

sabr_pipeline._gaussian_kernel = _gaussian_kernel

from sabr_pipeline import (
    load_csv_smile,
    calibrate_sabr,
    calibrate_sabr_logmoneyness,
    compute_model_vols,
    calibrate_local_sabr_surface,
)

# ---------- small helper for "has data?" without errors ----------
def _has_local_data(local_params):
    if local_params is None:
        return False
    # DataFrame
    if isinstance(local_params, pd.DataFrame):
        return not local_params.empty
    # list / array / tuple etc.
    try:
        return len(local_params) > 0
    except TypeError:
        # some custom object without len() – assume it has data
        return True


# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="SABR Volatility Dashboard", layout="wide")
st.markdown(
    "<h3 style='text-align:center;margin-bottom:0.2rem;'>📈 SABR Volatility Modelling Dashboard</h3>",
    unsafe_allow_html=True,
)

# Sidebar controls
mode = st.sidebar.radio("Mode", ["Global", "Local"], index=0)
index_name = st.sidebar.selectbox("Index", ["FTSE100", "S&P500", "EUROSTOXX50"], index=0)
beta = st.sidebar.slider("β (elasticity)", 0.0, 1.0, 0.8, 0.05)

if mode == "Local":
    h = st.sidebar.slider("Kernel width (h)", 0.05, 0.25, 0.12, 0.01)
    smooth = st.sidebar.select_slider(
        "Smoothing", options=[5, 9, 15], value=5,
        format_func=lambda v: {5: "Light", 9: "Moderate", 15: "Strong"}[v]
    )
else:
    show_sensitivity = st.sidebar.checkbox("Show parameter sensitivity", True)

Path("Results").mkdir(exist_ok=True)

# ---------------- LOAD MARKET DATA ----------------
df, meta = load_csv_smile(index_name)
F, T = meta["Forward"], meta["T"]
strikes = df["Strike"].astype(float).values
market = df["MarketVol"].astype(float).values

st.markdown(
    f"<p style='text-align:center;font-size:12px;margin-top:0.1rem;'>"
    f"<b>AsOf:</b> {meta['AsOf']} <b>Expiry:</b> {meta['Expiry']} "
    f"<b>T:</b> {T:.4f}y <b>F:</b> {F:.6f} <b>β:</b> {beta:.2f}</p>",
    unsafe_allow_html=True,
)

# ======================================================
# GLOBAL SABR CALIBRATION (strike-space)
# ======================================================
if mode == "Global":
    calib = calibrate_sabr(F, T, strikes, market, beta)
    model = compute_model_vols(F, T, calib["alpha"], beta, calib["rho"], calib["nu"], strikes)

    st.markdown(
        f"<p style='text-align:center;font-size:12px;'>"
        f"<b>Calibrated:</b> α={calib['alpha']:.4f}, "
        f"ρ={calib['rho']:.4f}, ν={calib['nu']:.4f}, SSE={calib['cost']:.6g}</p>",
        unsafe_allow_html=True,
    )

    # ---- Row 1: Market vs SABR | Smile vs Moneyness ----
    c1, c2 = st.columns(2)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(3.8, 2.8))
        ax1.scatter(strikes, market, s=12, label="Market", alpha=0.8)
        ax1.plot(strikes, model, color="orange", lw=1.4, label="SABR Fit")
        ax1.set_title("Market vs SABR", fontsize=10)
        ax1.set_xlabel("Strike", fontsize=8)
        ax1.set_ylabel("Implied Vol", fontsize=8)
        ax1.tick_params(labelsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with c2:
        mny = strikes / F
        fig2, ax2 = plt.subplots(figsize=(3.8, 2.8))
        ax2.scatter(mny, market, s=12, label="Market", alpha=0.8)
        ax2.plot(mny, model, color="orange", lw=1.4, label="SABR Fit")
        ax2.set_title("Smile vs Moneyness (K/F)", fontsize=10)
        ax2.set_xlabel("Moneyness K/F", fontsize=8)
        ax2.set_ylabel("Implied Vol", fontsize=8)
        ax2.tick_params(labelsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ---- Row 2: Sensitivities (α, ρ, ν) ----
    if show_sensitivity:
        st.markdown("### Parameter Sensitivity (±10%)")
        col_a, col_r, col_v = st.columns(3)

        # α sensitivity
        with col_a:
            fig_a, ax_a = plt.subplots(figsize=(3.0, 2.5))
            for fct, col in zip([0.9, 1.0, 1.1], ["#00bfff", "#ffaa00", "#00ff00"]):
                m = compute_model_vols(F, T, calib["alpha"] * fct, beta, calib["rho"], calib["nu"], strikes)
                ax_a.plot(strikes, m, color=col, lw=1, label=f"α×{fct:.1f}")
            ax_a.scatter(strikes, market, color="white", s=6, alpha=0.6, label="Market")
            ax_a.set_title("α Sensitivity", fontsize=9)
            ax_a.tick_params(labelsize=7)
            ax_a.grid(True, linestyle="--", alpha=0.3)
            ax_a.legend(fontsize=6)
            st.pyplot(fig_a, use_container_width=True)
            plt.close(fig_a)

        # ρ sensitivity
        with col_r:
            fig_r, ax_r = plt.subplots(figsize=(3.0, 2.5))
            for dlt, col in zip([-0.1, 0.0, 0.1], ["#00bfff", "#ffaa00", "#00ff00"]):
                m = compute_model_vols(F, T, calib["alpha"], beta, calib["rho"] + dlt, calib["nu"], strikes)
                ax_r.plot(strikes, m, color=col, lw=1, label=f"ρ{dlt:+.1f}")
            ax_r.scatter(strikes, market, color="white", s=6, alpha=0.6, label="Market")
            ax_r.set_title("ρ Sensitivity", fontsize=9)
            ax_r.tick_params(labelsize=7)
            ax_r.grid(True, linestyle="--", alpha=0.3)
            ax_r.legend(fontsize=6)
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)

        # ν sensitivity
        with col_v:
            fig_v, ax_v = plt.subplots(figsize=(3.0, 2.5))
            for fct, col in zip([0.8, 1.0, 1.2], ["#00bfff", "#ffaa00", "#00ff00"]):
                m = compute_model_vols(F, T, calib["alpha"], beta, calib["rho"], calib["nu"] * fct, strikes)
                ax_v.plot(strikes, m, color=col, lw=1, label=f"ν×{fct:.1f}")
            ax_v.scatter(strikes, market, color="white", s=6, alpha=0.6, label="Market")
            ax_v.set_title("ν Sensitivity", fontsize=9)
            ax_v.tick_params(labelsize=7)
            ax_v.grid(True, linestyle="--", alpha=0.3)
            ax_v.legend(fontsize=6)
            st.pyplot(fig_v, use_container_width=True)
            plt.close(fig_v)

    # ---- Table (compact) ----
    vol_table = pd.DataFrame({
        "Strike": strikes,
        "Moneyness (K/F)": strikes / F,
        "Market Vol (%)": market * 100,
        "SABR Model Vol (%)": model * 100,
        "Vol Diff (bps)": (model - market) * 10000,
    })
    st.markdown("#### Calibrated Volatility Table (preview)")
    st.dataframe(vol_table.head(10), use_container_width=True, height=220)

# ======================================================
# LOCAL SABR CALIBRATION (log-moneyness, adaptive h)
# ======================================================
else:
    st.markdown(f"### Local SABR Calibration — {index_name}")

    try:
        # --- Global calibration in log-moneyness (exact notebook logic) ---
        calib = calibrate_sabr_logmoneyness(F, T, strikes, market, beta)
        model = compute_model_vols(F, T, calib["alpha"], beta, calib["rho"], calib["nu"], strikes)

        st.markdown(
            f"<p style='font-size:12px;'><b>Global Parameters:</b> "
            f"α={calib['alpha']:.4f}, ρ={calib['rho']:.4f}, ν={calib['nu']:.4f}, "
            f"SSE={calib['cost']:.6g}</p>",
            unsafe_allow_html=True,
        )

        # ---- Row 1: Market vs SABR | Smile vs Moneyness ----
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(3.8, 2.8))
            ax1.scatter(strikes, market, s=12, label="Market", alpha=0.8)
            ax1.plot(strikes, model, color="orange", lw=1.4, label="SABR Fit")
            ax1.set_title("Market vs SABR (Global)", fontsize=10)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, linestyle="--", alpha=0.3)
            ax1.legend(fontsize=7)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with c2:
            mny = strikes / F
            fig2, ax2 = plt.subplots(figsize=(3.8, 2.8))
            ax2.scatter(mny, market, s=12, label="Market", alpha=0.8)
            ax2.plot(mny, model, color="orange", lw=1.4, label="SABR Fit")
            ax2.set_title("Smile vs Moneyness (K/F)", fontsize=10)
            ax2.tick_params(labelsize=8)
            ax2.grid(True, linestyle="--", alpha=0.3)
            ax2.legend(fontsize=7)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        # ---- Local surface calibration (adaptive h + retry) ----
        h_eff = max(0.08, h * (1.0 + 1.5 * (0.5 - beta) ** 2))

        local_params = calibrate_local_sabr_surface(
            F, T, strikes, market, beta,
            global_prior=calib,
            h=h_eff,
            lam_prior=(5e-4, 5e-4, 5e-4),
            min_points=4,
        )

        if not _has_local_data(local_params):
            st.warning(
                f"No local parameter data found (β={beta:.2f}, h={h_eff:.3f}). "
                "Retrying with broader kernel..."
            )
            local_params = calibrate_local_sabr_surface(
                F, T, strikes, market, beta,
                global_prior=calib,
                h=min(0.25, h_eff * 1.6),
                lam_prior=(5e-4, 5e-4, 5e-4),
                min_points=3,
            )

        if not _has_local_data(local_params):
            st.error(
                f"Still no local parameter data after retry (β={beta:.2f}). "
                "Try increasing h or check smile density."
            )
        else:
            df_params = pd.DataFrame(local_params)
            st.success(f"✅ Local SABR surface calibrated on {len(df_params)} points")

            # ---- Row 2: α, ρ, ν vs moneyness (3 small charts) ----
            st.markdown("#### Local SABR Parameters vs Moneyness (K/F)")
            col_a, col_r, col_v = st.columns(3)
            for col, (param, color) in zip(
                [col_a, col_r, col_v],
                [("alpha", "orange"), ("rho", "dodgerblue"), ("nu", "limegreen")]
            ):
                if param not in df_params.columns:
                    continue
                fig, ax = plt.subplots(figsize=(3.0, 2.5))
                x = df_params["moneyness"]
                y = df_params[param]
                ax.plot(x, y, color=color, alpha=0.6, lw=1.3, label="raw")
                if len(x) > smooth:
                    from scipy.signal import savgol_filter
                    y_smooth = savgol_filter(y, smooth, 3)
                    ax.plot(x, y_smooth, color=color, lw=1.8, label="smoothed")
                ax.set_title(param, fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(fontsize=6)
                col.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # ---- preview table ----
            st.markdown("#### Local Parameter Table (preview)")
            st.dataframe(df_params.head(10), use_container_width=True, height=220)

    except Exception as e:
        # Any unexpected failure in local logic is caught here
        st.error(f"Local SABR calibration failed: {e}")


