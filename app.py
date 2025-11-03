
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import sabr_pipeline

importlib.reload(sabr_pipeline)
from sabr_pipeline import (
    load_csv_smile, calibrate_sabr, compute_model_vols,
    calibrate_local_sabr_surface, plot_local_parameter_surface,
    plot_smile, plot_parameter_sensitivity
)

# ------------------- Streamlit Setup -------------------
st.set_page_config(page_title="SABR Volatility Dashboard", layout="wide")
st.title("📈 SABR Volatility Modelling Dashboard")

# ------------------- Sidebar Controls -------------------
mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Global SABR Calibration", "Local SABR Surface Analysis"],
    index=0
)

index_name = st.sidebar.selectbox("Index", ["FTSE100", "S&P500", "EUROSTOXX50"], index=0)
beta = st.sidebar.slider("β (elasticity)", 0.0, 1.0, 0.8, 0.05)

if mode == "Global SABR Calibration":
    show_sensitivity = st.sidebar.checkbox("Show parameter sensitivity", True)

elif mode == "Local SABR Surface Analysis":
    h = st.sidebar.slider("Kernel width (h)", 0.05, 0.25, 0.12, 0.01)
    smooth = st.sidebar.select_slider(
        "Smoothing Strength",
        options=[("Light (retain shape)", 5), ("Moderate", 9), ("Strong", 15)],
        value=("Light (retain shape)", 5),
        format_func=lambda x: x[0]
    )

# ------------------- Ensure Results folder -------------------
Path("Results").mkdir(exist_ok=True)

# ------------------- Load Market Data -------------------
df, meta = load_csv_smile(index_name)
F, T = meta["Forward"], meta["T"]
strikes = df["Strike"].astype(float).values
market = df["MarketVol"].astype(float).values

# ------------------- Metadata -------------------
st.markdown(
    f"**AsOf:** {meta['AsOf']} &nbsp;&nbsp; **Expiry:** {meta['Expiry']} &nbsp;&nbsp; "
    f"**T:** {T:.4f}y &nbsp;&nbsp; **F:** {F:.6f} &nbsp;&nbsp; **β:** {beta:.2f}"
)

# ======================================================
# GLOBAL SABR CALIBRATION MODE
# ======================================================
if mode == "Global SABR Calibration":
    calib = calibrate_sabr(F, T, strikes, market, beta=beta)
    model = compute_model_vols(F, T, calib["alpha"], beta, calib["rho"], calib["nu"], strikes)

    st.markdown(
        f"**Calibrated Parameters:** α={calib['alpha']:.4f}, ρ={calib['rho']:.4f}, "
        f"ν={calib['nu']:.4f} · SSE={calib['cost']:.6g}"
    )

    # --- Plot: Market vs Model ---
    fig1 = plt.figure()
    plot_smile(strikes, market, model, title=f"{index_name}: Market vs SABR Fit (β={beta:.2f})")
    st.pyplot(fig1)
    plt.close(fig1)

    # --- Plot: Moneyness Smile ---
    mny = strikes / float(F)
    fig2, ax2 = plt.subplots()
    ax2.scatter(mny, market, s=18, label="Market")
    ax2.plot(mny, model, label="SABR Fit", color="orange")
    ax2.set_xlabel("Moneyness (K/F)")
    ax2.set_ylabel("Implied Volatility")
    ax2.set_title(f"{index_name}: Smile vs Moneyness (β={beta:.2f})")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig2)
    plt.close(fig2)

    # --- Plot: Sensitivity ---
    if show_sensitivity:
        fig3 = plt.figure()
        plot_parameter_sensitivity(F, T, strikes, calib, beta, market)
        st.pyplot(fig3)
        plt.close(fig3)

    # --- Volatility Comparison Table ---
    vol_table = pd.DataFrame({
        "Strike": strikes,
        "Moneyness (K/F)": strikes / F,
        "Market Vol (%)": market * 100,
        "SABR Implied Vol (%)": model * 100,
        "Vol Diff (bps)": (model - market) * 10000
    })

    save_path = Path("Results") / f"{index_name}_GLOBAL_SABR_vol_table.csv"
    vol_table.to_csv(save_path, index=False)

    st.subheader("📄 Volatility Comparison (preview)")
    st.dataframe(
        vol_table.head(10).style.format({
            "Market Vol (%)": "{:.2f}",
            "SABR Implied Vol (%)": "{:.2f}",
            "Vol Diff (bps)": "{:+.1f}"
        }),
        use_container_width=True
    )

    st.download_button(
        "💾 Download full table as CSV",
        data=vol_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{index_name}_GLOBAL_SABR_vol_table.csv",
        mime="text/csv"
    )

# ======================================================
# LOCAL SABR SURFACE ANALYSIS MODE
# ======================================================
else:
    st.markdown(f"### Local SABR Parameter Surface for **{index_name}** (β={beta:.2f})")

    calib_global = calibrate_sabr(F, T, strikes, market, beta=beta)
    local_params = calibrate_local_sabr_surface(
        F, T, strikes, market, beta,
        global_prior=calib_global,
        h=h,
        lam_prior=(5e-4, 5e-4, 5e-4),
        min_points=5
    )

    if local_params is not None and len(local_params) > 0:
        plot_local_parameter_surface(local_params, smooth=True, sg_window=smooth[1], sg_poly=3)
    else:
        st.warning("⚠️ No local parameter data to plot — try adjusting kernel width (h).")


