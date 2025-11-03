# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import importlib
import sabr_pipeline
importlib.reload(sabr_pipeline)

from sabr_pipeline import (
    load_csv_smile, calibrate_sabr, compute_model_vols,
    plot_smile, plot_parameter_sensitivity
)

st.set_page_config(page_title="SABR Vol Dashboard", layout="wide")
st.title("📈 SABR Volatility Modelling Dashboard")

# --- Sidebar controls ---
index_name = st.sidebar.selectbox("Index", ["FTSE100", "S&P500", "EUROSTOXX50"], index=0)
beta = st.sidebar.slider("β (elasticity)", 0.0, 1.0, 0.8, 0.05)
show_sensitivity = st.sidebar.checkbox("Show parameter sensitivity", True)

# --- Load CSV market smile ---
df, meta = load_csv_smile(index_name)
F, T = meta["Forward"], meta["T"]
strikes = df["Strike"].astype(float).values
market = df["MarketVol"].astype(float).values

# --- Calibrate + model vols ---
calib = calibrate_sabr(F, T, strikes, market, beta=beta)
model = compute_model_vols(F, T, calib["alpha"], beta, calib["rho"], calib["nu"], strikes)

st.markdown(
    f"**AsOf:** {meta['AsOf']} &nbsp;&nbsp; **Expiry:** {meta['Expiry']} &nbsp;&nbsp; "
    f"**T:** {T:.4f}y &nbsp;&nbsp; **F:** {F:.6f} &nbsp;&nbsp; **β:** {beta:.2f}"
)
st.markdown(
    f"**Calibrated:** α={calib['alpha']:.4f}, ρ={calib['rho']:.4f}, ν={calib['nu']:.4f} · "
    f"SSE={calib['cost']:.6g}"
)

# --- Plots ---
st.pyplot(plot_smile(strikes, market, model, title=f"{index_name}: Market vs SABR Fit (β={beta:.2f})"))

if show_sensitivity:
    st.pyplot(plot_parameter_sensitivity(F, T, strikes, calib, beta, market))

# --- Table preview + save CSV into Results/ ---
vol_table = pd.DataFrame({
    "Strike": strikes,
    "Moneyness (K/F)": strikes / F,
    "Market Vol (%)": market * 100,
    "SABR Implied Vol (%)": model * 100,
    "Vol Diff (bps)": (model - market) * 10000
})

Path("Results").mkdir(exist_ok=True)
save_path = Path("Results") / f"{index_name}_SABR_vol_table.csv"
vol_table.to_csv(save_path, index=False)

st.subheader("📄 Volatility Comparison (preview)")
st.dataframe(vol_table.head(10).style.format({
    "Market Vol (%)": "{:.2f}",
    "SABR Implied Vol (%)": "{:.2f}",
    "Vol Diff (bps)": "{:+.1f}"
}), use_container_width=True)

st.download_button(
    "💾 Download full table as CSV",
    data=vol_table.to_csv(index=False).encode("utf-8"),
    file_name=f"{index_name}_SABR_vol_table.csv",
    mime="text/csv"
)

