# 🧮 SABR Volatility Smile Calibration Dashboard

Analyze, calibrate, and visualize **market-implied volatility smiles**  
for major equity indices (**FTSE 100**, **S&P 500**, **EURO STOXX 50**).  
Use **historical CSV data** or **live Yahoo Finance** feeds to compare  
**observed** versus **model-fitted** volatility curves.

---

## 📖 Overview

This project implements a full **SABR model calibration and volatility-smile analysis** framework.  
It estimates the SABR parameters α (initial vol), ρ (correlation), ν (vol-of-vol) for a fixed β,  
and plots market vs model implied volatilities across strikes or moneyness (K/F).

**Features**
- Historical calibration from **CSV market data**  
- Live calibration via **Yahoo Finance**

---

## 📂 Project Structure

| File / Folder | Purpose |
|:--|:--|
| `FTSE_SABR_Multi.ipynb` | Interactive notebook dashboard |
| `sabr_pipeline.py` | Core SABR workflow: load → calibrate → plot |
| `sabr_utils.py` | Mathematical utilities (Hagan implied vol etc.) |
| `sabr_data.py` | CSV I/O and time-to-maturity helpers |
| `sabr_yahoo.py` | Live option-smile fetcher (Yahoo Finance) |
| `MarketData/` | Historical / fetched option-smile data |
| `requirements.txt` | Python dependencies |

---

## 🧠 The SABR Model (Hagan et al., 2002)

The **SABR (Stochastic Alpha Beta Rho)** model describes the forward price and its stochastic volatility:

$$
dF_t = \sigma_t F_t^{\beta} dW_t
$$

$$
d\sigma_t = \nu\,\sigma_t\,dZ_t, \qquad dW_t\,dZ_t = \rho\,dt
$$

**Parameters**

- α = σ₀   — initial volatility level  
- β ∈ [0, 1]  — elasticity of volatility w.r.t Fₜ  
- ρ             — correlation between price & volatility  
- ν             — volatility of volatility  

---

## 🧮 Hagan’s Implied-Volatility Approximation

For strike K, forward F, and maturity T, the Black–Scholes implied volatility under SABR is:

$$
\sigma_{BS}(F,K) =
\frac{\alpha}{(F K)^{(1-\beta)/2}}
\frac{z}{x(z)}
\Bigg[
1 +
\Big(
\frac{(1-\beta)^2}{24}\frac{\alpha^2}{(F K)^{1-\beta}}
+ \frac{\rho\,\beta\,\nu\,\alpha}{4 (F K)^{(1-\beta)/2}}
+ \frac{(2-3\rho^2)\nu^2}{24}
\Big)T
\Bigg].
$$

where

$$
z = \frac{\nu}{\alpha}(F K)^{(1-\beta)/2}\ln\!\Big(\frac{F}{K}\Big),
\qquad
x(z) = \ln\!\Bigg(\frac{\sqrt{1-2\rho z + z^2}+z-\rho}{1-\rho}\Bigg).
$$

**ATM limit (F = K):**

$$
\sigma_{BS}(F,F) =
\frac{\alpha}{F^{1-\beta}}
\Bigg[
1 +
\Big(
\frac{(1-\beta)^2}{24}\frac{\alpha^2}{F^{2(1-\beta)}}
+ \frac{\rho\,\beta\,\nu\,\alpha}{4 F^{1-\beta}}
+ \frac{(2-3\rho^2)\nu^2}{24}
\Big)T
\Bigg].
$$

---

## 📊 Outputs

Each calibration produces:

- Fitted parameters (α, ρ, ν)  
- Objective function (SSE)  
- Plots:  
    • Market vs SABR Fit (by Strike)  
    • Market vs SABR Fit (by Moneyness K/F)  

---

## ⚙️ Environment Setup

```bash
# create and activate env
python -m venv sabr_model
sabr_model\Scripts\activate        # Windows
# or
source sabr_model/bin/activate     # macOS / Linux

# install dependencies
pip install -r requirements.txt
