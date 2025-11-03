# SABR Pipeline v2 (Fixed) — Upgrade Notes

- Drop-in replacement for `sabr_pipeline.py` with **kernel-weighted, regularized** local calibration.
- Includes `FTSE_SABR_LOCAL_v2_demo.ipynb` demo.

**Install**
1. Back up your current `sabr_pipeline.py`.
2. Replace with the one in this package.
3. Restart the notebook kernel and re-import:
   ```python
   import importlib, sabr_pipeline
   importlib.reload(sabr_pipeline)
   from sabr_pipeline import plot_parameter_sensitivity, calibrate_local_sabr_surface, plot_local_parameter_surface
   ```
