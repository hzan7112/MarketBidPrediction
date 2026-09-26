10 Final full-population feature-only bid prediction pipeline
===============================================================

Frozen final route
------------------
Inputs:
  83 feature-only variables
  = strategy profile + strategy transition + market + unit + calendar/slot

Forbidden model inputs:
  previous complete bid curve
  historical curve latent
  DeltaCurve / DeltaZ history
  theta
  template routing / template inertia
  current target curve fields

Target:
  8D absolute bid-curve latent representation

Primary model:
  Random Forest

Audit baselines:
  train mean
  Ridge
  Spline-GAM

Reference only:
  previous raw curve Persistence
  (extra historical bid information, not eligible as a same-input competitor)

Population:
  all curve-valid rows carried by the existing latent_forecasting_dataset.
  No Macro-B, template, or prediction-family filtering is applied.

Files
-----
10a_build_final_feature_only_dataset.py
  Freeze the full feature-only dataset.
  Enforces exactly 83 model inputs.
  Preserves previous raw curve only as reference_* evaluation columns.

10b_build_final_absolute_curve_latent.py
  TRAIN-only StandardScaler + PCA.
  Final latent dimension is FIXED at 8.
  No dimension selection is performed.

10c_train_final_curve_regressor.py
  Train final RF plus audit baselines.
  RF is the frozen primary model; TEST is not used for model selection.

10d_evaluate_final_bid_prediction.py
  Final acceptance metrics:
    coverage
    MAE / RMSE / MAPE / sMAPE / WAPE
    curve MAE P50/P90/P95
    21-knot price MAE / breakpoint proxy
    5 segment midpoint price MAEs
    q_anchor / q_span / q_max MAE and WAPE
    normalized shape MAE
    template consistency when Stage-2 template library is found
    RF by FLAT/SHAPE, true template, participant
    8D representation oracle
    Persistence reference

Run order
---------
python scripts\bidprediction\10a_build_final_feature_only_dataset.py --year 2025 --overwrite

python scripts\bidprediction\10b_build_final_absolute_curve_latent.py --year 2025 --overwrite

python scripts\bidprediction\10c_train_final_curve_regressor.py --year 2025 --overwrite

python scripts\bidprediction\10d_evaluate_final_bid_prediction.py --year 2025 --overwrite

Or run:
powershell -ExecutionPolicy Bypass -File scripts\bidprediction\10_run_final_pipeline.ps1

Important
---------
"Full population" here means the complete curve-valid population in the
already frozen latent_forecasting_dataset, with no Macro-B / family filtering.
10a prints exact source and retained row counts for train/val/test so the
final modeling universe is auditable.

The breakpoint_proxy_price_mae in 10d is the direct MAE at the 21 normalized
curve-representation knots. The final latent representation does not impose
artificial five-breakpoint theta parameters.
