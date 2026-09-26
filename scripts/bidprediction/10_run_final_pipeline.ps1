$ErrorActionPreference = "Stop"

python scripts\bidprediction\10a_build_final_feature_only_dataset.py --year 2025 --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\bidprediction\10b_build_final_absolute_curve_latent.py --year 2025 --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\bidprediction\10c_train_final_curve_regressor.py --year 2025 --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\bidprediction\10d_evaluate_final_bid_prediction.py --year 2025 --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Stage 10 final pipeline completed."
Write-Host "See:"
Write-Host "data\processed\bidprediction\2025\final_bid_prediction_evaluation\summary.txt"
