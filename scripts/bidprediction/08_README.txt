08 Macro-B feature-only absolute curve prediction
===============================================

Goal
----
Predict the current Macro-B bid curve WITHOUT direct previous bid-curve input.
Allowed model information:
- strategy profile / strategy-transition features
- market features
- unit features
- calendar / slot features

Direct historical bid curve, historical latent, DeltaZ history, theta/template inertia
are forbidden as model inputs.

Run order
---------
python scripts\bidprediction\08a_build_macro_b_absolute_curve_representation.py --year 2025 --overwrite
python scripts\bidprediction\08b_build_macro_b_feature_only_dataset.py --year 2025 --overwrite
python scripts\bidprediction\08c_train_macro_b_feature_only_regressors.py --year 2025 --overwrite
python scripts\bidprediction\08d_evaluate_macro_b_feature_only_forecasts.py --year 2025 --overwrite

Outputs
-------
08a -> data\processed\bidprediction\2025\macro_b_absolute_curve_representation\
08b -> data\processed\bidprediction\2025\macro_b_feature_only_dataset\
08c -> data\processed\bidprediction\2025\macro_b_feature_only_regression_models\
08d -> data\processed\bidprediction\2025\macro_b_feature_only_evaluation\

Primary comparison in 08d
-------------------------
train_mean vs Ridge vs Spline-GAM vs Random Forest

Persistence is only a cross-task reference because it uses the previous complete bid
curve, which the feature-only task intentionally does not provide to the model.
