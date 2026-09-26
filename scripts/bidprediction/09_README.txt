09 Prediction-Family route (after completed 09a)

Run in order:

1.
python scripts\bidprediction\09b_train_prediction_family_classifier.py --year 2025 --overwrite

2.
python scripts\bidprediction\09c_train_oracle_family_regressors.py --year 2025 --overwrite

3.
python scripts\bidprediction\09d_evaluate_full_family_routing_pipeline.py --year 2025 --overwrite

4.
python scripts\bidprediction\09e_summarize_prediction_family_experiment.py --year 2025 --overwrite

Definitions
-----------
09b:
83 feature-only inputs -> K=5 Prediction Family.
Models: Logistic Regression / Decision Tree / Random Forest.
Validation Macro-F1 selects classifier.

09c:
TRUE current family -> family-specific absolute-latent expert.
Candidates per family: family mean / Ridge / Spline-GAM / Random Forest.
Validation curve WAPE independently selects one expert for each family.
This is an oracle-routing upper-bound experiment.

09d:
Selected 09b classifier -> predicted family -> selected 09c expert -> curve.
Reports:
- global Random Forest reference
- oracle-family experts
- predicted-family experts
- correct-routing / wrong-routing WAPE
- true-family vs predicted-family routing-pair errors

09e:
Final error decomposition:
- family classification accuracy/Macro-F1
- oracle expert gain vs global RF
- routing penalty
- final gain/loss vs global RF

No deep learning is used.
No previous complete bid curve is used as a model input.
