Task 1 v2 — Annual Static Strategy Profile
===========================================

Why v2
------
The first annual validation exposed two definition issues:
1) annual_curve_bend_ratio collapsed around zero under annual median aggregation;
2) annual_shape_variability was missing for many structurally all-flat participants.

Changes
-------
- annual_curve_bend_ratio = mean(daily_curve_bend_ratio)
  instead of median(...). This is a signed annual asymmetry tendency.
- If annual_flat_curve_rate == 1 and there are no valid non-flat shape days,
  annual_shape_variability = 0 and
  annual_shape_variability_flat_fallback_flag = 1.
- Validation adds annual_feature_dispersion.csv and central-IQR-collapse flags.

Unchanged
---------
- No ST / Break in the frozen annual profile.
- No target-year recent bid history.
- Other annual structural/level features retain their previous aggregation.
- High feature correlation is reported but does not auto-delete a dimension.

Run
---
python scripts/bidprofile/02_build_annual_strategy_profile.py --year 2025
python scripts/bidprofile/03_validate_annual_strategy_profile.py --year 2025

No need to rerun 01.
