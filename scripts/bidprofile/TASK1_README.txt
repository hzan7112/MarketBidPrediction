Task 1 — Annual Static Strategy Profile
========================================

Goal
----
2025 full-year bid history -> one frozen annual static profile per participant.

Files
-----
annual_profile_core.py
02_build_annual_strategy_profile.py
03_validate_annual_strategy_profile.py

Placement
---------
Copy all three files to:
scripts/bidprofile/

Input
-----
Reuse the existing daily core output:
data/processed/final_clean/daily/2025/daily_strategy_core_2025.csv

You do NOT need to rerun 01 if this file already exists.

Run
---
python scripts/bidprofile/02_build_annual_strategy_profile.py --year 2025
python scripts/bidprofile/03_validate_annual_strategy_profile.py --year 2025

Main profile output
-------------------
data/processed/bidprofile_annual/2025/annual_strategy_profile_2025.csv

Validation output
-----------------
results/bidprofile_annual_validation/2025/

Method notes
------------
1. The frozen profile contains only 9 annual static strategy dimensions.
2. ST / Break are not frozen profile inputs.
3. annual_flat_curve_rate uses mean(daily_flat_curve_rate), because it is a rate.
4. Persistence retains the lag-1 correlation interpretation, but constant
   consecutive behavior is treated as persistence=1 instead of missing.
5. The validator recomputes H1/H2 profiles using identical definitions, so all
   9 annual dimensions are checked, including persistence and shape variability.
6. The same-year 2025 full-year profile may only be used for retrospective
   explanatory-power validation. It is not a causal 2025 forecast input.
