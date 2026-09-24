"""Read-only audit of the existing v3 decoder and saved test predictions."""
from pathlib import Path
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'data/processed/bidprediction/2025'
FROZEN = BASE / 'frozen_stable_theta_continuous_dataset'
spec = importlib.util.spec_from_file_location(
    'decoder', ROOT / 'scripts/bidprediction/04c_calibrate_conditional_theta_pipeline_v3.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manifest = json.loads((FROZEN / 'manifest.json').read_text(encoding='utf-8'))
centers, _ = m.discover_centers(ROOT / 'data/processed/bidtemplate/2025')
bundle = joblib.load(BASE / 'conditional_theta_model_v3/conditional_theta_model_v3.joblib')
for head in bundle['theta_heads']:
    head.n_jobs = 2

states = {}
examples = []
details = {'eligible_rows': 0, 'duplicate_key_rows_within_parts': 0,
           'history_date_mismatch_rows': 0, 'oracle_prediction_nonmonotone_rows': 0,
           'true_switch_rows': 0, 'history_cutoff_after_rows': 0}

def record(mode, true_p, pred_p, mask=None):
    if mask is not None:
        true_p, pred_p = true_p[mask], pred_p[mask]
    if not len(true_p):
        return
    s = states.setdefault(mode, {'rows': 0, 'ae': 0.0, 'true_abs': 0.0,
                                'nonmonotone_rows': 0})
    s['rows'] += len(true_p)
    s['ae'] += float(np.abs(pred_p - true_p).sum())
    s['true_abs'] += float(np.abs(true_p).sum())
    s['nonmonotone_rows'] += int((np.diff(pred_p, axis=1) < -1e-6).any(axis=1).sum())

def decode(theta, templates, true_q):
    return m.pred_price_on_true_q(true_q, theta, m.reconstruct_price(theta, templates, centers))

for rel in manifest['parts']['test']:
    d = pd.read_pickle(FROZEN / rel)
    d = d.loc[m.history_ready(d) & m.template_history_ready(d)].reset_index(drop=True)
    if not len(d):
        continue
    true_q, true_p = m.true_curve(d)
    theta = m.raw_theta(d)
    hist = m.raw_theta(d, hist=True)
    template = m.norm(d['y_template_id']).to_numpy(object)
    origin = m.norm(d['hist_lag1_template_id']).to_numpy(object)
    switched = template != origin
    details['eligible_rows'] += len(d)
    details['true_switch_rows'] += int(switched.sum())
    details['duplicate_key_rows_within_parts'] += int(d.duplicated(
        ['participant_id', 'local_date', 'local_slot_seconds'], keep=False).sum())
    if 'hist_days_since_prev_same_slot' in d:
        details['history_date_mismatch_rows'] += int((
            (d['hist_theta_days_since_lag1'] - d['hist_days_since_prev_same_slot']).abs() > 1e-5).sum())
    # Operating interval after cutoff is a provenance check, not proof of unavailable bids.
    previous_day = pd.to_datetime(d['local_date']) - pd.to_timedelta(
        d['hist_theta_days_since_lag1'], unit='D')
    previous_slot = previous_day + pd.to_timedelta(d['local_slot_seconds'], unit='s')
    cutoff_local = pd.to_datetime(d['local_date']) - pd.Timedelta(days=1) + pd.Timedelta(hours=11)
    details['history_cutoff_after_rows'] += int((previous_slot > cutoff_local).sum())

    persisted = decode(hist, origin, true_q)
    oracle = decode(theta, template, true_q)
    for subset, mask in [('all', None), ('switch', switched), ('unchanged', ~switched)]:
        record('persistence_' + subset, true_p, persisted, mask)
        record('true_theta_true_template_' + subset, true_p, oracle, mask)

    # Representation diagnostics only: no fitted or forecast values in these alternatives.
    unwarped = theta.copy()
    unwarped[:, 5:10] = 0.2
    unwarped[:, 1] = (true_p[:, 14] - true_p[:, 0]) / 0.7
    unwarped[:, 2] = true_p[:, -1] - true_p[:, 14] - 0.3 * unwarped[:, 1]
    no_warp_price = decode(unwarped, template, true_q)
    template_shape = np.vstack([centers[t] for t in template])
    basic = d['p_anchor'].to_numpy(float)[:, None] + d['p_span'].to_numpy(float)[:, None] * template_shape
    record('oracle_no_warp_three_anchors', true_p, no_warp_price)
    record('oracle_template_anchor_span', true_p, basic)
    for tid in m.TEMPLATES:
        mask = template == tid
        record('true_theta_template_' + tid, true_p, oracle, mask)
        record('no_warp_template_' + tid, true_p, no_warp_price, mask)

    row_mae = np.abs(oracle - true_p).mean(axis=1)
    for i in np.argsort(row_mae)[-3:]:
        examples.append({'sample_id': str(d.iloc[i]['sample_id']), 'template': str(template[i]),
                         'oracle_mae': float(row_mae[i]),
                         'no_warp_mae': float(np.abs(no_warp_price[i] - true_p[i]).mean()),
                         'theta': theta[i].tolist(), 'true_p': true_p[i].tolist(),
                         'oracle_p': oracle[i].tolist()})

    ds = d.loc[switched].reset_index(drop=True)
    if len(ds):
        qt, pt, tt, templates = true_q[switched], true_p[switched], theta[switched], template[switched]
        predicted = m.predict_switch_theta(ds, origin[switched], templates, bundle)
        details['oracle_prediction_nonmonotone_rows'] += int((
            (predicted[:, 1] < 0) | (0.3 * predicted[:, 1] + predicted[:, 2] < 0)).sum())
        record('oracle_destination_model_switch', pt, decode(predicted, templates, qt))
        for name, cols in [('true_price_predicted_quantity', slice(0, 3)),
                           ('predicted_price_true_quantity', slice(3, 10)),
                           ('predicted_except_true_qshares', slice(5, 10)),
                           ('predicted_except_true_qscale', slice(3, 5))]:
            mixed = predicted.copy()
            mixed[:, cols] = tt[:, cols]
            record(name + '_switch', pt, decode(mixed, templates, qt))
    print(f'Finished {Path(rel).name}: {len(d)} rows', flush=True)

rows = []
for mode, s in states.items():
    rows.append({'mode': mode, **s, 'mae': s['ae'] / (s['rows'] * 21),
                 'wape_pct': 100 * s['ae'] / max(s['true_abs'], 1e-12)})
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / 'reconstruction_audit_metrics.csv', index=False, encoding='utf-8-sig')
details['unchanged_share_of_persistence_absolute_error'] = (
    states['persistence_unchanged']['ae'] / states['persistence_all']['ae'])
(OUT / 'audit_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
(OUT / 'oracle_examples.json').write_text(json.dumps(sorted(
    examples, key=lambda v: v['oracle_mae'], reverse=True)[:10], indent=2), encoding='utf-8')
print(metrics.loc[~metrics['mode'].str.contains('template_T|template_FLAT'),
                  ['mode', 'rows', 'mae', 'wape_pct', 'nonmonotone_rows']].to_string(index=False))
print(json.dumps(details, indent=2))
