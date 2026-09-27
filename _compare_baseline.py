"""Verify the refactored pipeline reproduces the pre-refactor outputs.

Run after ``uv run python PART2.py``; compares ``part2_outputs/`` against the
frozen ``part2_outputs_baseline/``.

Two columns are *expected* to differ: ``region`` (and ``peak_channel`` in
``visual_units.csv``), because the loader now converts ALF's 1-based
``clusters.peakChannel`` to a 0-based channel index and the baseline was written
with the old off-by-one convention. Every other shared column must match.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("part2_outputs_baseline")
NEW = Path("part2_outputs")


def compare(fname, core_cols=None, label=None):
    a, b = pd.read_csv(BASE / fname), pd.read_csv(NEW / fname)
    cols = core_cols or [c for c in a.columns if c in b.columns]
    bad = []
    for c in cols:
        x, y = a[c], b[c]
        if len(x) != len(y):
            bad.append(f"{c}: length {len(x)} vs {len(y)}")
            continue
        try:
            if not np.allclose(x.astype(float), y.astype(float), equal_nan=True):
                bad.append(f"{c}: values differ")
        except (TypeError, ValueError):
            if not (x.astype(str) == y.astype(str)).all():
                bad.append(f"{c}: strings differ")
    extra = sorted(set(b.columns) - set(a.columns))
    status = "OK" if not bad else "DIFF"
    print(f"[{status}] {label or fname}")
    print(f"        rows {a.shape[0]} -> {b.shape[0]}, "
          f"shared cols checked: {len(cols)}, new cols: {extra}")
    for x in bad:
        print(f"        !! {x}")
    return not bad


ok = True
ok &= compare("grating_stimuli.csv")
ok &= compare("stimulus_design.csv")
ok &= compare("tuning.csv")
ok &= compare("regression.csv",
              core_cols=["cluster_id", "r2", "p", "side_preference",
                         "beta_intercept", "beta_contrast_left",
                         "beta_contrast_right", "beta_interaction",
                         "significant"])
ok &= compare("receptive_fields.csv",
              core_cols=["cluster_id", "z_max", "significant", "peak_x", "peak_y",
                         "n_flash_spikes"])

# --- flashes: 800 of 7962 timestamps are duplicated (simultaneous flashes), so
# the ordering among ties is arbitrary. Compare as a multiset of (time,x,y).
a = pd.read_csv(BASE / "flashes.csv")
b = pd.read_csv(NEW / "flashes.csv")
key_a = sorted(map(tuple, a[["time", "x", "y"]].round(9).values.tolist()))
key_b = sorted(map(tuple, b[["time", "x", "y"]].round(9).values.tolist()))
same_multiset = key_a == key_b
ok &= same_multiset
print(f"[{'OK' if same_multiset else 'DIFF'}] flashes.csv (multiset of tied timestamps)")
print(f"        rows {len(a)} -> {len(b)}, duplicate timestamps: "
      f"{len(a) - a['time'].nunique()}, same (time,x,y) multiset: {same_multiset}")

# --- visually_responsive: baseline wrote NaN for non-candidate units, the
# refactor writes explicit False; compare after normalising.
u1 = pd.read_csv(BASE / "visual_units.csv")
u2 = pd.read_csv(NEW / "visual_units.csv")
norm_a = u1["visually_responsive"].fillna(False).infer_objects(copy=False).astype(bool)
norm_b = u2["visually_responsive"].fillna(False).infer_objects(copy=False).astype(bool)
same_flag = bool((norm_a.values == norm_b.values).all())
ok &= same_flag
print(f"[{'OK' if same_flag else 'DIFF'}] visual_units.csv (visually_responsive after NaN->False)")
print(f"        rows {u1.shape[0]} -> {u2.shape[0]}, identical after normalisation: {same_flag}")
ok &= compare("visual_units.csv",
              core_cols=[c for c in u1.columns if c != "visually_responsive"])
ok &= bool((u1["visual_refined"].values == u2["refined"].values).all())

a = pd.read_csv(BASE / "part2_summary.csv").iloc[0]
b = pd.read_csv(NEW / "part2_summary.csv").iloc[0]
keys = [c for c in a.index if c in b.index and c != "session"]
diffs = {c: (a[c], b[c]) for c in keys if str(a[c]) != str(b[c])}
print(f"[{'OK' if not diffs else 'DIFF'}] part2_summary.csv (shared keys: {len(keys)})")
for k, v in diffs.items():
    print(f"        !! {k}: {v[0]} -> {v[1]}")

print("\nOVERALL:", "refactor reproduces baseline" if (ok and not diffs)
      else "see differences above")
