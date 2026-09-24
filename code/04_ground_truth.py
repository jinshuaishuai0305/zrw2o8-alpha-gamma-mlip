#!/usr/bin/env python3
"""
Task 3 -- cross-check against training labels.

NOTE ON DATA: the requested /home/jss/share/AlZrW2O8_MACE/v6_train.xyz does NOT
exist on this workstation (see the report's provenance section: it is generated
by merge_for_v6.py from v5b_train_1000.xyz + AlZrW_phonon2_train.xyz, none of
which are local either; config_v6.yaml points at it but the training ran on the
supercomputer).  The closest locally available labelled set with the *same*
-4.891 meV/atom gamma calibration is v4_zrw_part.xyz
(1163 alpha 44-atom frames + 600 gamma 132-atom frames, gamma shifted).

Selection: 40 alpha + 40 gamma frames, evenly spaced in energy after sorting, so
the sampled set spans the label energy range.

Run:  /home/jss/miniconda3/envs/mace/bin/python 04_ground_truth.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

OUT = "/home/jss/share/AlZrW2O8_MACE/diag_precision"
os.makedirs(f"{OUT}/json", exist_ok=True)

TRAIN = "/home/jss/share/AlZrW2O8_MACE/v4_zrw_part.xyz"
MODELS = {
    "v1": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v1_stagetwo.model",
    "v2": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v2_stagetwo.model",
    "v3b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v3b_stagetwo.model",
    "v5b": "/home/jss/share/AlZrW2O8_MACE/AlZr_v5b_stagetwo.model".replace("AlZr_", "AlZrW_"),
    "v6": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model",
}
NSAMP = 40

print(f"reading {TRAIN} ...")
all_frames = read(TRAIN, index=":")
print(f"  {len(all_frames)} frames")

picked = {}
for nat in (44, 132):
    sel = [a for a in all_frames if len(a) == nat]
    e = np.array([a.get_potential_energy() / nat for a in sel])
    order = np.argsort(e)
    idx = np.linspace(0, len(sel) - 1, NSAMP).round().astype(int)
    frames = [sel[i] for i in order[idx]]
    picked[nat] = frames
    print(f"  {nat}-atom frames: {len(sel)} available, using {len(frames)}; "
          f"label E/atom range [{e[order[idx]].min():.8f}, {e[order[idx]].max():.8f}] eV"
          f"  (= [{e[order[idx]].min()*1000:.4f}, {e[order[idx]].max()*1000:.4f}] meV/atom)")

labels = {nat: np.array([a.get_potential_energy() / nat for a in picked[nat]])
          for nat in (44, 132)}


def eval_model(model_path, dtype, repeat=1):
    """returns dict nat -> list over repeats of array of predicted E/atom"""
    res = {44: [], 132: []}
    for rep in range(repeat):
        calc = MACECalculator(model_paths=[model_path], device="cuda", default_dtype=dtype)
        for nat in (44, 132):
            preds = []
            for a in picked[nat]:
                a.calc = calc
                preds.append(float(a.get_potential_energy()) / nat)
            res[nat].append(np.array(preds))
        del calc
    return res


rows = {}
print(f"\n{'model':5s} {'dtype':8s} {'rep':>3s}  "
      f"{'RMSE_a':>8s} {'bias_a':>8s} | {'RMSE_g':>8s} {'bias_g':>8s} | "
      f"{'RMSE_all':>9s} {'bias_all':>9s}   (meV/atom)")

for mname, mpath in MODELS.items():
    rows[mname] = {}
    for dtype, nrep in (("float32", 3), ("float64", 1)):
        out = eval_model(mpath, dtype, repeat=nrep)
        entries = []
        for rep in range(nrep):
            stats = {}
            for nat in (44, 132):
                err = (out[nat][rep] - labels[nat]) * 1000.0
                stats[nat] = {
                    "rmse_meV_per_atom": float(np.sqrt((err ** 2).mean())),
                    "bias_meV_per_atom": float(err.mean()),
                    "max_abs_err": float(np.abs(err).max()),
                }
            eb = np.concatenate([(out[44][rep] - labels[44]) * 1000.0,
                                 (out[132][rep] - labels[132]) * 1000.0])
            stats["all"] = {"rmse_meV_per_atom": float(np.sqrt((eb ** 2).mean())),
                            "bias_meV_per_atom": float(eb.mean())}
            entries.append(stats)
            print(f"{mname:5s} {dtype:8s} {rep:3d}  "
                  f"{stats[44]['rmse_meV_per_atom']:8.4f} {stats[44]['bias_meV_per_atom']:8.4f} | "
                  f"{stats[132]['rmse_meV_per_atom']:8.4f} {stats[132]['bias_meV_per_atom']:8.4f} | "
                  f"{stats['all']['rmse_meV_per_atom']:9.4f} {stats['all']['bias_meV_per_atom']:9.4f}")
        rows[mname][dtype] = entries
    # mean over float32 repeats
    if len(rows[mname]["float32"]) > 1:
        r = np.array([e["all"]["rmse_meV_per_atom"] for e in rows[mname]["float32"]])
        print(f"      -> float32 RMSE_all run-to-run spread: "
              f"{r.min():.4f} .. {r.max():.4f}  ({r.max()-r.min():.4f} meV/atom)")

# ---------------- totals -------------------------------------------------
print("\n" + "=" * 100)
print("TASK 3 SUMMARY (mean over float32 repeats vs float64)")
print(f"{'model':5s} {'dtype':8s} {'RMSE_all meV/at':>16s} {'bias_all meV/at':>16s} "
      f"{'RMSE_alpha':>12s} {'RMSE_gamma':>12s}")
summary = {}
for mname in MODELS:
    summary[mname] = {}
    for dtype in ("float32", "float64"):
        e = rows[mname][dtype]
        rm = np.mean([x["all"]["rmse_meV_per_atom"] for x in e])
        bi = np.mean([x["all"]["bias_meV_per_atom"] for x in e])
        ra = np.mean([x[44]["rmse_meV_per_atom"] for x in e])
        rg = np.mean([x[132]["rmse_meV_per_atom"] for x in e])
        summary[mname][dtype] = {"rmse_all": float(rm), "bias_all": float(bi),
                                 "rmse_alpha": float(ra), "rmse_gamma": float(rg),
                                 "n_float32_repeats": len(e)}
        print(f"{mname:5s} {dtype:8s} {rm:16.4f} {bi:16.4f} {ra:12.4f} {rg:12.4f}")

with open(f"{OUT}/json/task3_ground_truth.json", "w") as fh:
    json.dump({"train_file_used": TRAIN,
               "note": "v6_train.xyz absent on this workstation",
               "n_alpha_frames": len(picked[44]), "n_gamma_frames": len(picked[132]),
               "label_range_eV_per_atom": {
                   "alpha": [float(labels[44].min()), float(labels[44].max())],
                   "gamma": [float(labels[132].min()), float(labels[132].max())]},
               "raw": rows, "summary": summary}, fh, indent=2)
print("\nwrote json/task3_ground_truth.json")
