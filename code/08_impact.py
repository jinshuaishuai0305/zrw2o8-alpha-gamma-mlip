#!/usr/bin/env python3
"""
Practical impact of the CUDA-float32 nondeterminism, measured on REAL
training-like frames (the local .xyz training files, which carry DFT energies
and forces, so the size of the nondeterminism can be compared with the model's
own DFT error).

For 30 frames drawn from the three local training files:
  * CUDA float32, NREP=10 repeats of the same frame   -> run-to-run spread
  * CUDA float64, 1 evaluation                        -> dtype reference
and we report per frame
  * E/atom spread (meV/atom) over the 10 float32 repeats
  * |E/atom(f32) - E/atom(f64)|            (dtype bias)
  * force matrix spread over the 10 repeats (meV/A)
  * force RMSE f32 vs f64                   (dtype force error)
  * relative energies between frames: spread of (E_i - E_0)/atom in float32
    vs the same quantity in float64  -- this is what the project actually uses
  * MACE(f64) vs DFT energy MAE / force RMSE for context

Run:
  /home/jss/miniconda3/envs/mace/bin/python 08_impact.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from ase.io import read

from mace.calculators import MACECalculator

OUT = "/home/jss/share/AlZrW2O8_MACE/diag_precision"
os.makedirs(f"{OUT}/json", exist_ok=True)

MODEL = "/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model"
FILES = {
    "zrw_aligned": ("/home/jss/share/AlZrW2O8_MACE/ZrW2O8_train_aligned.xyz", 10),
    "iface": ("/home/jss/share/AlZrW2O8_MACE/AlZrW_iface_train.xyz", 10),
    "bonded": ("/home/jss/share/AlZrW2O8_MACE/AlZrW_bonded_train.xyz", 10),
}
NREP = 10
ERR_SAMPLE = 50          # frames per file for the MACE-vs-DFT error context

print("=" * 100)
print(f"impact study: model {os.path.basename(MODEL)}, NREP={NREP}")
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
print("=" * 100)

frames = {}
dft = {}
for tag, (fn, n) in FILES.items():
    all_f = read(fn, index=":")
    take = np.linspace(0, len(all_f) - 1, n).astype(int)
    frames[tag] = [(int(i), all_f[int(i)]) for i in take]
    dft[tag] = all_f
    print(f"{tag:14s} {fn.split('/')[-1]:28s} {len(all_f):5d} frames, using {n}")


def build(calc, ats):
    b = calc._atoms_to_batch(ats)
    md = next(calc.models[0].parameters()).dtype
    for key in b.keys:
        v = b[key]
        if torch.is_tensor(v) and torch.is_floating_point(v):
            b[key] = v.to(dtype=md)
    return b


def ev(calc, ats):
    bd = calc._clone_batch(build(calc, ats)).to_dict()
    o = calc.models[0](bd, training=False, compute_force=True,
                       compute_stress=False, compute_edge_forces=False,
                       compute_atomic_stresses=False)
    return (float(o["energy"].detach().cpu()),
            o["node_energy"].detach().cpu().numpy().astype(np.float64),
            o["forces"].detach().cpu().numpy().astype(np.float64))


c32 = MACECalculator(model_paths=[MODEL], device="cuda", default_dtype="float32")
c64 = MACECalculator(model_paths=[MODEL], device="cuda", default_dtype="float64")

rows = []
for tag in FILES:
    for idx, ats in frames[tag]:
        n = len(ats)
        E32, V32, F32 = [], [], []
        for _ in range(NREP):
            e, v, f = ev(c32, ats)
            E32.append(e); V32.append(v); F32.append(f)
        E32 = np.array(E32); V32 = np.array(V32); F32 = np.array(F32)
        e64, v64, f64 = ev(c64, ats)
        sd = F32.std(axis=0)
        rows.append({
            "file": tag, "frame": idx, "n_atoms": n,
            "E_per_atom_f32_mean": float(E32.mean() / n),
            "E_per_atom_f32_range_meV": float((E32.max() - E32.min()) * 1000 / n),
            "E_per_atom_f32_std_meV": float(E32.std() * 1000 / n),
            "E_per_atom_f64": float(e64 / n),
            "E_per_atom_f32_minus_f64_meV": float((E32.mean() - e64) * 1000 / n),
            "peratom_vec_max_dev_meV": float(
                np.abs(V32 - V32[0]).max() * 1000),
            "force_max_comp_std_meV_per_A": float(sd.max() * 1000),
            "force_rms_comp_std_meV_per_A": float(np.sqrt((sd ** 2).mean()) * 1000),
            "force_rmse_f32_vs_f64_meV_per_A": float(
                np.sqrt(((F32.mean(axis=0) - f64) ** 2).mean()) * 1000),
        })
    print(f"  {tag}: done")

R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k not in ("file",)}

# ---- relative energies between frames: what the project actually uses -----
rel = {}
for tag in FILES:
    sel = [r for r in rows if r["file"] == tag]
    e32 = np.array([r["E_per_atom_f32_mean"] for r in sel])
    e64 = np.array([r["E_per_atom_f64"] for r in sel])
    # use the per-frame float32 *range* as the uncertainty of each relative energy
    rng = np.array([r["E_per_atom_f32_range_meV"] for r in sel])
    rel[tag] = {
        "n_frames": len(sel),
        "max_abs_f32_minus_f64_rel_meV_per_atom": float(np.abs(
            (e32 - e32[0]) - (e64 - e64[0])).max()),
        "min_frame_energy_separation_meV_per_atom": float(
            np.diff(np.sort(e64)).min()),
        "max_f32_range_meV_per_atom": float(rng.max()),
        "median_f32_range_meV_per_atom": float(np.median(rng)),
    }

# ---- MACE(f64) vs DFT error, for context ---------------------------------
ctx = {}
for tag, (fn, _) in FILES.items():
    fl = dft[tag]
    take = np.linspace(0, len(fl) - 1, min(ERR_SAMPLE, len(fl))).astype(int)
    de, fr = [], []
    for i in take:
        a = fl[int(i)]
        E, _, F = ev(c64, a)
        de.append(E / len(a) - a.get_potential_energy() / len(a))
        fr.append(((F - a.get_forces()) ** 2).mean())
    ctx[tag] = {
        "n_frames": int(len(take)),
        "E_MAE_meV_per_atom": float(np.abs(np.array(de)).mean() * 1000),
        "E_RMSE_meV_per_atom": float(np.sqrt((np.array(de) ** 2).mean()) * 1000),
        "F_RMSE_meV_per_A": float(np.sqrt(np.mean(fr)) * 1000),
    }
    print(f"  context {tag}: E MAE={ctx[tag]['E_MAE_meV_per_atom']:.3f} meV/at  "
          f"F RMSE={ctx[tag]['F_RMSE_meV_per_A']:.3f} meV/A")

print("\n" + "=" * 100)
print("1) RUN-TO-RUN SPREAD OF A CUDA-float32 EVALUATION (30 frames, 10 repeats each)")
print(f"{'file':14s} {'E/at range med':>15s} {'E/at range max':>15s} "
      f"{'|f32-f64| max':>14s} {'perat vec maxdev':>17s} {'F comp std max':>15s} "
      f"{'F rmse f32-f64':>15s}")
print("-" * 100)
for tag in FILES:
    s = [r for r in rows if r["file"] == tag]
    g = lambda k: np.array([r[k] for r in s])
    print(f"{tag:14s} {np.median(g('E_per_atom_f32_range_meV')):15.4f} "
          f"{g('E_per_atom_f32_range_meV').max():15.4f} "
          f"{np.abs(g('E_per_atom_f32_minus_f64_meV')).max():14.4f} "
          f"{g('peratom_vec_max_dev_meV').max():17.6f} "
          f"{g('force_max_comp_std_meV_per_A').max():15.6f} "
          f"{g('force_rmse_f32_vs_f64_meV_per_A').max():15.4f}")

allrng = R["E_per_atom_f32_range_meV"]
print(f"\nOVERALL: E/atom run-to-run spread  median={np.median(allrng):.4f} "
      f"mean={allrng.mean():.4f} max={allrng.max():.4f} meV/atom")
print(f"         frames with spread > 1 meV/atom : "
      f"{(allrng > 1).sum()}/{len(allrng)}")
print(f"         frames with spread > 0.1 meV/atom: "
      f"{(allrng > 0.1).sum()}/{len(allrng)}")
print(f"         frames with spread == 0         : {(allrng == 0).sum()}/{len(allrng)}")

print("\n2) DTYPE BIAS (float32 mean - float64), meV/atom")
b = np.abs(R["E_per_atom_f32_minus_f64_meV"])
print(f"   median={np.median(b):.4f}  mean={b.mean():.4f}  max={b.max():.4f}")

print("\n3) FORCES AND PER-ATOM ENERGIES UNDER float32 (are they safe?)")
print(f"   per-atom energy vector: max deviation over 10 repeats, worst frame = "
      f"{R['peratom_vec_max_dev_meV'].max():.6f} meV")
print(f"   force components      : max std over 10 repeats, worst frame = "
      f"{R['force_max_comp_std_meV_per_A'].max():.6f} meV/A")
print(f"   force RMSE f32 vs f64 : max over frames = "
      f"{R['force_rmse_f32_vs_f64_meV_per_A'].max():.4f} meV/A")

print("\n4) RELATIVE ENERGIES BETWEEN FRAMES (what the project really uses)")
for tag in FILES:
    r = rel[tag]
    print(f"   {tag:14s} min frame separation (f64) = "
          f"{r['min_frame_energy_separation_meV_per_atom']:8.4f} meV/at | "
          f"|rel_f32 - rel_f64| max = "
          f"{r['max_abs_f32_minus_f64_rel_meV_per_atom']:7.4f} meV/at | "
          f"f32 spread per frame median = "
          f"{r['median_f32_range_meV_per_atom']:.4f} meV/at")

print("\n5) CONTEXT: MACE(float64) vs DFT on the same files")
for tag in FILES:
    c = ctx[tag]
    print(f"   {tag:14s} E MAE={c['E_MAE_meV_per_atom']:8.4f} meV/at  "
          f"F RMSE={c['F_RMSE_meV_per_A']:8.4f} meV/A")

out = {
    "model": MODEL, "nrep": NREP,
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "per_frame": rows,
    "summary": {
        "E_per_atom_f32_range_meV": {
            "median": float(np.median(allrng)), "mean": float(allrng.mean()),
            "max": float(allrng.max()), "min": float(allrng.min()),
            "n_gt_1meV": int((allrng > 1).sum()), "n": int(len(allrng)),
            "n_exactly_zero": int((allrng == 0).sum()),
        },
        "dtype_bias_meV_per_atom": {
            "median": float(np.median(b)), "max": float(b.max())},
        "peratom_vec_max_dev_meV": float(R["peratom_vec_max_dev_meV"].max()),
        "force_max_comp_std_meV_per_A": float(
            R["force_max_comp_std_meV_per_A"].max()),
        "force_rmse_f32_vs_f64_meV_per_A": float(
            R["force_rmse_f32_vs_f64_meV_per_A"].max()),
    },
    "relative_energies": rel,
    "mace_f64_vs_dft": ctx,
}
fn = f"{OUT}/json/task3_impact.json"
with open(fn, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwrote {fn}")
