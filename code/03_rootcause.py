#!/usr/bin/env python3
"""
Task 2 root cause -- decisive experiments.

E1  Determinism: repeat the same float32 and float64 evaluation many times,
    in-process and across processes.  Reports the run-to-run scatter of dE.
E2  E0 isolation: zero the atomic_energies buffer IN MEMORY (E0 enters the
    total energy purely additively, so this removes the ~1.8 keV/atom reference
    without touching the network).  Compare dE with and without E0, per dtype.
E3  Direct measurement of the model's own E0 accumulation: capture the per-atom
    E0 tensor with a forward hook, then accumulate it exactly the way the model
    does (index_add_ / scatter_sum in the working dtype) and compare with the
    exact float64 sum.

Run:  /home/jss/miniconda3/envs/mace/bin/python 03_rootcause.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

OUT = "/home/jss/share/AlZrW2O8_MACE/diag_precision"
os.makedirs(f"{OUT}/json", exist_ok=True)

MODELS = {
    "v1": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v1_stagetwo.model",
    "v2": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZr_v2_stagetwo.model".replace("AlZr_", "AlZrW_"),
    "v3b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v3b_stagetwo.model",
    "v5b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v5b_stagetwo.model",
    "v6": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model",
}
CELLS = {
    "alpha": "/home/jss/share/AlZrW2O8_MACE/elastic/_alpha_lmps.vasp",
    "gamma": "/home/jss/share/AlZrW2O8_MACE/elastic/_gamma_lmps.vasp",
}
N = {"alpha": 44, "gamma": 132}

atoms = {k: read(v) for k, v in CELLS.items()}


def build(model_path, dtype, device, zero_e0=False):
    calc = MACECalculator(model_paths=[model_path], device=device, default_dtype=dtype)
    m = calc.models[0]
    if zero_e0:
        with torch.no_grad():
            m.atomic_energies_fn.atomic_energies.zero_()
    return calc, m


def energy(calc, m, ats, capture_e0=False):
    box = {}
    h = None
    if capture_e0:
        def hook(_mod, _inp, outp):
            box["node_e0"] = outp.detach().clone()
            box["batch"] = None
        h = m.atomic_energies_fn.register_forward_hook(hook)
    ats.calc = calc
    e = float(ats.get_potential_energy())
    if h is not None:
        h.remove()
    return e, box


def exact_e0_sum(node_e0):
    """float64 exact sum of the per-atom E0 vector"""
    return float(node_e0.to(torch.float64).sum().item())


def model_style_e0_sum(node_e0):
    """replicate MACE scatter_sum: zeros(dtype).index_add_(0, batch, src)"""
    n = node_e0.shape[0]
    acc = torch.zeros(1, dtype=node_e0.dtype, device=node_e0.device)
    idx = torch.zeros(n, dtype=torch.int64, device=node_e0.device)
    acc.index_add_(0, idx, node_e0.reshape(-1).to(node_e0.dtype))
    return float(acc.to(torch.float64).item())


def dE_from(ea, eg, na, ng):
    return (eg / ng - ea / na) * 1000.0


res = {"determinism": {}, "zero_e0": {}, "e0_accumulation": {}}

# ---------------------------------------------------------------- E1
print("=" * 100)
print("E1  DETERMINISM  (repeat the identical evaluation N times)")
REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for mname, mpath in MODELS.items():
    res["determinism"][mname] = {}
    for dtype in ("float32", "float64"):
        for device in ("cuda", "cpu"):
            vals_a, vals_g = [], []
            for _ in range(REPEAT):
                calc, m = build(mpath, dtype, device)
                ea, _ = energy(calc, m, atoms["alpha"])
                eg, _ = energy(calc, m, atoms["gamma"])
                vals_a.append(ea)
                vals_g.append(eg)
                del calc
            da = dE_from(np.mean(vals_a), np.mean(vals_g), N["alpha"], N["gamma"])
            spread_a = (max(vals_a) - min(vals_a)) * 1000 / N["alpha"]
            spread_g = (max(vals_g) - min(vals_g)) * 1000 / N["gamma"]
            dEs = [dE_from(a, g, N["alpha"], N["gamma"])
                   for a, g in zip(vals_a, vals_g)]
            res["determinism"][mname][f"{dtype}_{device}"] = {
                "dE_mean_meV_per_atom": da,
                "dE_min": min(dEs), "dE_max": max(dEs),
                "dE_spread_meV_per_atom": max(dEs) - min(dEs),
                "E_alpha_spread_meV_per_atom": spread_a,
                "E_gamma_spread_meV_per_atom": spread_g,
                "E_alpha_spread_eV_total": (max(vals_a) - min(vals_a)),
                "E_gamma_spread_eV_total": (max(vals_g) - min(vals_g)),
                "n_repeat": REPEAT,
            }
            print(f"  {mname:4s} {dtype:8s} {device:5s} dE = {da:8.4f} meV/at"
                  f"   spread over {REPEAT} runs = {max(dEs)-min(dEs):8.4f} meV/at"
                  f"   (E_gamma total spread = {(max(vals_g)-min(vals_g))*1000:9.4f} meV)")

# ---------------------------------------------------------------- E2
print("\n" + "=" * 100)
print("E2  E0 ISOLATION  (atomic_energies zeroed in memory; network untouched)")
print(f"{'model':5s} {'dtype':8s} {'dE full meV/at':>16s} {'dE E0=0 meV/at':>16s} "
      f"{'E0 contribution':>16s}")
for mname, mpath in MODELS.items():
    res["zero_e0"][mname] = {}
    for dtype in ("float32", "float64"):
        calc, m = build(mpath, dtype, "cuda")
        ea, _ = energy(calc, m, atoms["alpha"])
        eg, _ = energy(calc, m, atoms["gamma"])
        d_full = dE_from(ea, eg, N["alpha"], N["gamma"])
        calc0, m0 = build(mpath, dtype, "cuda", zero_e0=True)
        ea0, _ = energy(calc0, m0, atoms["alpha"])
        eg0, _ = energy(calc0, m0, atoms["gamma"])
        d_zero = dE_from(ea0, eg0, N["alpha"], N["gamma"])
        res["zero_e0"][mname][dtype] = {
            "dE_full": d_full, "dE_E0zeroed": d_zero,
            "E0_contribution_meV_per_atom": d_full - d_zero,
            "E_alpha_full_eV": ea, "E_gamma_full_eV": eg,
            "E_alpha_E0zero_eV": ea0, "E_gamma_E0zero_eV": eg0,
        }
        print(f"{mname:5s} {dtype:8s} {d_full:16.4f} {d_zero:16.4f} "
              f"{d_full-d_zero:16.4f}")

# ---------------------------------------------------------------- E3
print("\n" + "=" * 100)
print("E3  E0 SCATTER-ACCUMULATION ERROR (model's own index_add_ vs exact float64)")
print(f"{'model':5s} {'dtype':8s} {'cell':6s} {'E0 index_add_ (eV)':>22s} "
      f"{'E0 exact f64 (eV)':>20s} {'error (meV)':>14s} {'error (meV/at)':>16s}")
for mname, mpath in MODELS.items():
    res["e0_accumulation"][mname] = {}
    for dtype in ("float32", "float64"):
        res["e0_accumulation"][mname][dtype] = {}
        for cname in ("alpha", "gamma"):
            calc, m = build(mpath, dtype, "cuda")
            e, box = energy(calc, m, atoms[cname], capture_e0=True)
            ne = box["node_e0"]
            if ne.dim() > 1:
                ne = ne.squeeze(-1)
            acc = model_style_e0_sum(ne)
            ex = exact_e0_sum(ne)
            err = (acc - ex) * 1000.0
            res["e0_accumulation"][mname][dtype][cname] = {
                "E0_index_add_eV": acc, "E0_exact_f64_eV": ex,
                "error_meV": err, "error_meV_per_atom": err / N[cname],
                "node_e0_dtype": str(ne.dtype),
                "E_total_eV": e,
            }
            print(f"{mname:5s} {dtype:8s} {cname:6s} {acc:22.9f} {ex:20.9f} "
                  f"{err:14.4f} {err/N[cname]:16.4f}")

with open(f"{OUT}/json/task2_rootcause.json", "w") as fh:
    json.dump(res, fh, indent=2)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 100)
print("SUMMARY: dE float64 - dE float32  vs  E0-accumulation contribution")
for mname in MODELS:
    det = res["determinism"][mname]
    d64 = det["float64_cuda"]["dE_mean_meV_per_atom"]
    d32 = det["float32_cuda"]["dE_mean_meV_per_atom"]
    z = res["zero_e0"][mname]
    print(f"  {mname:4s}  dE32={d32:8.4f}  dE64={d64:8.4f}  diff={d64-d32:7.4f} | "
          f"E0-free: dE32={z['float32']['dE_E0zeroed']:8.4f} "
          f"dE64={z['float64']['dE_E0zeroed']:8.4f} "
          f"diff={z['float64']['dE_E0zeroed']-z['float32']['dE_E0zeroed']:7.4f} | "
          f"f32 dE run-to-run spread={det['float32_cuda']['dE_spread_meV_per_atom']:.4f}")
print("\nwrote json/task2_rootcause.json")
