#!/usr/bin/env python3
"""
Aggregate every 03_nondeterminism.py JSON into the final tables, verify the
float32 quantisation lattice of the total energy, and establish what the
model's `node_energy` output actually is.

Run:  /home/jss/miniconda3/envs/mace/bin/python 09_summarize.py
"""
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

OUT = "/home/jss/share/AlZrW2O8_MACE/diag_precision"
CONFIGS = ["base", "samebatch", "det", "notf32", "cublas",
           "f64scatter", "f64", "cpu32"]
MODELS = ["v6", "v3b"]

pooled = {}
print("=" * 108)
print("A) POOLED dE = E_gamma/132 - E_alpha/44, CUDA float32 vs configurations")
print("=" * 108)
print(f"{'config':12s} {'model':5s} {'variant':13s} {'n':>4s} {'min':>9s} "
      f"{'max':>9s} {'range':>8s} {'std':>7s}  verdict")
print("-" * 108)
for cfg in CONFIGS:
    files = sorted(glob.glob(f"{OUT}/json/task3_{cfg}_p*.json"))
    if not files:
        continue
    for mk in MODELS:
        for tag in ("freshbatch", "samebatch", "ase_calculator"):
            vals, rngs = [], []
            for fn in files:
                d = json.load(open(fn))
                e = d["cells"].get(mk, {}).get(tag)
                if e is None:
                    continue
                vals += e["dE_values_meV_per_atom"]
                rngs.append(e["dE_meV_per_atom"]["range"])
            if not vals:
                continue
            v = np.array(vals)
            pooled[(cfg, mk, tag)] = v
            verdict = "REPRODUCIBLE" if v.max() - v.min() == 0 else "NONDETERMINISTIC"
            print(f"{cfg:12s} {mk:5s} {tag:13s} {v.size:4d} {v.min():9.4f} "
                  f"{v.max():9.4f} {v.max()-v.min():8.4f} {v.std():7.4f}  {verdict}")
    print()

print("=" * 108)
print("B) WHERE DOES IT LIVE? per-atom energy vector vs total vs forces")
print("=" * 108)
print(f"{'config':12s} {'model':5s} {'perat-vec maxdev meV':>21s} "
      f"{'E/at range max meV':>19s} {'F comp std max meV/A':>21s}")
print("-" * 108)
for cfg in ["base", "det", "f64scatter", "f64", "cpu32"]:
    files = sorted(glob.glob(f"{OUT}/json/task3_{cfg}_p*.json"))
    if not files:
        continue
    for mk in MODELS:
        pv, er, fs = [], [], []
        for fn in files:
            d = json.load(open(fn))
            for tag in ("freshbatch", "samebatch", "ase_calculator"):
                e = d["cells"].get(mk, {}).get(tag)
                if e is None:
                    continue
                for c in ("alpha", "gamma"):
                    if f"peratom_energy_vec_{c}" in e:
                        pv.append(e[f"peratom_energy_vec_{c}"]["max_abs_dev_meV"])
                    er.append(e[f"E_{c}_per_atom_eV"]["range"] * 1000)
                    fs.append(e[f"forces_{c}"]["max_component_std_meV_per_A"])
        if not er:
            continue
        print(f"{cfg:12s} {mk:5s} {max(pv):21.6f} {max(er):19.4f} {max(fs):21.6f}")

print()
print("=" * 108)
print("C) IS THE TOTAL ENERGY QUANTISED TO THE float32 ulp?")
print("=" * 108)
Q = 0.015625  # eV = 2^-6, the float32 ulp of a magnitude ~1e5 eV number
for cfg in ["base", "samebatch"]:
    files = sorted(glob.glob(f"{OUT}/json/task3_{cfg}_p*.json"))
    for mk in ["v6"]:
        for c in ("alpha", "gamma"):
            vals = []
            for fn in files:
                d = json.load(open(fn))
                for tag in ("freshbatch", "samebatch", "ase_calculator"):
                    e = d["cells"].get(mk, {}).get(tag)
                    if e:
                        vals += e[f"E_{c}_total_eV"]["min"], e[f"E_{c}_total_eV"]["max"]
            v = np.array(sorted(set(np.round(vals, 9))))
            if v.size < 2:
                continue
            # express each distinct level as an integer multiple of the quantum
            k = (v - v[0]) / Q
            print(f"  {cfg}/{mk}/{c}: {v.size} distinct totals over "
                  f"{len(files)} processes; |E|~{abs(v.mean())/1000:.1f} keV")
            print(f"     E/atom spread = {(v.max()-v.min())/ (44 if c=='alpha' else 132)*1000:.4f} meV/atom")
            print(f"     offsets in units of {Q*1000:.4f} meV (2^-6 eV): "
                  f"{np.round(k,4).tolist()}")
            print(f"     all integral? {np.allclose(k, np.round(k), atol=1e-3)}")

print()
print("=" * 108)
print("D) what is the model's node_energy output?")
print("=" * 108)
calc = MACECalculator(model_paths=["/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model"],
                      device="cuda", default_dtype="float64")
for tag, p in [("alpha", "/home/jss/share/AlZrW2O8_MACE/elastic/_alpha_lmps.vasp"),
               ("gamma", "/home/jss/share/AlZrW2O8_MACE/elastic/_gamma_lmps.vasp")]:
    ats = read(p)
    b = calc._atoms_to_batch(ats)
    md = next(calc.models[0].parameters()).dtype
    for k in b.keys:
        v = b[k]
        if torch.is_tensor(v) and torch.is_floating_point(v):
            b[k] = v.to(dtype=md)
    o = calc.models[0](calc._clone_batch(b).to_dict(), training=False,
                       compute_force=True, compute_stress=False,
                       compute_edge_forces=False, compute_atomic_stresses=False)
    ne = o["node_energy"].detach().cpu().numpy().ravel()
    E = float(o["energy"].detach().cpu())
    print(f"  {tag}: n={len(ats)}  sum(node_energy)={ne.sum():.6f} eV   "
          f"out['energy']={E:.6f} eV   diff={ne.sum()-E:.6f} eV")
    print(f"        node_energy/atom mean={ne.mean():.6f} eV  "
          f"(e0/atom = -1791.041659)  -> node_energy is the "
          f"{'NET' if abs(ne.mean()) < 100 else 'FULL'} per-atom energy")
    print(f"        sum(contributions)={float(o['contributions'].sum().detach().cpu()):.6f} eV")
