#!/usr/bin/env python3
"""
Task 2 -- root cause: load every .model with torch.load, inspect the state dict.
Reports (a) tensor dtypes, (b) E0 per species per dtype, (c) whether E0/buffers
change on conversion to float64, (d) the E0 sum arithmetic for alpha/gamma.

CPU only.  Run:
  /home/jss/miniconda3/envs/mace/bin/python 01_state_dict.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

MODELS = {
    "v1": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v1_stagetwo.model",
    "v2": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v2_stagetwo.model",
    "v3b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v3b_stagetwo.model",
    "v5b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v5b_stagetwo.model",
    "v6": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model",
}

# alpha-ZrW2O8 P2_1_3, 44 atoms; gamma-ZrW2O8 P2_1_2_1, 132 atoms (= 3 x alpha)
ALPHA_COMP = {40: 4, 74: 8, 8: 32}    # Zr, W, O
GAMMA_COMP = {40: 12, 74: 24, 8: 96}  # Zr, W, O

out = {}

for name, path in MODELS.items():
    print("=" * 78)
    print(f"### {name}: {path}")
    print("=" * 78)
    model = torch.load(f=path, map_location="cpu", weights_only=False)
    model.eval()

    sd = model.state_dict()

    # ---- (a) dtype census of the state dict -------------------------------
    census = {}
    for k, v in sd.items():
        dt = str(v.dtype) if torch.is_tensor(v) else type(v).__name__
        census.setdefault(dt, []).append(k)
    print("\n(a) state-dict dtype census:")
    for dt, keys in sorted(census.items(), key=lambda kv: -len(kv[1])):
        print(f"    {dt:12s} : {len(keys):4d} tensors")
    non_float = {dt: ks for dt, ks in census.items()
                 if dt not in ("torch.float32", "torch.float64")}
    for dt, ks in non_float.items():
        print(f"    -> {dt}: {ks[:8]}{' ...' if len(ks) > 8 else ''}")

    # integer buffers that never move with dtype
    int_buffers = sorted(k for k, v in sd.items()
                         if torch.is_tensor(v) and not v.is_floating_point())

    # ---- atomic reference energies E0 -------------------------------------
    e0_keys = [k for k in sd if "atomic_energies" in k]
    print(f"\n(b) E0 tensors: {e0_keys}")
    e0_32 = {}
    for k in e0_keys:
        v32 = sd[k].to(torch.float32).clone()
        e0_32[k] = v32
        print(f"    {k}: dtype={sd[k].dtype} shape={tuple(sd[k].shape)}")
        print(f"        values(float32 view): {v32.flatten().tolist()}")

    z = sd["atomic_numbers"].tolist()
    zmap = {int(zz): i for i, zz in enumerate(z)}
    print(f"    atomic_numbers = {z}  (Z->index {zmap})   Z=8:O 40:Zr 74:W 13:Al")
    species = {8: "O", 40: "Zr", 74: "W", 13: "Al"}

    # ---- (d) E0 sum arithmetic -------------------------------------------
    e0row = sd[e0_keys[0]]
    if e0row.dim() == 2:          # (1, 4) = (n_heads, n_elements)
        e0row = e0row[0]
    e0_np32 = e0row.to(torch.float32).numpy().astype(np.float64)   # exact float32 values
    e0_np64 = e0row.to(torch.float64).numpy()

    def e0_sum(comp, e0vals, npdt):
        """sum_i n_i * E0_i, computed with the given numpy accumulator dtype"""
        acc = npdt(0.0)
        for zz, n in comp.items():
            acc = acc + npdt(n) * npdt(e0vals[zmap[zz]])
        return acc

    print("\n(d) E0 sum arithmetic (sum_i n_i * E0_i), eV:")
    print(f"    alpha 44 atoms = {ALPHA_COMP}")
    print(f"    gamma 132 atoms = {GAMMA_COMP}   (exactly 3 x alpha)")
    rows = {}
    for label, comp in (("alpha", ALPHA_COMP), ("gamma", GAMMA_COMP)):
        per_atom32 = sum(n * e0_np32[zmap[zz]] for zz, n in comp.items()) / sum(comp.values())
        rows[label] = {
            "sum_f32vals_f64acc": float(e0_sum(comp, e0_np32, np.float64)),
            "sum_f32vals_f32acc": float(e0_sum(comp, e0_np32, np.float32)),
            "sum_f64vals_f64acc": float(e0_sum(comp, e0_np64, np.float64)),
            "n_atoms": sum(comp.values()),
        }
        rows[label]["per_atom_f32vals"] = per_atom32
        print(f"    {label:5s}: f32-acc={rows[label]['sum_f32vals_f32acc']:.6f}"
              f"  f64-acc={rows[label]['sum_f32vals_f64acc']:.6f}"
              f"  E0/atom={per_atom32:.9f} eV = {per_atom32*1000:.6f} meV")

    dE0_peratom = (rows["gamma"]["sum_f32vals_f64acc"] / 132
                   - rows["alpha"]["sum_f32vals_f64acc"] / 44)
    print(f"    exact (float64 acc) E0_gamma/132 - E0_alpha/44 ="
          f" {dE0_peratom:.3e} eV/atom = {dE0_peratom*1000:.3e} meV/atom"
          "   <-- must be ~0: identical composition")

    # ---- (c) does .double() change E0 / any buffer? -----------------------
    m64 = torch.load(f=path, map_location="cpu", weights_only=False).double()
    sd64 = m64.state_dict()
    changed, promoted = [], []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        v64 = sd64[k]
        if v.dtype != v64.dtype:
            promoted.append((k, str(v.dtype), str(v64.dtype)))
        if v.is_floating_point():
            # value identity: float32 -> float64 promotion must be exact
            same = bool(torch.equal(v.to(torch.float64), v64))
            if not same:
                changed.append(k)
    print("\n(c) effect of model .double():")
    print(f"    floating tensors promoted f32->f64 : {len(promoted)}")
    print(f"    tensors whose VALUES changed      : {len(changed)} {changed[:10]}")
    print(f"    -> E0 float64 == float32(E0) exactly:"
          f" {bool(torch.equal(sd[e0_keys[0]].to(torch.float64), sd64[e0_keys[0]]))}")

    out[name] = {
        "path": path,
        "model_dtype": str(next(model.parameters()).dtype),
        "n_state_dict_entries": len(sd),
        "dtype_census": {dt: len(ks) for dt, ks in census.items()},
        "int_buffers": int_buffers,
        "e0_keys": e0_keys,
        "e0_values": {k: v.tolist() for k, v in e0_32.items()},
        "atomic_numbers": z,
        "n_floating_promoted": len(promoted),
        "values_changed_on_double": changed,
        "e0_sum": rows,
        "dE0_per_atom_meV": dE0_peratom * 1000,
        "r_max": float(model.r_max),
        "num_interactions": int(model.num_interactions) if hasattr(model, "num_interactions") else None,
    }
    del model, m64

os.makedirs("/home/jss/share/AlZrW2O8_MACE/diag_precision/json", exist_ok=True)
with open("/home/jss/share/AlZrW2O8_MACE/diag_precision/json/task2_state_dict.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote json/task2_state_dict.json")
