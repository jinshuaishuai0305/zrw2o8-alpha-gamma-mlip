#!/usr/bin/env python3
"""
Run-to-run reproducibility of MACE evaluation.

The question: float32 evaluation on CUDA is NOT reproducible run-to-run for the
same structure with the same calculator, while float64 (CPU and CUDA) is
bit-stable.  This script characterises and localises that.

Configurations (argv[1]):
  base        : as-is, CUDA float32
  samebatch   : as-is, but the torch_geometric batch is built ONCE and reused
                (separates graph construction from the forward pass)
  det         : torch.use_deterministic_algorithms(True)
  notf32      : allow_tf32=False (matmul + cudnn), matmul precision "highest"
  cublas      : CUBLAS_WORKSPACE_CONFIG=:4096:8 (set by the driver, before CUDA init)
  f64scatter  : mace's scatter_sum patched to accumulate in float64
                -> causal proof that scatter_add_ is the source

For every configuration, and for the two reference cells, we repeat the SAME
evaluation N times in one process and report the distribution of
    E/atom, dE = E_gamma/132 - E_alpha/44,
    the per-atom energy vector (model "energies" output),
    the force matrix.

Usage:
  python 03_nondeterminism.py <config> [model_key]
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

OUT = "/home/jss/share/AlZrW2O8_MACE/diag_precision"
os.makedirs(f"{OUT}/json", exist_ok=True)

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL_KEYS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["v6"]
NREP = int(os.environ.get("NREP", "10"))

MODELS = {
    "v6": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model",
    "v3b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v3b_stagetwo.model",
    "v5b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v5b_stagetwo.model",
}
CELLS = {
    "alpha": "/home/jss/share/AlZrW2O8_MACE/elastic/_alpha_lmps.vasp",
    "gamma": "/home/jss/share/AlZrW2O8_MACE/elastic/_gamma_lmps.vasp",
}
NAT = {"alpha": 44, "gamma": 132}

# NOTE: CUBLAS_WORKSPACE_CONFIG has to be set before the CUDA context exists;
# the driver script sets it, we only report it here.
CUBLAS_ENV = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>")

# --------------------------------------------------------------------------
# config-specific set-up, done BEFORE any model is built
# --------------------------------------------------------------------------
patch_info = None
DEVICE, DTYPE = "cuda", "float32"
if CONFIG == "det":
    torch.use_deterministic_algorithms(True)
elif CONFIG == "f64":
    DTYPE = "float64"
elif CONFIG == "cpu32":
    DEVICE, DTYPE = "cpu", "float32"
elif CONFIG == "notf32":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
elif CONFIG == "f64scatter":
    import mace.tools.scatter as _msc
    import mace.modules.models as _mm
    import mace.modules.blocks as _mb

    def scatter_sum_f64(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        """scatter_sum with float64 accumulation, returned in the input dtype.

        float64 has ~1e-16 relative precision, so the *order* in which CUDA
        atomics happen to combine the terms no longer changes the result at any
        precision we care about -> if this removes the run-to-run spread, the
        spread was caused by float32 add-order in scatter_add_.
        """
        assert reduce == "sum"
        # broadcast index against src, mirroring mace.tools.scatter._broadcast
        idx = index
        if dim < 0:
            dim = src.dim() + dim
        if idx.dim() == 1:
            for _ in range(0, dim):
                idx = idx.unsqueeze(0)
        for _ in range(idx.dim(), src.dim()):
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src)
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif idx.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(idx.max()) + 1
        acc = torch.zeros(size, dtype=torch.float64, device=src.device)
        # scatter_add_ (not index_add_) because mace broadcasts the index to the
        # full src shape, giving a 2-D index for dim=0
        acc.scatter_add_(dim, idx, src.to(torch.float64))
        if out is not None:
            out.copy_(acc.to(src.dtype))
            return out
        return acc.to(src.dtype)

    _mm.scatter_sum = scatter_sum_f64
    _mb.scatter_sum = scatter_sum_f64
    _msc.scatter_sum = scatter_sum_f64
    patch_info = {
        "patched": [
            "mace.modules.models.scatter_sum",
            "mace.modules.blocks.scatter_sum",
            "mace.tools.scatter.scatter_sum",
        ],
        "impl": "index_add_ in float64, cast back to src.dtype",
        "file": "mace/tools/scatter.py:46-48",
    }

import mace.tools.scatter as msc  # noqa: E402
from ase.io import read  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

print("=" * 100)
print(f"CONFIG = {CONFIG}   models={MODEL_KEYS}   N_rep={NREP}   device={DEVICE} dtype={DTYPE}")
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"device {torch.cuda.get_device_name(0)}")
print(f"allow_tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
      f"cudnn={torch.backends.cudnn.allow_tf32} "
      f"| matmul_precision={torch.get_float32_matmul_precision()} "
      f"| deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}")
print(f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_ENV}")
print(f"scatter_sum module in use: {msc.scatter_sum.__module__}."
      f"{getattr(msc.scatter_sum, '__name__', '?')}"
      f"{'  <-- PATCHED' if patch_info else ''}")
print("=" * 100)

atoms = {k: read(v) for k, v in CELLS.items()}
for k, a in atoms.items():
    assert len(a) == NAT[k], (k, len(a), NAT[k])


def spread(x):
    """distribution summary of a 1-D array of repeated scalars"""
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "std": float(x.std(ddof=0)),
    }


def prep_batch(calc, ats):
    """exactly what MACECalculator.calculate does to build the model input"""
    b = calc._atoms_to_batch(ats)
    md = next(calc.models[0].parameters()).dtype
    for key in b.keys:
        v = b[key]
        if torch.is_tensor(v) and torch.is_floating_point(v):
            b[key] = v.to(dtype=md)
    return b


def forward_once(calc, batch):
    """one model evaluation on a prepared batch -> (E, node_energy, forces)"""
    bd = calc._clone_batch(batch).to_dict()
    out = calc.models[0](bd, training=False, compute_force=True,
                         compute_stress=False, compute_edge_forces=False,
                         compute_atomic_stresses=False)
    return (float(out["energy"].detach().cpu()),
            out["node_energy"].detach().cpu().numpy().astype(np.float64),
            out["forces"].detach().cpu().numpy().astype(np.float64))


def run_repeats(model_key, cell, nrep, reuse_batch=False):
    """evaluate the same structure nrep times in one process, same calculator.

    node_energy returned by the model is the per-atom energy vector (e0
    included); its sum is the total energy, so we can see whether the spread
    lives in the vector itself or only in the reduction.
    """
    calc = MACECalculator(model_paths=[MODELS[model_key]], device=DEVICE,
                          default_dtype=DTYPE)
    ats = atoms[cell]
    fixed = prep_batch(calc, ats) if reuse_batch else None

    E, Evec, F = [], [], []
    for _ in range(nrep):
        batch = fixed if fixed is not None else prep_batch(calc, ats)
        e, evec, f = forward_once(calc, batch)
        E.append(e)
        Evec.append(evec)
        F.append(f)
        del batch
    return np.array(E), np.array(Evec), np.array(F), np.array([])


def run_repeats_calc(model_key, cell, nrep):
    """same protocol as the reference measurement: ASE MACECalculator.calculate"""
    calc = MACECalculator(model_paths=[MODELS[model_key]], device=DEVICE,
                          default_dtype=DTYPE)
    ats = atoms[cell]
    E, Evec, F = [], [], []
    for _ in range(nrep):
        calc.results = {}
        calc.calculate(ats, properties=["energy", "forces"])
        E.append(float(calc.results["energy"]))
        F.append(np.array(calc.results["forces"], dtype=np.float64).copy())
        if "energies" in calc.results:
            Evec.append(np.array(calc.results["energies"], dtype=np.float64).copy())
    return np.array(E), np.array(Evec), np.array(F), np.array([])


report = {
    "config": CONFIG,
    "nrep": NREP,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "matmul_precision": torch.get_float32_matmul_precision(),
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cublas_workspace_config": CUBLAS_ENV,
    "patch": patch_info,
    "cells": {},
}

t0 = time.time()
for mkey in MODEL_KEYS:
    report["cells"][mkey] = {}
    variants = [("freshbatch", run_repeats,
                 dict(reuse_batch=False)),
                ("samebatch", run_repeats,
                 dict(reuse_batch=True))]
    if CONFIG == "base":
        variants.append(("ase_calculator", run_repeats_calc, {}))
    for tag, fn_run, kw in variants:
        E = {}
        per_atom_vec = {}
        forces = {}
        net_atom_vec = {}
        for cell in ("alpha", "gamma"):
            e, evec, f, net = fn_run(mkey, cell, NREP, **kw)
            E[cell] = e
            if evec.size:
                per_atom_vec[cell] = evec
            if net.size:
                net_atom_vec[cell] = net
            forces[cell] = f
        dE = (E["gamma"] / NAT["gamma"] - E["alpha"] / NAT["alpha"]) * 1000.0

        entry = {
            "n_repeats": NREP,
            "E_alpha_per_atom_eV": spread(E["alpha"] / NAT["alpha"]),
            "E_gamma_per_atom_eV": spread(E["gamma"] / NAT["gamma"]),
            "dE_meV_per_atom": spread(dE),
            "E_alpha_total_eV": spread(E["alpha"]),
            "E_gamma_total_eV": spread(E["gamma"]),
            "dE_values_meV_per_atom": [float(v) for v in dE],
        }
        # per-atom energy vector: max deviation of any atom component vs repeat 0
        for cell in ("alpha", "gamma"):
            if cell in per_atom_vec:
                v = per_atom_vec[cell]           # [nrep, natoms]
                dev = np.abs(v - v[0])
                entry[f"peratom_energy_vec_{cell}"] = {
                    "max_abs_dev_meV": float(dev.max() * 1000.0),
                    "rms_dev_meV": float(np.sqrt((dev ** 2).mean()) * 1000.0),
                    "spread_of_sum_meV": float(
                        (v.sum(axis=1).max() - v.sum(axis=1).min()) * 1000.0),
                }
            if cell in net_atom_vec:
                v = net_atom_vec[cell]
                dev = np.abs(v - v[0])
                entry[f"net_peratom_energy_vec_{cell}"] = {
                    "max_abs_dev_meV": float(dev.max() * 1000.0),
                    "rms_dev_meV": float(np.sqrt((dev ** 2).mean()) * 1000.0),
                    "spread_of_sum_meV": float(
                        (v.sum(axis=1).max() - v.sum(axis=1).min()) * 1000.0),
                }
            # forces
            f = forces[cell]                     # [nrep, natoms, 3]
            sd = f.std(axis=0)                   # per component std over repeats
            entry[f"forces_{cell}"] = {
                "max_component_std_meV_per_A": float(sd.max() * 1000.0),
                "rms_component_std_meV_per_A": float(np.sqrt((sd ** 2).mean()) * 1000.0),
                "max_pairwise_rmse_vs_rep0_meV_per_A": float(
                    np.sqrt(((f - f[0]) ** 2).mean()) * 1000.0),
            }
        report["cells"][mkey][tag] = entry

        print(f"\n--- {mkey} / {tag} ---")
        for cell in ("alpha", "gamma"):
            s = entry[f"E_{cell}_per_atom_eV"]
            print(f"  E_{cell}/atom  min={s['min']:.9f} max={s['max']:.9f} "
                  f"range={s['range']*1000:.4f} meV  std={s['std']*1000:.4f} meV")
        s = entry["dE_meV_per_atom"]
        print(f"  dE meV/atom    min={s['min']:.4f} max={s['max']:.4f} "
              f"range={s['range']:.4f} std={s['std']:.4f}  "
              f"{'REPRODUCIBLE' if s['range'] == 0.0 else 'NONDETERMINISTIC'}")
        print(f"  dE values: {[round(v,4) for v in entry['dE_values_meV_per_atom']]}")
        for cell in ("alpha", "gamma"):
            if f"peratom_energy_vec_{cell}" in entry:
                pv = entry[f"peratom_energy_vec_{cell}"]
                print(f"  per-atom E vec {cell}: max_dev={pv['max_abs_dev_meV']:.6f} meV "
                      f"rms_dev={pv['rms_dev_meV']:.6f} meV "
                      f"sum_spread={pv['spread_of_sum_meV']:.4f} meV")
            if f"net_peratom_energy_vec_{cell}" in entry:
                pv = entry[f"net_peratom_energy_vec_{cell}"]
                print(f"  NET per-atom vec {cell}: max_dev={pv['max_abs_dev_meV']:.6f} meV "
                      f"rms_dev={pv['rms_dev_meV']:.6f} meV "
                      f"sum_spread={pv['spread_of_sum_meV']:.4f} meV")
            ff = entry[f"forces_{cell}"]
            print(f"  forces {cell}: max_comp_std={ff['max_component_std_meV_per_A']:.4f} "
                  f"rms_comp_std={ff['rms_component_std_meV_per_A']:.6f} "
                  f"rmse_vs_rep0={ff['max_pairwise_rmse_vs_rep0_meV_per_A']:.4f} meV/A")

report["elapsed_s"] = time.time() - t0
fn = f"{OUT}/json/task3_nondeterminism_{CONFIG}.json"
with open(fn, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote {fn}   ({report['elapsed_s']:.1f} s)")
