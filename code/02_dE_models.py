#!/usr/bin/env python3
"""
Task 1 + Task 2 root-cause decomposition.

For all five models x {float32, float64}:
  * E_alpha/atom, E_gamma/atom, dE = E_gamma/132 - E_alpha/44
  * decompose E_total = E0_sum + E_net, where E0_sum is the actual atomic
    reference energy tensor consumed by the model (captured with a forward
    hook on atomic_energies_fn) and E_net = E_total - E0_sum.
  This splits the float32-vs-float64 shift into an E0-accumulation part and a
  network part.

Also repeats the float32 run on CPU to test whether GPU float32 arithmetic
(TF32 / reduced-precision matmul) contributes.

Run:
  /home/jss/miniconda3/envs/mace/bin/python 02_dE_models.py
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

MODELS = {
    "v1": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v1_stagetwo.model",
    "v2": "/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v2_stagetwo.model",
    "v3b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v3b_stagetwo.model",
    "v5b": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v5b_stagetwo.model",
    "v6": "/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model",
}
CELLS = {
    "alpha": "/home/jss/share/AlZrW2O8_MACE/elastic/_alpha_lmps.vasp",
    "gamma": "/home/jss/share/AlZrW2O8_MACE/elastic/_gamma_lmps.vasp",
}

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("allow_tf32 matmul:", torch.backends.cuda.matmul.allow_tf32,
      "| cudnn:", torch.backends.cudnn.allow_tf32,
      "| fp32 precision:", torch.get_float32_matmul_precision())

atoms = {k: read(v) for k, v in CELLS.items()}
for k, a in atoms.items():
    from collections import Counter
    c = Counter(a.get_chemical_symbols())
    print(f"{k}: {len(a)} atoms {dict(c)}  V={a.get_volume():.4f} A^3  "
          f"V/atom={a.get_volume()/len(a):.6f}")


def evaluate(model_path, dtype, device, ats):
    """return (E_total, E0_sum, E_net) for one configuration"""
    calc = MACECalculator(model_paths=[model_path], device=device, default_dtype=dtype)
    m = calc.models[0]
    store = {}

    def hook(_mod, _inp, outp):
        store["e0"] = outp.detach().to(torch.float64).sum().item()

    h = m.atomic_energies_fn.register_forward_hook(hook)
    ats.calc = calc
    e_tot = float(ats.get_potential_energy())
    h.remove()
    e0 = store["e0"]
    return e_tot, e0, e_tot - e0


results = {}
for mname, mpath in MODELS.items():
    results[mname] = {}
    print("\n" + "=" * 78)
    print(f"### {mname}")
    for dtype in ("float32", "float64"):
        results[mname][dtype] = {}
        for cname, ats in atoms.items():
            e_tot, e0, e_net = evaluate(mpath, dtype, "cuda", ats)
            n = len(ats)
            results[mname][dtype][cname] = {
                "n_atoms": n,
                "E_total_eV": e_tot,
                "E_per_atom_eV": e_tot / n,
                "E0_sum_eV": e0,
                "E0_per_atom_eV": e0 / n,
                "E_net_eV": e_net,
                "E_net_per_atom_eV": e_net / n,
            }
    # extra: float32 on CPU, to test GPU float32 arithmetic
    for cname, ats in atoms.items():
        e_tot, e0, e_net = evaluate(mpath, "float32", "cpu", ats)
        results[mname].setdefault("float32_cpu", {})[cname] = {
            "n_atoms": len(ats), "E_total_eV": e_tot,
            "E_per_atom_eV": e_tot / len(ats), "E0_sum_eV": e0,
            "E0_per_atom_eV": e0 / len(ats),
            "E_net_eV": e_net, "E_net_per_atom_eV": e_net / len(ats),
        }

# ---------------- report ------------------------------------------------
hdr = (f"{'model':6s} {'dtype':16s} {'E_alpha/at':>16s} {'E_gamma/at':>16s} "
       f"{'dE meV/at':>12s} {'E0_a/at':>14s} {'E0_g/at':>14s} "
       f"{'dE_net meV/at':>14s}")
print("\n" + "=" * 110)
print("TASK 1: absolute energies and dE = E_gamma/132 - E_alpha/44")
print(hdr)
print("-" * 110)

summary = {}
for mname in MODELS:
    summary[mname] = {}
    for dtype in ("float32", "float64", "float32_cpu"):
        r = results[mname].get(dtype)
        if r is None:
            continue
        ea = r["alpha"]["E_per_atom_eV"]
        eg = r["gamma"]["E_per_atom_eV"]
        dE = (eg - ea) * 1000.0
        e0a = r["alpha"]["E0_per_atom_eV"]
        e0g = r["gamma"]["E0_per_atom_eV"]
        dna = r["alpha"]["E_net_per_atom_eV"]
        dng = r["gamma"]["E_net_per_atom_eV"]
        dEnet = (dng - dna) * 1000.0
        summary[mname][dtype] = {
            "E_alpha_per_atom_eV": ea, "E_gamma_per_atom_eV": eg,
            "dE_meV_per_atom": dE, "E0_alpha_per_atom_eV": e0a,
            "E0_gamma_per_atom_eV": e0g, "dE0_meV_per_atom": (e0g - e0a) * 1000.0,
            "dE_net_meV_per_atom": dEnet,
            "E_net_alpha_per_atom_eV": dna, "E_net_gamma_per_atom_eV": dng,
        }
        print(f"{mname:6s} {dtype:16s} {ea:16.9f} {eg:16.9f} {dE:12.4f} "
              f"{e0a:14.6f} {e0g:14.6f} {dEnet:14.4f}")

print("\n" + "=" * 110)
print("float64 - float32 difference per model")
print(f"{'model':6s} {'d(E_alpha/at) meV':>20s} {'d(E_gamma/at) meV':>20s} "
      f"{'d(dE) meV/at':>14s} {'d(E0_a/at) meV':>16s} {'d(E0_g/at) meV':>16s} "
      f"{'d(dE_net) meV/at':>18s}")
print("-" * 110)
for mname in MODELS:
    s = summary[mname]
    d_ea = (s["float64"]["E_alpha_per_atom_eV"] - s["float32"]["E_alpha_per_atom_eV"]) * 1000
    d_eg = (s["float64"]["E_gamma_per_atom_eV"] - s["float32"]["E_gamma_per_atom_eV"]) * 1000
    d_de = s["float64"]["dE_meV_per_atom"] - s["float32"]["dE_meV_per_atom"]
    d_e0a = (s["float64"]["E0_alpha_per_atom_eV"] - s["float32"]["E0_alpha_per_atom_eV"]) * 1000
    d_e0g = (s["float64"]["E0_gamma_per_atom_eV"] - s["float32"]["E0_gamma_per_atom_eV"]) * 1000
    d_denet = s["float64"]["dE_net_meV_per_atom"] - s["float32"]["dE_net_meV_per_atom"]
    summary[mname]["dtype_delta"] = {
        "dE_alpha_per_atom_meV": d_ea, "dE_gamma_per_atom_meV": d_eg,
        "d_dE_meV_per_atom": d_de, "d_E0_alpha_meV": d_e0a,
        "d_E0_gamma_meV": d_e0g, "d_dE_net_meV_per_atom": d_denet,
        "d_E_total_alpha_meV": d_ea * 44, "d_E_total_gamma_meV": d_eg * 132,
    }
    print(f"{mname:6s} {d_ea:20.4f} {d_eg:20.4f} {d_de:14.4f} {d_e0a:16.4f} "
          f"{d_e0g:16.4f} {d_denet:18.4f}")

with open(f"{OUT}/json/task1_dE.json", "w") as fh:
    json.dump({"raw": results, "summary": summary}, fh, indent=2)
print("\nwrote json/task1_dE.json")
