#!/usr/bin/env python3
"""
Task 4 -- deployed/exported artifact.

No *-mliap_lammps.pt or any other TorchScript export exists under
/home/jss/share/AlZrW2O8_MACE/ or /home/jss/work/ for the AlZrW models (the
convert_v3b.sbatch / interface/convert_more.sbatch jobs ran on the cluster and
the .pt files were never copied back).  So we build the export locally here, in
diag_precision/ only, from an untouched COPY of the v6 model, and evaluate it.

Run:  /home/jss/miniconda3/envs/mace/bin/python 05_exported_artifact.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from ase.io import read

OUT = "/home/jss/share/AlZrW2O8_MACE/diag_precision"
os.makedirs(f"{OUT}/json", exist_ok=True)

PT = f"{OUT}/v6_copy.model-lammps.pt"
CELLS = {
    "alpha": "/home/jss/share/AlZrW2O8_MACE/elastic/_alpha_lmps.vasp",
    "gamma": "/home/jss/share/AlZrW2O8_MACE/elastic/_gamma_lmps.vasp",
}
N = {"alpha": 44, "gamma": 132}
DV = 18.040711333333333 - 17.146112878787878   # A^3/atom, measured from the cells
CONV = 160.2177                                 # GPa per eV/A^3

res = {}

# ---- dtype of the exported TorchScript --------------------------------
m = torch.jit.load(PT, map_location="cpu")
pd = sorted({str(x.dtype) for x in m.parameters()})
bd = sorted({str(x.dtype) for x in m.buffers() if x.is_floating_point()})
res["export_dtype"] = {"param_dtypes": pd, "float_buffer_dtypes": bd,
                       "n_params": len(list(m.parameters())),
                       "n_buffers": len(list(m.buffers()))}
print(f"exported TorchScript {os.path.basename(PT)}")
print(f"  parameter dtypes      : {pd}")
print(f"  float buffer dtypes   : {bd}")
print(f"  -> the export runs in {pd[0]}")
del m

# ---- evaluate it: drive the TorchScript with MACE's own input pipeline ----
# The exported object is a LAMMPS_MACE wrapper whose forward takes the same
# `data` dict MACE builds internally, so we capture that dict from a normal
# float64 MACECalculator run and feed it to the jit module.
import torch.jit  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

jit_model = torch.jit.load(PT, map_location="cuda")
orig_path = "/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model"

E = {}
E_ref = {}
for cname, path in CELLS.items():
    at = read(path)
    calc = MACECalculator(model_paths=[orig_path], device="cuda", default_dtype="float64")
    box = {}

    def pre_hook(_mod, args, kwargs=None, box=box):
        if args:
            box["data"] = args[0]

    h = calc.models[0].register_forward_pre_hook(pre_hook)
    at.calc = calc
    E_ref[cname] = float(at.get_potential_energy())
    h.remove()

    data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in box["data"].items()}
    data["positions"] = data["positions"].clone().detach().requires_grad_(True)
    local_or_ghost = torch.ones(len(at), dtype=torch.bool, device="cuda")
    out = jit_model(data, local_or_ghost, False)
    E[cname] = float(out["total_energy_local"].to(torch.float64).sum().item())
    print(f"  {cname}: {len(at)} atoms  jit E = {E[cname]:.9f} eV "
          f"= {E[cname]/len(at):.12f} eV/atom   "
          f"(float64 MACECalculator: {E_ref[cname]:.9f} eV, "
          f"diff = {(E[cname]-E_ref[cname])*1000:+.6f} meV)")

res["E_alpha_eV_via_MACECalculator_f64"] = E_ref["alpha"]
res["E_gamma_eV_via_MACECalculator_f64"] = E_ref["gamma"]

dE = (E["gamma"] / N["gamma"] - E["alpha"] / N["alpha"]) * 1000.0
P = (dE * 1e-3) / DV * CONV
res["E_alpha_eV"] = E["alpha"]
res["E_gamma_eV"] = E["gamma"]
res["dE_meV_per_atom"] = dE
res["dV_A3_per_atom"] = DV
res["P_GPa"] = P

print(f"\n  dE(gamma-alpha) = {dE:+.4f} meV/atom")
print(f"  dV              = {DV:.6f} A^3/atom")
print(f"  P_t             = {P:.4f} GPa")

# ---- reference numbers for the verdict --------------------------------
DFT = 7.4295
LABEL = DFT - 4.891
res["reference"] = {
    "dE_DFT_meV_per_atom": DFT,
    "dE_label_calibrated_meV_per_atom": LABEL,
    "P_DFT_GPa": DFT * 1e-3 / DV * CONV,
    "P_label_GPa": LABEL * 1e-3 / DV * CONV,
    "dE_v6_float64_meV_per_atom": 7.9633,
    "P_v6_float64_GPa": 7.9633 * 1e-3 / DV * CONV,
}
print(f"\n  reference: dE_DFT={DFT:.4f} -> P={res['reference']['P_DFT_GPa']:.4f} GPa")
print(f"             dE_label={LABEL:.4f} -> P={res['reference']['P_label_GPa']:.4f} GPa")
print(f"             dE_v6(f64)={res['reference']['dE_v6_float64_meV_per_atom']:.4f} "
      f"-> P={res['reference']['P_v6_float64_GPa']:.4f} GPa")

with open(f"{OUT}/json/task4_export.json", "w") as fh:
    json.dump(res, fh, indent=2)
print("\nwrote json/task4_export.json")
