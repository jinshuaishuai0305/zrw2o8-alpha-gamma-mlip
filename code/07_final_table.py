#!/usr/bin/env python3
"""Final Task-1 table: E_alpha/atom, E_gamma/atom, dE per model and dtype."""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from ase.io import read
from mace.calculators import MACECalculator
M = {"v1":"/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v1_stagetwo.model",
     "v2":"/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v2_stagetwo.model",
     "v3b":"/home/jss/share/AlZrW2O8_MACE/AlZrW_v3b_stagetwo.model",
     "v5b":"/home/jss/share/AlZrW2O8_MACE/AlZrW_v5b_stagetwo.model",
     "v6":"/home/jss/share/AlZrW2O8_MACE/AlZrW_v6_stagetwo.model"}
cells = {k: read("/home/jss/share/AlZrW2O8_MACE/elastic/"+f)
         for k,f in (("a","_alpha_lmps.vasp"),("g","_gamma_lmps.vasp"))}
N = {"a":44,"g":132}
out={}
hdr=f"{'model':5s} {'dtype':11s} {'E_alpha/at (eV)':>20s} {'E_gamma/at (eV)':>20s} {'dE (meV/at)':>13s} {'spread':>9s}"
print(hdr); print("-"*len(hdr))
for name,p in M.items():
    out[name]={}
    for dtype,rep in (("float32",6),("float64",1)):
        A,G=[],[]
        for _ in range(rep):
            c=MACECalculator(model_paths=[p],device="cuda",default_dtype=dtype)
            e={}
            for k,at in cells.items():
                at.calc=c; e[k]=float(at.get_potential_energy())/N[k]
            A.append(e['a']); G.append(e['g']); del c
        dEs=[(g-a)*1000 for a,g in zip(A,G)]
        out[name][dtype]={"E_alpha_per_atom_mean":float(np.mean(A)),"E_gamma_per_atom_mean":float(np.mean(G)),
                          "E_alpha_spread_meV":float((max(A)-min(A))*1000),"E_gamma_spread_meV":float((max(G)-min(G))*1000),
                          "dE_mean":float(np.mean(dEs)),"dE_min":float(min(dEs)),"dE_max":float(max(dEs)),
                          "dE_spread":float(max(dEs)-min(dEs)),"n_repeat":rep,"all_dE":dEs}
        o=out[name][dtype]
        print(f"{name:5s} {dtype:11s} {o['E_alpha_per_atom_mean']:20.9f} {o['E_gamma_per_atom_mean']:20.9f} "
              f"{o['dE_mean']:13.4f} {o['dE_spread']:9.4f}")
json.dump(out,open("json/task1_final_table.json","w"),indent=2)
print("\nwrote json/task1_final_table.json")
