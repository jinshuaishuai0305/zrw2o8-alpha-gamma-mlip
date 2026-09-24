#!/usr/bin/env python3
"""One fresh-process evaluation of dE for a given model+dtype (reproducibility)."""
import sys, warnings, os, torch
warnings.filterwarnings("ignore")
from ase.io import read
from mace.calculators import MACECalculator
M = {"v1":"/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v1_stagetwo.model",
     "v2":"/home/jss/share/AlZrW2O8_MACE/mace_train/AlZrW_v2_stagetwo.model",
     "v3b":"/home/jss/share/AlZrW2O8_MACE/AlZrW_v3b_stagetwo.model",
     "v5b":"/home/jss/share/AlZrW2O8_MACE/AlZr_v5b_stagetwo.model".replace("AlZr_","AlZrW_"),
     "v6":"/home/jss/share/AlZrW2O8_MACE/AlZr_v6_stagetwo.model".replace("AlZr_","AlZrW_")}
name, dtype, dev = sys.argv[1], sys.argv[2], sys.argv[3]
calc = MACECalculator(model_paths=[M[name]], device=dev, default_dtype=dtype)
E = {}
for k, f, n in (("a","_alpha_lmps.vasp",44), ("g","_gamma_lmps.vasp",132)):
    at = read("/home/jss/share/AlZrW2O8_MACE/elastic/"+f); at.calc = calc
    E[k] = float(at.get_potential_energy())/n
print("%s %s %s dE=%.4f meV/at  E_a=%.9f E_g=%.9f" %
      (name, dtype, dev, (E['g']-E['a'])*1000, E['a'], E['g']), flush=True)
