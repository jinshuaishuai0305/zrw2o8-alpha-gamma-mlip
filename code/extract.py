#!/usr/bin/env python3
"""Parse ABACUS running_*.log into a key=value summary.

Usage: extract.py <rundir> [<rundir> ...]
Prints one block per rundir: energy, ionic steps, last max|F| (eV/A),
last max|stress| (kbar and GPa), relaxation-converged flag.
"""
import glob
import math
import os
import re
import sys


def newest_log(d):
    fs = glob.glob(os.path.join(d, "OUT.*", "running_*.log"))
    return max(fs, key=os.path.getmtime) if fs else None


def last_energy(txt):
    e = None
    for m in re.finditer(r"FINAL_ETOT_IS\s+(-?\d+\.\d+)", txt):
        e = float(m.group(1))
    return e


def block_rows(txt, marker):
    """Numeric triples of the LAST <marker> block (a non-numeric line ends it)."""
    chunks = re.split(marker, txt)
    if len(chunks) < 2:
        return []
    rows = []
    for line in chunks[-1].splitlines():
        t = line.split()
        if t:
            try:
                rows.append([float(x) for x in t[-3:]])
                continue
            except ValueError:
                pass
        if rows:
            break
    return rows


def parse(d):
    p = newest_log(d)
    if p is None:
        return {"dir": d, "log": None}
    txt = open(p, errors="ignore").read()
    fr = block_rows(txt, r"\s*TOTAL-FORCE[^\n]*\n")
    st = block_rows(txt, r"\s*TOTAL-STRESS[^\n]*\n")
    out = {
        "dir": d,
        "log": p,
        "etot_eV": last_energy(txt),
        "n_ionic_steps": len(re.findall(r"TOTAL-FORCE", txt)),
        "maxF_eV_per_A": max((math.sqrt(sum(x * x for x in r)) for r in fr), default=None),
        "maxF_component_eV_per_A": max((abs(x) for r in fr for x in r), default=None),
        "max_stress_kbar": max((abs(x) for r in st for x in r), default=None),
        "relax_converged": ("Relaxation is converged!" in txt),
        "scf_converged": ("charge density convergence is achieved" in txt) or None,
        "nkstot": None,
    }
    m = re.search(r"nkstot\s*=\s*(\d+)", txt)
    if m:
        out["nkstot"] = int(m.group(1))
    if out["max_stress_kbar"] is not None:
        out["max_stress_GPa"] = out["max_stress_kbar"] / 10.0
    return out


for d in sys.argv[1:]:
    r = parse(d)
    print("#", r.get("log"))
    for k, v in r.items():
        if k == "log":
            continue
        print(f"{k}={v}")
