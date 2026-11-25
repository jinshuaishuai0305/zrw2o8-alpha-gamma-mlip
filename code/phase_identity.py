#!/usr/bin/env python3
"""Phase-identity check for ABACUS STRU files: α vs γ ZrW2O8.

For a (unrelaxed, relaxed) pair it reports
  * space group at three tolerances (spglib),
  * lattice constants and volume (and % change),
  * W–O nearest-neighbour bond lengths inside 2.2 Å, per-W coordination,
  * Zr–O coordination distribution,
  * max|Δpos| between the two structures (minimum image).

    python3 phase_identity.py <label> <unrelaxed.Stru> <relaxed.Stru> [...]
Exit code 0 always; numbers go to stdout as a small table per pair.
"""
import json
import os
import sys

import numpy as np
import spglib
from ase import Atoms
from ase.neighborlist import neighbor_list

BOHR2A = 0.529177210903

WO_CUT = 2.2
ZRO_CUT = 2.4


def read_stru(path):
    """ABACUS STRU -> ase.Atoms (LATTICE_CONSTANT in Bohr, fractional positions)."""
    lines = open(path).read().splitlines()
    up = [l.strip().upper() for l in lines]

    def idx(key):
        return next(i for i, l in enumerate(up) if l == key)

    lc = float(lines[idx("LATTICE_CONSTANT") + 1].split()[0]) * BOHR2A
    j = idx("LATTICE_VECTORS")
    lat = np.array([[float(x) for x in lines[j + 1 + k].split()[:3]] for k in range(3)]) * lc
    a = idx("ATOMIC_POSITIONS")
    mode = lines[a + 1].strip().split()[0].lower()
    sym, frac, i = [], [], a + 2
    while i < len(lines):
        t = lines[i].split()
        i += 1
        if not t:
            continue
        i += 1  # magnetism line
        n = int(lines[i].split()[0])
        i += 1
        sym.extend([t[0]] * n)
        for _ in range(n):
            c = [float(x) for x in lines[i].split("#")[0].split()[:3]]
            i += 1
            frac.append(c)
    frac = np.array(frac)
    if mode.startswith("dir"):
        pos = frac @ lat
    else:
        pos = frac * lc
    return Atoms(symbols=sym, positions=pos, cell=lat, pbc=True), np.array(frac), lat


def spacegroup(atoms, frac, lat, symprec):
    ds = spglib.get_symmetry_dataset((lat, frac, atoms.get_atomic_numbers()), symprec=symprec)
    return ds.international, ds.number, len(ds.rotations)


def coord_stats(atoms, center, ligand, cut):
    i, j, d = neighbor_list("ijd", atoms, cut)
    per = {}
    for ii, jj, dd in zip(i, j, d):
        if atoms[ii].symbol == center and atoms[jj].symbol == ligand:
            per.setdefault(ii, []).append(dd)
    cn = [len(v) for v in per.values()] or [0]
    bonds = [x for v in per.values() for x in v]
    return cn, bonds


def main(argv):
    pairs = [argv[k:k + 3] for k in range(0, len(argv), 3)]
    out = {}
    for label, f_unrel, f_rel in pairs:
        A, fracA, latA = read_stru(f_unrel)
        B, fracB, latB = read_stru(f_rel)
        assert len(A) == len(B), "atom count differs"

        volA = A.get_volume()
        volB = B.get_volume()
        d = B.positions - A.positions
        d -= np.round(d @ np.linalg.inv(latA)) @ latA
        dpos = np.linalg.norm(d, axis=1)

        rec = {"files": {"unrelaxed": os.path.abspath(f_unrel),
                         "relaxed": os.path.abspath(f_rel)},
               "natoms": len(A),
               "unrelaxed": {"a_b_c": np.linalg.norm(latA, axis=1).round(4).tolist(),
                             "volume_A3": round(volA, 4)},
               "relaxed": {"a_b_c": np.linalg.norm(latB, axis=1).round(4).tolist(),
                           "volume_A3": round(volB, 4)},
               "volume_change_pct": round(100 * (volB - volA) / volA, 4),
               "max_dpos_A": round(float(dpos.max()), 4),
               "mean_dpos_A": round(float(dpos.mean()), 4)}

        for name, atoms, frac, lat in (("unrelaxed", A, fracA, latA), ("relaxed", B, fracB, latB)):
            sg = {}
            for tol in (0.01, 0.05, 0.1):
                s, n, nop = spacegroup(atoms, frac, lat, tol)
                sg["%.2f" % tol] = "%s (No.%d, %d ops)" % (s, n, nop)
            rec[name]["spacegroup"] = sg
            cnw, bw = coord_stats(atoms, "W", "O", WO_CUT)
            cnz, bz = coord_stats(atoms, "Zr", "O", ZRO_CUT)
            rec[name]["W_O"] = {"cutoff_A": WO_CUT, "n_W": len(cnw),
                                "CN_distribution": {str(k): cnw.count(k) for k in sorted(set(cnw))},
                                "n_bonds": len(bw),
                                "bond_min": round(min(bw), 4), "bond_mean": round(float(np.mean(bw)), 4),
                                "bond_max": round(max(bw), 4), "bond_sd": round(float(np.std(bw)), 4)}
            rec[name]["Zr_O"] = {"cutoff_A": ZRO_CUT, "n_Zr": len(cnz),
                                 "CN_distribution": {str(k): cnz.count(k) for k in sorted(set(cnz))},
                                 "n_bonds": len(bz),
                                 "bond_min": round(min(bz), 4), "bond_mean": round(float(np.mean(bz)), 4),
                                 "bond_max": round(max(bz), 4), "bond_sd": round(float(np.std(bz)), 4)}
        out[label] = rec

        print("=" * 78)
        print("%s   natoms=%d" % (label, len(A)))
        print("  unrelaxed: a,b,c = %s   V = %.3f A^3" % (rec["unrelaxed"]["a_b_c"], volA))
        print("  relaxed  : a,b,c = %s   V = %.3f A^3  (dV = %+.4f %%)"
              % (rec["relaxed"]["a_b_c"], volB, rec["volume_change_pct"]))
        print("  max|dpos| = %.4f A   mean %.4f A" % (rec["max_dpos_A"], rec["mean_dpos_A"]))
        for name in ("unrelaxed", "relaxed"):
            r = rec[name]
            print("  [%s] space group:" % name)
            for tol, s in r["spacegroup"].items():
                print("      symprec=%s A -> %s" % (tol, s))
            for key in ("W_O", "Zr_O"):
                b = r[key]
                print("      %s (<%.1f A): CN dist %s | n_bonds=%d  bond %.3f-%.3f mean %.3f sd %.4f"
                      % (key, b["cutoff_A"], b["CN_distribution"], b["n_bonds"],
                         b["bond_min"], b["bond_max"], b["bond_mean"], b["bond_sd"]))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(argv[1])),
                                     "phase_identity.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
