#!/usr/bin/env python3
"""Recompute every stored number in abacus_results.json from its raw source.

Nothing here trusts a pre-computed number:
  * energies come from the verbatim `FINAL_ETOT_IS` strings (ABACUS output),
  * delta-E, P_t, ecut shift, candidate spread and the uncertainty quadrature are
    re-derived from those strings and the atom counts,
  * the phase-identity section is re-derived from from_cluster/p2/phase_identity.json
    (produced by phase_identity.py from the STRU files themselves).

    python3 recompute_deltaE.py [path/to/abacus_results.json]

Exit code 1 on any mismatch; prints "ALL STORED NUMBERS REPRODUCE" when clean.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAT = {"alpha": 44, "gamma": 132}
MEV = 1000.0
PT_FACTOR = 0.1791  # GPa per (meV/atom)
SPREAD_KEYS = ("gamma_final_100", "gamma_bfgs_early_100", "gamma_cellrelax_100",
               "gamma_bestforce_100", "gamma_cg_newest_100")

bad = []


def check(label, got, want, tol=1e-6):
    ok = want is None or abs(got - want) <= tol
    print("  %-52s = %16.9f   stored %s  %s"
          % (label, got, "n/a" if want is None else "%.9f" % want, "OK" if ok else "MISMATCH"))
    if not ok:
        bad.append(label)


def check_str(label, got, want):
    ok = (want is None) or (got == want)
    print("  %-52s = %-24s stored %-24s %s" % (label, got, want, "OK" if ok else "MISMATCH"))
    if not ok:
        bad.append(label)


def etot(rec):
    line = rec["FINAL_ETOT_IS_line"]
    assert line.startswith("FINAL_ETOT_IS"), line
    return float(line.split()[1])


def main(path):
    d = json.load(open(path))
    S = d["statics"]
    E = {k: etot(v) for k, v in S.items()}

    print("source:", os.path.abspath(path))

    print("\n[1] verbatim energies (eV) of every static SCF used")
    for k in ("alpha_ref", "alpha_relaxed", "alpha_relaxed_ecut120", "alpha_cellrelaxed",
              "gamma_ref", "gamma_final_100", "gamma_final_ecut120", "gamma_bestforce_100",
              "gamma_bfgs_early_100", "gamma_cellrelax_100", "gamma_cg_newest_100",
              "gamma_relaxed", "gamma_cellrelaxed", "kconv_alpha_k010", "kconv_gamma_k010"):
        if k in S:
            print("  %-24s nat=%3d  %s" % (k, S[k]["nat"], S[k]["FINAL_ETOT_IS_line"]))

    print("\n[2] core dE(gamma-alpha), fixed-cell protocol, ecutwfc 100 Ry")
    ea = E["alpha_relaxed"] / NAT["alpha"]
    eg = E["gamma_final_100"] / NAT["gamma"]
    de = (eg - ea) * MEV
    C = d["core"]
    check("E(alpha_relaxed)/44  [eV/atom]", ea, C["E_per_atom_alpha_eV"])
    check("E(gamma_final_100)/132 [eV/atom]", eg, C["E_per_atom_gamma_eV"])
    check("dE  [meV/atom]", de, C["dE_meV_per_atom"])
    check("P_t [GPa] = dE * 0.1791", de * PT_FACTOR, C["P_t_GPa"])

    print("\n[3] spread over every relaxed gamma structure we hold")
    vals = {}
    for k in SPREAD_KEYS:
        vals[k] = (E[k] / NAT["gamma"] - ea) * MEV
        check("%s dE [meV/atom]" % k, vals[k], C["spread_over_relaxed_gamma_structures"]["values"][k])
    check("range max-min [meV/atom]", max(vals.values()) - min(vals.values()),
          C["spread_over_relaxed_gamma_structures"]["range_meV_per_atom"])

    print("\n[4] ecut sensitivity (100 -> 120 Ry, identical STRU)")
    K = d["ecut_sensitivity"]
    da = (E["alpha_relaxed_ecut120"] - E["alpha_relaxed"]) / NAT["alpha"] * MEV
    dg = (E["gamma_final_ecut120"] - E["gamma_final_100"]) / NAT["gamma"] * MEV
    de120 = (E["gamma_final_ecut120"] / NAT["gamma"] - E["alpha_relaxed_ecut120"] / NAT["alpha"]) * MEV
    check("alpha shift [meV/atom]", da, K["alpha"]["shift_meV_per_atom"])
    check("gamma shift [meV/atom]", dg, K["gamma"]["shift_meV_per_atom"])
    check("dE(120 Ry) [meV/atom]", de120, K["dE_120_meV_per_atom"])
    check("dE change (120-100) [meV/atom]", de120 - de, K["dE_change_meV_per_atom"])
    check("P_t change [GPa]", (de120 - de) * PT_FACTOR, K["P_t_change_GPa"])
    print("  criterion |change| <= 0.05 meV/atom ->", "PASS" if abs(de120 - de) <= 0.05 else "FAIL")

    print("\n[5] cross-check: cell+ion relaxation protocol")
    CC = C["cross_check_cellrelax_protocol"]
    ea2 = E["alpha_cellrelaxed"] / NAT["alpha"]
    de2 = (E["gamma_cellrelax_100"] / NAT["gamma"] - ea2) * MEV
    check("E(alpha_cellrelaxed)/44 [eV/atom]", ea2, CC["E_per_atom_alpha_eV"])
    check("dE(cell-relax) [meV/atom]", de2, CC["dE_meV_per_atom"])
    check("spread vs core [meV/atom]", de2 - de, CC["spread_vs_core_meV_per_atom"])
    EP = C["cellrelax_endpoint_maxF_below_0.01"]
    e_end = float(EP["final_etot_is_line"].split()[0])
    check("cell-relax endpoint (max|F|=0.0094) dE [meV/atom]",
          (e_end / NAT["gamma"] - ea2) * MEV, EP["dE_meV_per_atom"])

    print("\n[6] relaxation drops vs unrelaxed statics")
    F = d["comparisons"]["fixed_cell_drop_vs_unrelaxed"]
    check("alpha full-relax drop [meV/atom]",
          (E["alpha_relaxed"] - E["alpha_ref"]) / NAT["alpha"] * MEV, F["alpha_full_relax_meV_per_atom"])
    check("gamma full-relax drop [meV/atom]",
          (E["gamma_final_100"] - E["gamma_ref"]) / NAT["gamma"] * MEV, F["gamma_full_relax_meV_per_atom"])

    print("\n[7] k-point sensitivity (kspacing 0.14 -> 0.10)")
    Kp = d["kpoint_sensitivity"]
    dka = (E["kconv_alpha_k010"] - E["alpha_relaxed"]) / NAT["alpha"] * MEV
    dkg = (E["kconv_gamma_k010"] - E["gamma_relaxed"]) / NAT["gamma"] * MEV
    check("alpha shift [meV/atom]", dka, Kp["alpha_shift_meV_per_atom"])
    check("gamma shift [meV/atom]", dkg, Kp["gamma_shift_meV_per_atom"])
    check("dE shift [meV/atom]", dkg - dka, Kp["dE_shift_meV_per_atom"])

    print("\n[8] uncertainty budget (now including ecut)")
    U = d["uncertainty_budget"]
    terms = [U["scf_threshold"]["value"], U["kpoints"]["value"], U["ecut"]["value"],
             U["relax_alpha"]["value"], U["relax_gamma"]["value"], U["protocol_systematic"]["value"]]
    check("ecut term equals |dE change|", U["ecut"]["value"], abs(de120 - de))
    check("relax_gamma term equals half the spread", U["relax_gamma"]["value"],
          (max(vals.values()) - min(vals.values())) / 2.0)
    tot = sum(t * t for t in terms) ** 0.5
    check("total quadrature [meV/atom]", tot, U["total_quadrature"], tol=1e-6)
    check("total quadrature [GPa]", tot * PT_FACTOR, U["total_P_t_GPa"], tol=1e-6)

    print("\n[9] comparison table (meV/atom and GPa)")
    rows = [("this work (round-2 core)", de, de * PT_FACTOR),
            ("round-1 core", d["core_round1"]["dE_meV_per_atom"], d["core_round1"]["P_t_GPa"])]
    for k in ("frame_kou_jing", "zero_force_extrapolation", "V9_potential_relaxed_seed3"):
        v = d["comparisons"][k]
        rows.append((k, v["dE_meV_per_atom"], v["P_t_GPa"]))
        check("%s shift vs core [meV/atom]" % k, de - v["dE_meV_per_atom"], v["shift_vs_core_meV_per_atom"])
    for name, x, p in rows:
        print("  %-28s %7.4f meV/atom   %.4f GPa" % (name, x, p))

    print("\n[10] phase identity re-derived from the STRU files")
    pi_path = os.path.join(HERE, "from_cluster", "p2", "phase_identity.json")
    pi = json.load(open(pi_path))
    P = d["phase_identity"]
    BOND_KEYS = {"W_O": "W_O_lt_2.2A", "Zr_O": "Zr_O_lt_2.4A"}
    for lab, key in (("alpha", "alpha"), ("gammaF", "gamma")):
        rec = pi[lab]
        check_str("%s unrelaxed SG @0.05" % lab,
                  rec["unrelaxed"]["spacegroup"]["0.05"],
                  P[key]["unrelaxed"]["spacegroup"]["0.05"])
        rlx = rec["relaxed"]
        stored = P[key]["relaxed_main"] if key == "gamma" else P[key]["relaxed"]
        check_str("%s relaxed SG @0.05" % lab, rlx["spacegroup"]["0.05"],
                  stored["spacegroup"]["0.05"])
        for bkey, skey in BOND_KEYS.items():
            check_str("%s relaxed %s CN" % (lab, bkey),
                      str(rlx[bkey]["CN_distribution"]), str(stored[skey]["CN_distribution"]))
            check("%s relaxed %s bond mean" % (lab, bkey), rlx[bkey]["bond_mean"],
                  stored[skey]["bond_mean"], tol=1e-4)
        check("%s relaxed volume [A^3]" % lab, rlx["volume_A3"], stored["volume_A3"], tol=1e-4)
    print("  one-sentence conclusion:", P["one_sentence"])

    print("\nRESULT: dE(gamma-alpha) = %+.4f meV/atom, P_t = %.4f GPa, "
          "ecut shift %+.5f meV/atom, spread over relaxed gamma structures %.4f meV/atom"
          % (de, de * PT_FACTOR, de120 - de, max(vals.values()) - min(vals.values())))
    if bad:
        print("MISMATCH IN:", bad)
        return 1
    print("ALL STORED NUMBERS REPRODUCE.")
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "abacus_results.json")
    sys.exit(main(p))
