# `code/` — what each script is, and what it can and cannot reproduce

Two groups live here:

- **V11 scripts** (listed in §1 and §2) produced the numbers and figures in this repository.
- **V6-era scripts** (§3) are kept for lineage. They reproduce nothing in the V11 package.

Section titles below are those of `manuscript/jcp_v2/main.tex`.

---

## 1. Figure scripts (V11 data)

Run from anywhere; input is resolved as `<script dir>/../data/json`, output is written to
`<script dir>/../figures`. Python 3 + numpy + matplotlib (3.10.8 was used for the shipped PDFs).

| script | figure | reads | manuscript location |
|---|---|---|---|
| `make_fig1.py` | `figures/fig1.pdf` | `data/json/eos.json` (`phases.*.table/.BM`, `delta`), `pt_line.json` (`consistency`) | Fig. 1; §"Energy decomposition at zero temperature", §"The response coefficient as a design constraint" |
| `make_fig2.py` | `figures/fig2.pdf` | `data/json/phonons_all.json` (`phases.*.vol_*/fmin_THz`, `n_imag`) | Fig. 2; §"Convergence of the reported energy differences" |
| `make_fig3.py` | `figures/fig3.pdf` | `data/json/zpe_phdos.json`, `data/json/zpe_mesh_check.json` | Fig. 3; §"Origin of the zero-point difference: not the soft modes" |
| `make_fig4.py` | `figures/fig4.pdf` | `data/json/pt_line.json` (`rows`), `data/json/cte.json` | Fig. 4; §"Coexistence line and its range of validity" |
| `make_fig5.py` | `figures/fig5.pdf` | `data/json/pt_line.json` (`rows`), `data/json/eos.json` (`delta`), `data/json/zpe_phdos.json` (`delta`) | Fig. 5; §"Tripartite decomposition", §"Reference protocol and its cost" |

Only the two path constants differ from the manuscript originals in
`manuscript/jcp_v2/figures/make_fig{1..5}.py`; the plotting code is identical.

## 2. Analysis scripts (V11 data)

These are the scripts that produced `data/json/`. They are copied **unmodified** from the
compute tree (`ZrW2O8-beta/qha_v11/`, `ZrW2O8-beta/ref_audit/`), so they still expect that
tree's layout; the paths below are what each one needs to run. `stage_eos.py`,
`phonon_pipeline.py` and `stage_elastic.py` take the potential from the `MACE_MODEL`
environment variable (default: the cluster path recorded in the file) — point it at
`model/ZrW_v11_seed3_stagetwo.model` for a local rerun.

| script | produces | inputs it expects | manuscript location |
|---|---|---|---|
| `stage_eos.py` | `eos_raw.json` → `eos.json` | a `cluster_qha_v11/eos/<phase>/<tag>/poscar.json` volume grid; MACE + ASE (FIRE, ion-only relaxation) | Methods §"Quasi-harmonic free energies"; §"Equilibrium volumes and the collapse of the framework" |
| `phonon_pipeline.py` | `phonons_all.json` | `eos/<phase>/<tag>/poscar.json`, then phonopy + MACE; three stages `gen` / `forces` / `post` | Methods §"Quasi-harmonic free energies"; §"Convergence of the reported energy differences" |
| `qha_analysis.py` | `eos.json`, `thermal.json`, `zpe_phdos.json`, `pt_line.json`, `cte.json`, `elastic.json`, `SUMMARY.md` | `--raw` directory holding `eos_raw.json`, `phonons_all.json`, `elastic_raw.json` | Methods §"Quasi-harmonic free energies"; §"Coexistence line and its range of validity" |
| `stage_elastic.py` | `elastic_raw.json` → `elastic.json` | `QHA_ROOT=<cluster_qha_v11>` with the relaxed `vol_1.0000` cells; stress–strain at ±0.005 | §"Elastic response" |
| `extract.py` | console summaries, consumed when building `abacus_results.json` | ABACUS `OUT.*/running_*.log` directories; usage `extract.py <rundir> ...` | Methods §"Reference data" |
| `recompute_deltaE.py` | re-derives every number of `abacus_results.json` from the verbatim `FINAL_ETOT_IS` strings | `abacus_results.json` plus the ABACUS run logs it names | §"Reference protocol and its cost"; §"What the apparent agreement does and does not mean" |
| `phase_identity.py` | `phase_identity.json` (space group, W–O coordination, max displacement) | ABACUS `.Stru` files; usage `phase_identity.py <label> <unrelaxed> <relaxed> ...` | §"Is the reference shape relaxed?" |
| `audit.py` | `audit_results.json` | `TRAIN` and `OUT` path constants at the top of the file point **outside** this repository (the project training dump and the compute tree). Set them to your own copies before running | §"Tripartite decomposition"; §"Reference protocol and its cost" |

## 3. V6-era scripts — lineage only, not reproducible

Everything else in this directory (`01_state_dict.py` … `55_zpe_prec_cluster.py`,
`scan_xyz.py`, the six `*.sbatch` files, 49 files at this level) and the seven scripts in
`code/superseded_v6/` — the five V6 figure scripts plus `04_ground_truth.py` and
`05_exported_artifact.py`, which hard-code the −4.891 meV/atom shift and were moved here so
that no V11 script directory contains it — belong to the V6 generation.

They are kept so that the V6 claims can be traced, but **they cannot be used to reproduce any
number in this paper**:

- they were written against the V6 potential (`model/superseded_v6/AlZrW_v6_stagetwo.model`),
  whose training set contained 277 Al-containing frames and whose γ labels were shifted by
  −4.891 meV/atom;
- the V6 figure scripts read from `/home/jss/share/AlZrW2O8_MACE/diag_precision/json/`, outside
  this repository, and consume the V6 JSON names now in `data/superseded_v6_json/`;
- their outputs are the files in `data/superseded_v6_json/` and `figures/superseded_v6/`.

`CHANGELOG.md` maps each V6 path to its V11 counterpart.

## 4. Requirements

Python 3 with `numpy`, `scipy`, `matplotlib`, `ase`, `phonopy`, `spglib`; MACE (`mace` 0.3.x,
`torch`) for anything that evaluates the potential.
