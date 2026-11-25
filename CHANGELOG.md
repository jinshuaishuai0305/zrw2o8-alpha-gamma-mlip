# CHANGELOG — V6 → V11 replacement, path by path

This file is the audit trail for the generation change. Every row gives the old path, the new
path, and the evidence for the replacement: which file, and which number, changed.

**One sentence on why the V6 generation was superseded:** the V6 potential was fine-tuned through
an intermediate Zr-W-O-Al model from a training set that contained **277 Al-containing frames**
(fcc Al, Al₂O₃, Al/ZrW₂O₈ interfaces) and whose **γ labels carried an energy shift of
−4.891 meV/atom**, so every V6 result inherits an artefact that has nothing to do with ZrW₂O₈.

**What V11 is instead:** pure-ZrW₂O₈ data (1539 train / 258 valid frames, α 991/171, γ 515/85,
β 33/2), no Al and **no energy shift at all** (`data/dataset_meta.json` :
`gamma_shift_applied_meV_per_atom = 0.0`), two-stage fine-tuning from MACE-MP-0 (small) at
`r_max = 6.0 Å` / 128 channels / 2 layers / 402,192 parameters.

Nothing was deleted. Every V6 file is byte-identical to what the previous commit published,
merely moved. Two V6 defects are recorded here for the record rather than silently repaired:

1. the old `README.md` described `data/train.xyz` as "1827 configurations including fcc Al,
   Al2O3 and Al/ZrW2O8 interfaces". The file actually contains **1763** frames, all of them
   Zr/W/O — no Al at all (counted from the `Lattice` lines and the species column of the file
   now at `data/raw_pbe_frames.xyz`). It was never the V6 training set either.
2. the old figure scripts did not read anything inside this repository: they pointed at
   `/home/jss/share/AlZrW2O8_MACE/diag_precision/json/` (`code/superseded_v6/make_fig*.py`,
   `ROOT` constant, line 9 or 20). They could not be run from a clone of the deposit.

---

## 1. Potential — `model/`

| old path | new path | evidence / number that changed |
|---|---|---|
| `model/AlZrW_v6_stagetwo.model` (md5 `6578fcfa950c3a472775e79556716704`) | `model/superseded_v6/AlZrW_v6_stagetwo.model` (same md5, moved) | replaced by V11: MACE-MP-0 small → stage one → stage two; `r_max 6.0` Å, 402,192 parameters, both verified by loading the file (`p.numel()` sum, `model.r_max`) |
| — (no V11 predecessor) | `model/ZrW_v11_seed3_stagetwo.model` (md5 `a53d16758ef2feece25749eeefd2334f`) | the potential that produces every number in `data/json/`; `pt_line.json:model` records this filename as the one used for the QHA run |
| — (no V11 predecessor) | `model/ZrW_v11_seed3.model` (md5 `43a09965f4da8373fcebabbbe4861b5d`) | stage-one optimum, added so the two-stage lineage is auditable; identical architecture and parameter count |

## 2. Training data — `data/`

| old path | new path | evidence / number that changed |
|---|---|---|
| `data/train.xyz` (md5 `de4b5c58537e9f28e834db3914bf5cad`, unchanged by the rename, 1763 frames, Zr/W/O only) | `data/raw_pbe_frames.xyz` (same md5 `de4b5c58537e9f28e834db3914bf5cad`, renamed only) | kept as provenance; it is **not** the training set — the V6 README's "1827 configurations including fcc Al" claim is false for this file (1763 frames, no Al) |
| — | `data/train_v9.xyz` (md5 `647178d1af9874ed4c15a20dc21bb0cb`) | the actual V11 training set: 1539 frames, zones α 991 / γ 515 / β 33 |
| — | `data/valid_v9.xyz` (md5 `822db89399e2948bd67bda5fda048436`) | 258 frames, zones α 171 / γ 85 / β 2; grouped split, zero shared DFT groups with the training file |
| — | `data/e0s_v9.json` (md5 `1ca73aaa4c98d5647d7136016110d54a`) | O −2284.122929, Zr −285.515366, W −571.030732 eV/atom |
| — | `data/dataset_meta.json` (md5 `b965b4fe3954b1b9b2bb640d3c9c33f3`) | counts, `zero_alignment_eV_per_atom = -1790.98299`, `delta_E_gamma_alpha_meV_per_atom = 7.43` (raw PBE, not the V6 value), `gamma_shift_applied_meV_per_atom = 0.0` |

## 3. Results — `data/json/` (46 V6 files → 11 V11 files)

The 46 V6 JSON files are at `data/superseded_v6_json/<same name>`, unchanged. Their V11
counterparts are in `data/json/`. Grouped by the quantity they carry:

| old paths | new path | evidence / number that changed |
|---|---|---|
| `pt_line_v6_raw.json`, `pt_line_dft_aligned.json` | `pt_line.json` | ΔE_st(γ−α) **1.9224 → 2.7064** meV/atom; P_t(0 K) **0.5087 → 0.6345** GPa. The V6 pair also needed a DFT-alignment shift of 5.5071 meV/atom to reach a second P_t of 1.4068 GPa; V11 has one line and no alignment (`pt_line.json:consistency`, `Pt0K_GPa`) |
| `ev_for_figures.json`, `qha_f32_result.json` | `eos.json` | V6 E(V) tables built from float32 runs; V11 uses float64, adds the Birch–Murnaghan fits (`phases.*.BM`), ΔV₀/V = −5.06 % and `delta.dEst_vs_DFT = +0.1864` |
| `phdos_zpe.json` | `zpe_phdos.json` | ΔZPE(γ−α) **+0.986 → +0.7907** meV/atom (V6 mesh 16³ for both phases: α 68.582, γ 69.568 meV/atom; V11 α 68.661 / γ 69.452 with per-phase converged meshes), plus PHDOS, band decomposition and the mesh-convention uncertainty |
| `thermal_compression.json`, `thermo_table.json`, `alpha_thermo_v6.json` | `thermal.json`, `cte.json` | V6 stored raw compression runs and a separate α thermo table; V11 stores one thermal table (`thermal.json`) and one CTE file per phase (`cte.json`), from which α_V(300 K) = −2.62×10⁻⁵ / +1.79×10⁻⁵ K⁻¹ and dV(0→300 K) = −0.548 % / +0.251 % are read |
| `elastic_v6_alpha_0K.json`, `elastic_v6_alpha_300K.json`, `elastic_v6_gamma_0K.json`, `elastic_v6_gamma_300K.json` | `elastic.json` | V6 kept four files (two phases × two temperatures, float32 chain); V11 is one 0 K float64 stress–strain run (`elastic.json:dtype = "float64"`, `delta = 0.005`): B_Hill 91.63 / 67.48 GPa, G_Hill 31.92 / 34.29 GPa |
| `gamma_333_phonons.json`, `gamma_shape_relax.json` | `phonons_all.json` | V6 had a single 3×3×3 γ phonon set plus a shape-relaxation study; V11 `phonons_all.json` holds the full volume series for both phases with per-volume `fmin_THz`, `n_imag` and the geometry gates that replaced the shape study |
| `task3_ground_truth.json` | `abacus_results.json` | V6's ground-truth file recorded only the two frame energies; V11 re-derives ΔE(γ−α) from verbatim ABACUS `FINAL_ETOT_IS` strings with a full uncertainty budget: 2.4608 ± 0.1427 meV/atom, relaxed-endpoint 2.5247, spread 2.4608–2.6638 |
| `task1_dE.json`, `task1_final_table.json`, `task3_impact.json`, `task2_rootcause.json`, `task2_state_dict.json` | `audit_results.json` | V6 diagnostics were split over five files; V11 has one audit (`partA`/`partB`/`partC`/`summary`) that traces the same protocol/functional decomposition of the ΔE gap. The V6 root-cause files are kept because they document the state-dict transplant that carried the shift |
| `task3_base_p{1,2,3}.json`, `task3_cpu32_p{1,2,3}.json`, `task3_cublas_p{1,2,3}.json`, `task3_det_p{1,2,3}.json`, `task3_f64_p{1,2,3}.json`, `task3_f64scatter_p{1,2,3}.json`, `task3_notf32_p{1,2,3}.json`, `task3_samebatch_p{1,2,3}.json` (24 files) | no 1:1 counterpart; superseded by `zpe_mesh_check.json` + `abacus_results.json:uncertainty_budget`. They measure one-off bit-reproducibility of the V6 inference path (backend, dtype, scatter order, batch composition). V11 replaces point measurements with the mesh/supercell convergence numbers in `zpe_mesh_check.json` (`delta.same_mesh_24_meV_at` = 0.7626, mesh contribution 0.0007 meV/atom) and the explicit per-contribution budget in `abacus_results.json` |
| `task4_export.json`, `eval_v6_on_v3b.json` | no counterpart; superseded | these evaluate the *exported V6 artefact* and the V6 potential on a legacy dataset, so no V11 number can stand in for them. Kept in `superseded_v6_json/` as evidence for the V6 defect |

## 4. Figures — `figures/`

| old path | new path | evidence |
|---|---|---|
| `figures/fig1.pdf` (md5 `2d2d66d785e842ed75c4c0639c07ce86`) | `figures/superseded_v6/fig1.pdf` | replaced by V11 redraw (518.4 × 183.6 pt): ΔV/V label −5.06 %, slope 0.1755 GPa/(meV/atom), V11 points 2.7064 and 3.4971 |
| `figures/fig2.pdf` (md5 `0cd25767fe5f7bff652c39dbb3139c31`) | `figures/superseded_v6/fig2.pdf` | replaced (518.4 × 194.4 pt): lowest-frequency curves from `phonons_all.json` for both phases instead of the single γ 3×3×3 set |
| `figures/fig3.pdf` (md5 `2f5ab9fbad002a68da15e5743c27544b`) | `figures/superseded_v6/fig3.pdf` | replaced (518.4 × 187.2 pt): band-resolved ΔZPE sums to +0.7626 meV/atom, versus V6's +0.986 |
| `figures/fig4.pdf` (md5 `b864fcc88774b01812cb760b33f95f7e`) | `figures/superseded_v6/fig4.pdf` | replaced (518.4 × 194.4 pt): one coexistence line (T ≤ 475 K) instead of the raw/DFT-aligned pair |
| `figures/fig5.pdf` (md5 `c0021a85d4009eeb9294d32a9fcbc863`) | `figures/superseded_v6/fig5.pdf` | replaced (518.4 × 198 pt): ledger with C = 0.1755, residual 0.1864, ΔZPE 0.7907, ΔG 3.4971 |

The new PDFs were produced by the scripts in `code/` from `data/json/` only. Regenerating them
reproduces the shipped bytes except for the PDF `CreationDate`/`ID` (5 to 7 differing bytes per
file, identical size and identical extracted text); `README.md` §4 records the stable sizes.

## 5. Code — `code/`

| old path | new path | evidence |
|---|---|---|
| `code/make_fig1.py` … `code/make_fig5.py` | `code/superseded_v6/make_fig1.py` … `make_fig5.py` | the V6 scripts carry `ROOT = '/home/jss/share/AlZrW2O8_MACE/'` (or `…/diag_precision/json/`), i.e. they read outside the repository, and they read the V6 JSON names (`ev_for_figures.json`, `pt_line_v6_raw.json`, `pt_line_dft_aligned.json`, `phdos_zpe.json`, `alpha_thermo_v6.json`, `thermal_compression.json`, `thermo_table.json`, `gamma_333_phonons.json`). The new `code/make_fig{1..5}.py` read `../data/json/` and write `../figures/` |
| — | `code/make_fig1.py` … `code/make_fig5.py` | V11 versions of the same five figures, adapted from the manuscript scripts (`manuscript/jcp_v2/figures/`): only the two path constants changed, all plotting code is identical |
| — | `code/stage_eos.py`, `code/phonon_pipeline.py`, `code/qha_analysis.py`, `code/stage_elastic.py` | the scripts that produced `eos.json`, `phonons_all.json`, `zpe_phdos.json`, `pt_line.json`, `cte.json`, `elastic.json`, `thermal.json`; copied unmodified from `ZrW2O8-beta/qha_v11/` (they expect a `cluster_qha_v11` work root, documented in `code/README.md`) |
| — | `code/audit.py`, `code/phase_identity.py`, `code/recompute_deltaE.py`, `code/extract.py` | the scripts behind `audit_results.json` and `abacus_results.json`; copied unmodified from `ZrW2O8-beta/ref_audit/` |
| `code/01_state_dict.py` … `code/55_zpe_prec_cluster.py`, `code/scan_xyz.py`, `code/*.sbatch` (49 files) | unchanged, same paths | V6-era scripts; kept for lineage only. `code/README.md` states they cannot reproduce any V11 number |
| `code/04_ground_truth.py`, `code/05_exported_artifact.py` | `code/superseded_v6/04_ground_truth.py`, `code/superseded_v6/05_exported_artifact.py` | both hard-code the removed shift (`LABEL = DFT - 4.891` at `05_exported_artifact.py:103`, the `-4.891 meV/atom gamma calibration` note at `04_ground_truth.py:10`). Moved so that the only place the number survives outside `superseded*` is documentation |
| — | `code/README.md` | new: maps each script to the paper section / JSON file it produces, and marks the V6-era group as non-reproducible |

## 6. Documentation

| old path | new path | evidence |
|---|---|---|
| `README.md` | `README.md` (rewritten) | the old text claimed a V6 potential and a "1827 configurations including fcc Al, Al2O3 and Al/ZrW2O8 interfaces" training set. Both are wrong for the current repository: the shipped data is 1539 + 258 pure-ZrW₂O₈ frames and the potential is V11 (402,192 parameters, `r_max` 6.0 Å) |
| — | `CHANGELOG.md` | this file |
| `CITATION.cff`, `LICENSE`, `LICENSE-DATA` | unchanged | licenses still apply to the new files (MIT for code, CC-BY-4.0 for data, potential and figures) |

## 7. Pre-submission sync with `manuscript/jcp_v2/` (this round)

The manuscript moved on after Sections 1-6 were written; this section records what was brought
back into line. Nothing above is withdrawn: the potential, the training data and the ten result
JSONs are unchanged.

| item | what changed | evidence |
|---|---|---|
| `code/make_fig1.py`, `make_fig3.py`, `make_fig4.py`, `make_fig5.py` | replaced by the current manuscript scripts (`manuscript/jcp_v2/figures/make_figN.py`); the only edit is the two path constants (`QHA`/`OUT` → `<script dir>/../data/json` and `<script dir>/../figures`) plus the provenance lines of the docstrings | all plotting code identical, so the `pdftotext` output of `figures/figN.pdf` and `manuscript/jcp_v2/figures/figN.pdf` is identical for N = 1, 2, 3, 4, 5 |
| experimental anchor, Fig. 1(b) and Fig. 5 | 1.173 meV/atom (0.210 GPa / 0.1791, the withdrawn hard-coded ratio) → **≤1.197 meV/atom = 0.210 GPa / 0.1755**, with 0.210 GPa stated as an upper bound on the equilibrium pressure | `pt_line.json` : `consistency.coef_variants.v11_fit_dV.coef_GPa_per_meV_atom` = 0.175469; 0.210 / 0.175469 = 1.1968 |
| Fig. 1(a) | now draws the third-order Birch–Murnaghan curve of each phase, with the three reconstructed γ volumes (V/V₀ ≥ 1.04) as open squares, excluded from fit and curve | `eos.json` : `phases.*.BM`, `phases.gamma.n_points_used` = 10 of 13, `phases.gamma.table[*].restructured` |
| Fig. 5(b) | seven-bar ledger → **waterfall** of five increments stacked from zero: 2.5247 → 2.7064 → 3.4971 → 3.6034 → 7.2207 meV/atom, experimental upper bound ≤1.197 as a dashed line, total discrepancy ≥6.024 as a double arrow, MLIP residual 0.1817 = 3.0 % of it | `pt_line.json` : `rows[0].dG_P0_eV_at` (3.6034), `rows[300].dG_P0_eV_at` (7.2207), `consistency.sum_meV_at` (3.4971); `eos.json` : `delta.dEst_meV_per_atom` (2.7064); `zpe_phdos.json` : `delta.dZPE_meV_at` (0.7907) |
| Fig. 3 | band-group tick labels now use en dashes (`0–4`, `4–16`, `>16` THz) | `code/make_fig3.py` |
| Fig. 4 | panel (b) framed to −8.2…+6.2 ×10⁻⁵ K⁻¹, and both curves truncated at the QHA validity limit T ≤ 475 K | `pt_line.json` : `Pt_valid_T_max_K` = 475.0 |
| — | `data/json/precision_check.json` added: the eleventh result file, and the only one with no V6 counterpart | float32 7.2798 / float64 7.8156 meV/atom, 4 repeats each (spread 0.0), systematic offset −0.5358 |
| `README.md` | the Fig. 1/3/4/5 descriptions, a new §2.1 on the 300 K ledger, the reproducibility note (which interpreter carries matplotlib) and the file inventory are brought to the state above; the sentence pointing at a TODO marker in the manuscript caption is deleted — there is no such marker any more | `README.md` §2, §2.1, §3, §4, §5 |
| `figures/fig1.pdf`, `fig3.pdf`, `fig4.pdf`, `fig5.pdf` | regenerated with matplotlib 3.10.8 (`/home/jss/miniconda3/bin/python`); `fig2.pdf` untouched, its input `data/json/phonons_all.json` is unchanged | sizes 43119 / 37640 / 33824 / 38153 bytes, page sizes as before, `pdftotext` text identical to the manuscript PDFs |
