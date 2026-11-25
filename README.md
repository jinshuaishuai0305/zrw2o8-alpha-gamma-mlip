# ZrW2O8 machine-learning potential and processed data — V11 generation

Companion repository for

> **Thermodynamic Error Amplification in Near-Degenerate Phase Transitions: Machine-Learning
> Potentials and the alpha/gamma Boundary of ZrW2O8**
> Shuaishuai Jin, Zhong Guan, Zhiyong You, Bing Li, Hang Li
> (submitted; journal reference to be added)

It contains the fitted interatomic potential (V11), the training/validation sets it was fitted
to, the processed results behind every number and figure of the paper, the five figures, and
the analysis scripts.

**This is the V11 generation.** The V6 material that this repository previously published is
still present, unmodified, under `model/superseded_v6/`, `data/superseded_v6_json/`,
`figures/superseded_v6/` and `code/superseded_v6/`. `CHANGELOG.md` maps every old path to its
V11 counterpart and gives the reason for the replacement. Nothing outside those `superseded*`
directories belongs to the V6 generation.

---

## 1. Definitions (what the V11 generation is)

**Potential.** MACE-MP-0 (`small`, foundation model `2023-12-03-mace-mp.model`) fine-tuned in
two stages on PBE data. Stage two, `model/ZrW_v11_seed3_stagetwo.model`, produces every number
in this repository; stage one, `model/ZrW_v11_seed3.model`, is included so the two-stage
lineage can be audited.

| property | value | how to check |
|---|---|---|
| cut-off radius | `r_max = 6.0 Å` | loaded attribute, `float(model.r_max)` = `6.0` (inherited from the MACE-MP-0 foundation model) |
| channels / layers | 128 channels, 2 interaction blocks | loaded model: `len(model.interactions)` = `2`, `interactions[0].hidden_irreps` = `128x0e` |
| parameters | **402,192** | `sum(p.numel() for p in model.parameters())` on either `.model` file |
| dtype used for results | float64 where stated in the result JSONs (`elastic.json:dtype`, `eos.json`) | — |

**Training data.** Pure ZrW2O8, no aluminium, **no energy shift of any kind**
(`dataset_meta.json:gamma_shift_applied_meV_per_atom = 0.0`). Split by DFT calculation group,
so no DFT frame appears in both splits.

| split | α | γ | β | total |
|---|---|---|---|---|
| train (`data/train_v9.xyz`) | 991 | 515 | 33 | **1539** |
| valid (`data/valid_v9.xyz`) | 171 | 85 | 2 | **258** |
| both | 1162 | 600 | 35 | 1797 |

Sources: `data/dataset_meta.json` (`n_train`, `n_valid`, `zones.alpha = 1162`,
`zones.gamma = 600`, `zones.beta = 35`) and the `zone=` tags of the two XYZ files.

Other label conventions, all from `data/dataset_meta.json`:

- `zero_alignment_eV_per_atom = -1790.98299`, applied uniformly to all phases (calibrated from
  seven identical-geometry cross-calculations, std 0.01 meV/atom); it cancels in every phase
  energy difference.
- `delta_E_gamma_alpha_meV_per_atom = 7.43` — the PBE target for ΔE(γ−α), i.e. the raw PBE
  number, not a shifted one.
- `gamma_shift_forbidden_legacy_meV_per_atom = -4.891` — **not applied to any frame.** This
  field records, for audit purposes only, the shift that the superseded V6 data set applied to
  the γ frames; `gamma_shift_applied_meV_per_atom = 0.0` is what the present set uses. The two
  fields are kept side by side so that a reader can see both the withdrawn value and its
  replacement without consulting any other document.
- elemental references in `data/e0s_v9.json`: O `-2284.122929`, Zr `-285.515366`,
  W `-571.030732` eV/atom.
- stress convention: `eV/Å³`, positive = tension, `stress = -virial/V`.

**Data that is deliberately *not* the training set.** `data/raw_pbe_frames.xyz` is the raw PBE
frame dump inherited from the V6 era: 1763 frames, elements Zr/W/O only. It is kept for
provenance (frame counts, group tags) and is **not** the file the V11 potential was fitted to —
that is `data/train_v9.xyz`. The V6 README called this file "1827 configurations including fcc
Al, Al2O3 and Al/ZrW2O8 interfaces"; both numbers and the species list were wrong, which is why
it is renamed here.

---

## 2. Results at a glance

Each row is one number of the paper and the JSON key it comes from; nothing here is quoted from
the manuscript.

| quantity | value | file : key |
|---|---|---|
| ΔE_st(γ−α), static, V11 | 2.7064 meV/atom | `data/json/eos.json` : `delta.dEst_meV_per_atom` |
| ΔV₀/V between the phases | −5.06 % | `data/json/eos.json` : `delta.dV0_pct` |
| response coefficient *C* = dP/d(ΔG) | 0.1755 GPa/(meV/atom) | `data/json/eos.json` : `delta.coef_GPa_per_meV_atom` = `pt_line.json` : `consistency.coef_variants.v11_fit_dV.coef_GPa_per_meV_atom` |
| ΔE_st minus the PBE reference (2.52) | +0.1864 meV/atom | `data/json/eos.json` : `delta.dEst_vs_DFT` |
| ΔZPE(γ−α) at each phase's own V₀ | +0.7907 meV/atom | `data/json/zpe_phdos.json` : `delta.dZPE_meV_at` |
| ΔZPE on one common 24³ mesh | +0.7626 meV/atom | `data/json/zpe_mesh_check.json` : `delta.same_mesh_24_meV_at.value` |
| ΔG(0 K) = ΔE_st + ΔZPE | 3.4971 meV/atom | `data/json/pt_line.json` : `consistency.sum_meV_at` |
| transition pressure at 0 K | 0.6345 GPa | `data/json/pt_line.json` : `Pt0K_GPa` |
| critical temperature at P = 0 | none (ΔG stays positive 0–600 K) | `data/json/pt_line.json` : `Tc_at_P0_K` = `null`, `Tc_note` |
| coexistence line valid up to | 475 K | `data/json/pt_line.json` : `Pt_valid_T_max_K` |
| α: volume expansion 0→300 K | −0.548 % (negative, as measured) | `data/json/cte.json` : `phases.alpha.dV_pct_0_300K` |
| α: α_V at 300 K | −2.62×10⁻⁵ K⁻¹ | `data/json/cte.json` : `phases.alpha.alpha_V_300K` |
| γ: volume expansion 0→300 K | +0.251 % | `data/json/cte.json` : `phases.gamma.dV_pct_0_300K` |
| γ: α_V at 300 K | +1.79×10⁻⁵ K⁻¹ | `data/json/cte.json` : `phases.gamma.alpha_V_300K` |
| bulk modulus B_Hill, α / γ | 91.63 / 67.48 GPa | `data/json/elastic.json` : `phases.*.B_Hill` |
| shear modulus G_Hill, α / γ | 31.92 / 34.29 GPa | `data/json/elastic.json` : `phases.*.G_Hill` |
| PBE reference ΔE(γ−α), relaxed endpoint | +2.5247 meV/atom | `data/json/abacus_results.json` : `core.cellrelax_endpoint_maxF_below_0.01.dE_meV_per_atom` |
| PBE reference ΔE(γ−α), statics protocol | +2.4608 ± 0.1427 meV/atom | `data/json/abacus_results.json` : `core.dE_meV_per_atom`, `uncertainty_budget.total_quadrature` |
| spread over relaxed γ structures | 2.4608–2.6638 meV/atom | `data/json/abacus_results.json` : `core.spread_over_relaxed_gamma_structures` |
| PBE reference P_t | 0.4407 GPa | `data/json/abacus_results.json` : `core.P_t_GPa` |
| experimental upper bound on the equilibrium pressure | 0.210 GPa (upper bound) → ≤1.197 meV/atom = 0.210 / 0.1755 | experimental input, not computed here; converted with `data/json/pt_line.json` : `consistency.coef_variants.v11_fit_dV.coef_GPa_per_meV_atom` |
| ΔE(γ−α), one static point, float32 / float64 | 7.2798 / 7.8156 meV/atom (4 repeats each, spread 0.0, systematic offset −0.5358) | `data/json/precision_check.json` : `float32`, `float64`, `systematic_float32_minus_float64_meV_at` |

`data/json/audit_results.json` is the companion audit of the *training-frame* route to the same
quantity (frame-level ΔE, zero-force extrapolation, and the decomposition of the 7.43 → 2.52 gap).

### 2.1 The 300 K error ledger behind Fig. 5(b)

Fig. 5(b) is a waterfall, not the old seven-bar ledger: five increments, each bar starting where
the previous one ended, so a bar's height is an increment and its top is the running total
(`data/json/pt_line.json` : `rows[0]`, `rows[300]`; `eos.json` : `delta`; `zpe_phdos.json` : `delta`).

| # | increment (source) | amount (meV/atom) | running total |
|---|---|---|---|
| 1 | PBE static reference, `abacus_results.json` : `core.cellrelax_endpoint_maxF_below_0.01.dE_meV_per_atom` = 2.5247 | +2.5247 | 2.5247 |
| 2 | MLIP residual, `eos.json` : `delta.dEst_meV_per_atom` = 2.7064 | +0.1817 | 2.7064 |
| 3 | zero-point energy, `zpe_phdos.json` : `delta.dZPE_meV_at` = 0.7907 | +0.7907 | 3.4971 |
| 4 | volume-relaxation closure | +0.1063 | 3.6034 |
| 5 | −TΔS at 300 K | +3.6173 | 7.2207 |

Row 4 is the closure between the QHA Gibbs minimisation at 0 K (`rows[0].dG_P0_eV_at` = 3.6034)
and ΔE_st + ΔZPE (`pt_line.json` : `consistency.sum_meV_at` = 3.4971); row 5 is the difference of
the same two quantities at 300 K. The five increments sum to 7.2207 meV/atom exactly.

Two levels are drawn but are **not** bars: the experimental upper bound ≤1.197 meV/atom (dashed
line — 0.210 GPa, an upper bound on the equilibrium pressure, divided by *C* = 0.1755
GPa/(meV/atom)) and the total discrepancy of at least **6.024** meV/atom (7.2207 − 1.1968, drawn as
a double arrow). The potential's own learning residual, 0.1817 meV/atom, is
0.1817 / 6.024 = **3.0 %** of that discrepancy.

The running totals above are stacked from unrounded JSON values; a table stacked from rounded
increments prints 2.707 / 3.498 / 3.604 for rows 2–4.


---

## 3. The five figures

All five are regenerated from `data/json/` only — no electronic-structure, force or phonon
calculation is rerun. Width is 518.4 pt (7.2 in) for every figure. The four scripts that changed
this round (`make_fig1.py`, `make_fig3.py`, `make_fig4.py`, `make_fig5.py`) are the manuscript
scripts `manuscript/jcp_v2/figures/make_figN.py` with exactly one edit: the two path constants, so
that they read `<script dir>/../data/json` and write `<script dir>/../figures` instead of an
absolute `qha_v11` directory. All plotting code is identical, which is why the extracted text of
each PDF matches its manuscript counterpart character for character (Section 4).
`make_fig2.py` was already current and is unchanged.

| figure | script | data read | what it shows |
|---|---|---|---|
| `figures/fig1.pdf` | `code/make_fig1.py` | `eos.json` (`phases.*.table`, `phases.*.BM`, `delta`), `pt_line.json` (`consistency`) | (a) E_st(V) of α and γ with the third-order Birch–Murnaghan curve fit to each phase (13 points for α, 10 of 13 for γ) and ΔV₀/V = −5.06 %; the three γ volumes at V/V₀ ≥ 1.04 are the WO₄→WO₅ reconstruction, drawn as open squares and excluded from both fit and curve; (b) the energy→pressure ruler with slope 0.1755 GPa/(meV/atom), marking the experimental **upper bound** ≤1.197 meV/atom (= 0.210 GPa / 0.1755), the PBE value 2.5247, the V11 static value 2.7064 and the V11+ZPE value 3.4971 |
| `figures/fig2.pdf` | `code/make_fig2.py` | `phonons_all.json` (`phases.*.vol_*/{fmin_THz,n_imag}`) | lowest phonon frequency against V/V₀ for both phases, with the number of imaginary modes marked; shows the α instability window and the WO₄→WO₅ reconstruction region |
| `figures/fig3.pdf` | `code/make_fig3.py` | `zpe_phdos.json` (`phases.*.phdos_representative`, `phases.*.bands_detail`, `bands`), `zpe_mesh_check.json` | phonon DOS of both phases at V/V₀ = 1 and the band-resolved decomposition of ΔZPE over the three band groups `0–4`, `4–16` and `>16` THz (en dashes in the tick labels), whose band sums add to +0.7626 meV/atom, the common-24³-mesh value |
| `figures/fig4.pdf` | `code/make_fig4.py` | `pt_line.json` (`rows[*]`, `Pt_valid_T_max_K`), `cte.json` (`phases.*`) | (a) coexistence line P_t(T) over its validated range — both curves stop at T = 475 K (`Pt_valid_T_max_K`) and the region beyond it is shaded "no QHA solution"; (b) volumetric α_V of both phases, framed to −8.2…+6.2 ×10⁻⁵ K⁻¹, with α_V(300 K) = −2.62×10⁻⁵ / +1.79×10⁻⁵ K⁻¹ marked |
| `figures/fig5.pdf` | `code/make_fig5.py` | `pt_line.json` (`rows[*]`), `eos.json` (`delta`), `zpe_phdos.json` (`delta`) | (a) Clausius–Clapeyron isotherms of G_γ − G_α at 0–400 K; (b) the 300 K ledger, now a **waterfall**: each of the five increments starts where the previous one ended and the running total climbs 2.5247 → 2.7064 → 3.4971 → 3.6034 → 7.2207 meV/atom, with the experimental upper bound and the total discrepancy drawn as a line and an arrow rather than as bars (Section 2.1) |

Not reproduced from V11 data: the crystal-structure panel of the old Fig. 1 (it needs a
re-rendered reference cell) and the numerical-drift sticks of the old Fig. 2 panel (b). Both
depended on V6/V9 measurements and neither belongs to the present five-figure set; the old
scripts are kept in `code/superseded_v6/` for lineage only.

---

## 4. Reproducing the figures

```bash
cd deposit
python3 code/make_fig1.py     # writes figures/fig1.pdf
python3 code/make_fig2.py
python3 code/make_fig3.py
python3 code/make_fig4.py
python3 code/make_fig5.py
```

Each script resolves its input as `<script dir>/../data/json` and its output as
`<script dir>/../figures`, so the commands work from any working directory. Requirements:
Python 3 with numpy and matplotlib. The `python3` on `PATH` must be the interpreter that carries
them — on the machine that produced the shipped PDFs that is `/home/jss/miniconda3/bin/python`
(Python 3.13.0, numpy 1.26.4, **matplotlib 3.10.8**), whereas `/usr/bin/python3` has no
matplotlib. Apart from the two path constants, the four scripts that changed are identical to
`manuscript/jcp_v2/figures/make_figN.py`, so the extracted text of `figures/figN.pdf` matches the
manuscript PDF for N = 1, 3, 4, 5 — and for N = 2, which was not touched this round.

Regenerating the five PDFs with matplotlib 3.10.8 reproduces the files in `figures/`
**byte-identically except for the PDF `CreationDate`/`ID`** — 5 bytes out of a 30–43 kB
file. The PDFs are therefore *not* byte-reproducible and no md5 sum is quoted here; the stable
check is the file size plus the extracted text, both of which repeat exactly:

| figure | size (bytes) | page size (pt) |
|---|---|---|
| `figures/fig1.pdf` | 43119 | 518.4 × 183.6 |
| `figures/fig2.pdf` | 30707 | 518.4 × 194.4 |
| `figures/fig3.pdf` | 37640 | 518.4 × 187.2 |
| `figures/fig4.pdf` | 33824 | 518.4 × 194.4 |
| `figures/fig5.pdf` | 38153 | 518.4 × 198 |

---

## 5. File inventory

```
deposit/
  README.md                     this file
  CHANGELOG.md                  V6 -> V11, path by path, with the evidence for each move
  CITATION.cff  LICENSE  LICENSE-DATA
  model/
    ZrW_v11_seed3_stagetwo.model   stage-two V11 potential: produces every number here
    ZrW_v11_seed3.model            stage-one V11 potential (best of stage one)
    superseded_v6/AlZrW_v6_stagetwo.model
  data/
    train_v9.xyz                   1539 frames (991/515/33 alpha/gamma/beta)
    valid_v9.xyz                   258 frames (171/85/2)
    e0s_v9.json                    elemental reference energies
    dataset_meta.json              counts, zero alignment, gamma_shift_applied = 0.0
    raw_pbe_frames.xyz             1763 raw PBE frames - provenance only, NOT the training set
    json/                          the eleven result files listed in Sections 2-4
    json/precision_check.json      float32 vs float64 single point: 4 repeats each,
                                   7.2798 vs 7.8156 meV/atom, systematic offset -0.5358
    superseded_v6_json/            46 V6-era JSON files
  figures/
    fig1.pdf .. fig5.pdf           regenerated from data/json
    superseded_v6/fig1.pdf .. fig5.pdf
  code/
    README.md                      which script produces which number, and which are V6-era
    make_fig1.py .. make_fig5.py   the five figure scripts (read ../data/json)
    stage_eos.py                   E_st(V) staging: ion-only relaxation on a volume grid
    phonon_pipeline.py             finite-displacement phonons (gen / forces / post)
    qha_analysis.py                BM fits, ZPE, QHA, coexistence line, CTE, elastic summary
    stage_elastic.py               stress-strain elastic constants
    extract.py                     parse ABACUS running_*.log into a summary
    audit.py                       audit of the training-frame route to DeltaE
    phase_identity.py              alpha vs gamma phase-identity check on STRU files
    recompute_deltaE.py            rebuild every number of abacus_results.json from raw logs
    superseded_v6/make_fig1.py .. make_fig5.py
    NN_*.py, *.sbatch              49 V6-era scripts at this level (43 analysis/post-
                                   processing .py + 6 Slurm .sbatch), kept for lineage only
```

`data/json/` holds exactly: `eos.json`, `zpe_phdos.json`, `pt_line.json`, `cte.json`,
`elastic.json`, `thermal.json`, `zpe_mesh_check.json`, `phonons_all.json`,
`abacus_results.json`, `audit_results.json`, `precision_check.json`.

`precision_check.json` records the single-point precision check behind the systematic-offset
statement: ΔE(γ−α) at the two reference geometries (44-atom α, 132-atom γ), no relaxation,
evaluated four times in float32 and four times in float64 with the same model
(`model/ZrW_v11_seed3_stagetwo.model`). Each precision is run-to-run reproducible (spread 0.0
meV/atom in both), and the two differ by a systematic **−0.5358 meV/atom** (float32 7.2798,
float64 7.8156) — a precision offset, not scatter. Keys: `repeats`, `float32`, `float64`,
`spread_float32_meV_at`, `spread_float64_meV_at`, `systematic_float32_minus_float64_meV_at`,
`how`.

---

## 6. What is not included

The finite-displacement force sets (one `FORCE_SETS` per displacement) and the ABACUS relaxation
archives are not stored here because of their size. `code/` contains the scripts that
produced them, with the directory layout they expect; the archives themselves are available from
the corresponding author on reasonable request.

## 7. Citation

If you use the potential or the data, please cite the paper above and this repository.

**Version.** This is release **v1.1.0** (2026-11-25). It replaces the V6 bundle of v1.0.0 with the
V11 potential, the 1539/258 training set and the re-derived results. Pin the content by the tag:

```
https://github.com/jinshuaishuai0305/zrw2o8-alpha-gamma-mlip/tree/v1.1
```

No separate data DOI is registered for this bundle: the Zenodo draft prepared for the raw force sets
is not being published, and the raw force sets and relaxation archives (~1 GB) remain available from
the corresponding author on reasonable request, as stated in the paper's Data availability section.
The tag fixes the exact content of the data and code released here.

## 8. License

- Code (`code/`): MIT License.
- Data (`data/`, `model/`, `figures/`): Creative Commons Attribution 4.0 (CC-BY-4.0).

The fitted potential is a derivative work of MACE-MP-0; the terms of that model and of the MACE
software apply to redistribution.
