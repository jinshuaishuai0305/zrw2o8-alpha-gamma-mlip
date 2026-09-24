# Data and code for "Thermodynamic Error Amplification in Near-Degenerate Phase Transitions"

This repository accompanies the manuscript

> **Thermodynamic Error Amplification in Near-Degenerate Phase Transitions: Machine-Learning
> Potentials and the alpha/gamma Boundary of ZrW2O8**
> Shuaishuai Jin, Zhong Guan, Zhiyong You, Bing Li, Hang Li
> (submitted; journal reference to be added)

It contains the fitted interatomic potential, the processed data behind every number and
figure in the paper, and the analysis scripts needed to regenerate them.

## Contents

| Path | What it is |
|---|---|
| `model/AlZrW_v6_stagetwo.model` | The fitted MACE potential ("V6") used for all results in the paper. Fine-tuned in two stages from MACE-MP-0 (small) via an intermediate Zr-W-O-Al model; see the manuscript, Sec. II B. |
| `data/train.xyz` | The training set (1827 configurations: alpha- and gamma-ZrW2O8, fcc Al, Al2O3 and Al/ZrW2O8 interfaces). |
| `data/json/` | Processed results in JSON form: coexistence line and its DFT-aligned counterpart, phonon density of states and zero-point energies, thermal-expansion tables, elastic constants, convergence/precision diagnostics, energy-volume curves. These files are the direct input of the figure scripts. |
| `code/` | Analysis and post-processing scripts (structure generation, force collection, QHA/Gibbs solution, PHDOS and ZPE decomposition, elastic constants, precision diagnostics) plus the Slurm submission scripts used on the cluster, and the five figure scripts. |
| `figures/` | The five vector figures of the paper (PDF). |

## Reproducing the figures

The figure scripts read only the archived JSON in `data/json/`; no electronic-structure
calculation is rerun. From a directory containing `figures/` and `data/json/` side by side:

```bash
python3 code/make_fig1.py    # motivated example: structures, E(V), sensitivity ruler
python3 code/make_fig2.py    # convergence and stability diagnostics
python3 code/make_fig3.py    # phonon DOS and band-resolved zero-point decomposition
python3 code/make_fig4.py    # coexistence line and thermal expansion
python3 code/make_fig5.py    # Clausius-Clapeyron amplification and the tripartite ledger
```

The scripts in this repository point at the archived JSON through a `ROOT` constant near the
top of each file; adjust that path to your local checkout. Requirements: Python 3.12 with
numpy, scipy and matplotlib; the phonon and potential-related scripts additionally need
`mace` (0.3.x), `ase` and `phonopy`.

## Not included here

The raw finite-displacement force sets (39 x FORCE_SETS, ~1 GB uncompressed) and the
relaxation archives are not stored in this repository because of their size. They are
deposited separately and are available from the corresponding author on reasonable request.

## Citation

If you use the potential or the data, please cite the paper above (and this repository, once
its DOI is registered).

## License

- Code (`code/`): MIT License.
- Data (`data/`, `model/`, `figures/`): Creative Commons Attribution 4.0 (CC-BY-4.0).

The fitted potential is a derivative work of MACE-MP-0; the terms of that model and of the
MACE software apply to redistribution.
