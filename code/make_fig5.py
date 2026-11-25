#!/usr/bin/env python3
"""Fig. 5 - isotherms of the Gibbs free-energy difference and the 300 K error ledger, V11 data.

Data sources (all read-only):
  deposit/data/json/pt_line.json
        rows[*].{T,Pt_GPa,Va_A3_at,Vg_A3_at,dG_P0_eV_at}
  deposit/data/json/eos.json      delta.dEst_meV_per_atom, :delta.coef_GPa_per_meV_atom
  deposit/data/json/zpe_phdos.json delta.dZPE_meV_at
  PBE relaxed reference 2.5247 meV/atom               (abacus_results.json : core.cellrelax_endpoint_maxF_below_0.01)
  experimental onset 0.210 GPa                        (experimental input, not computed here)

Panel (b) is drawn in meV/atom on purpose: at 300 K a given energy is NOT converted into a
pressure by the 0 K coefficient C, because dV has changed with temperature
(7.2207 meV/atom <-> 1.5204 GPa, ratio 0.2105, not 0.1755). The pressure 1.5204 GPa comes
from the Gibbs minimisation in pt_line.json, not from dE * C.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QHA = os.path.join(HERE, '..', 'data', 'json') + os.sep
OUT = os.path.join(HERE, '..', 'figures')
GPA_A3_PER_EV = 160.2177

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 6.6,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.1,
    'axes.spines.top': False, 'axes.spines.right': False,
})

pt = json.load(open(QHA + 'pt_line.json'))
eos = json.load(open(QHA + 'eos.json'))
zpe = json.load(open(QHA + 'zpe_phdos.json'))

C = eos['delta']['coef_GPa_per_meV_atom']          # 0.175469 GPa/(meV/atom), 0 K ruler
dEst = eos['delta']['dEst_meV_per_atom']           # 2.70642
dZPE = zpe['delta']['dZPE_meV_at']                 # 0.79070
PBE_REF = 2.5247                                   # abacus_results.json : core.cellrelax_endpoint_maxF_below_0.01
EXP_P = 0.210                                      # experimental input
EXP_E = EXP_P / C                                  # 1.19680 meV/atom

rows = {int(r['T']): r for r in pt['rows']}
dG_0K = rows[0]['dG_P0_eV_at'] * 1e3               # 3.60339
dG_300K = rows[300]['dG_P0_eV_at'] * 1e3           # 7.22073
closure = dG_0K - (dEst + dZPE)                    # 0.10627
dS_term = dG_300K - dG_0K                          # 3.61735

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.2, 2.75))
gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.075, right=0.985, top=0.92, bottom=0.22,
                      width_ratios=[1.15, 1.0])

# ------------------------------------------------ (a) Clausius-Clapeyron isotherms
ax = fig.add_subplot(gs[0, 0])
Pax = np.linspace(0, 2.6, 400)
colors = plt.cm.viridis(np.linspace(0.1, 0.82, 5))
for c, Tk in zip(colors, [0, 100, 200, 300, 400]):
    r = rows[Tk]
    dV = r['Va_A3_at'] - r['Vg_A3_at']
    dGP = (r['Pt_GPa'] - Pax) * dV / GPA_A3_PER_EV * 1000.0     # meV/atom
    ax.plot(Pax, dGP, color=c, lw=1.2, label='%d K' % Tk)
    ax.plot(r['Pt_GPa'], 0.0, 'o', ms=3.4, mfc='w', mec=c, mew=0.9)
ax.axhline(0, color='k', lw=0.7)
ax.axvline(EXP_P, color='k', lw=0.9, ls=':')
ax.plot(EXP_P, np.interp(EXP_P, Pax, (rows[300]['Pt_GPa'] - Pax) *
        (rows[300]['Va_A3_at'] - rows[300]['Vg_A3_at']) / GPA_A3_PER_EV * 1000.0),
        marker='D', ms=5, color='k', label='experiment (onset, 300 K)')
ax.plot(EXP_P, 0.0, marker='D', ms=4, color='k', mfc='none')
ax.set_xlim(0, 2.6); ax.set_ylim(-9.0, 7.4)
ax.set_xlabel('$P$ (GPa)'); ax.set_ylabel(r'$G_\gamma - G_\alpha$ (meV/atom)')
ax.legend(frameon=False, loc='lower left', ncol=2, columnspacing=0.8, handlelength=1.2)
ax.set_title('(a)', loc='left', fontsize=8, fontweight='bold')

# ------------------------------------------------ (b) 300 K ledger, energy units only
ax = fig.add_subplot(gs[0, 1])
labels = ['PBE\nstatic', 'MLIP\nresidual', 'ZPE', 'volume\nclosure',
          r'$-T\Delta S$' + '\n(300 K)']
incs = [PBE_REF, dEst - PBE_REF, dZPE, closure, dS_term]
cols = ['#D55E00', '#0072B2', '#009E73', '0.55', '#332288']
base = np.concatenate([[0.0], np.cumsum(incs)[:-1]])     # each bar starts where the last ended
tops = np.cumsum(incs)
ax.bar(range(5), incs, bottom=base, width=0.55, color=cols, edgecolor='k', lw=0.5)
for i in range(4):                                        # waterfall connectors
    ax.plot([i + 0.275, i + 1 - 0.275], [tops[i], tops[i]], ls=':', color='0.45', lw=0.7)
for i, (b0, h) in enumerate(zip(base, incs)):
    ax.text(i, b0 + h + 0.14, '%.3f' % tops[i], ha='center', va='bottom', fontsize=5.8)
ax.axhline(EXP_E, ls='--', color='0.35', lw=0.8)
ax.text(3.62, EXP_E + 0.18, 'experiment, $\\leq 1.197$', fontsize=5.8, color='0.3',
        ha='right', va='bottom')
ax.annotate('', xy=(4.62, dG_300K), xytext=(4.62, EXP_E),
            arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.3'))
ax.text(4.43, 0.5 * (dG_300K + EXP_E), 'discrepancy  $\\geq 6.024$', rotation=90,
        ha='center', va='center', fontsize=5.8, color='0.3',
        bbox=dict(facecolor='w', edgecolor='none', pad=0.6))
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=5.9)
ax.set_xlim(-0.55, 4.85)
ax.set_ylabel('computed driving force (meV/atom)'); ax.set_ylim(0, 9.0)
ax.set_title('(b)', loc='left', fontsize=8, fontweight='bold')

fig.savefig(os.path.join(OUT, 'fig5.pdf'))
print('fig5.pdf written')
print('C               = %.6f' % C)
print('exp anchor      = %.5f meV/atom (0.210 GPa / C)' % EXP_E)
print('PBE increment   = %.5f' % (PBE_REF - EXP_E))
print('MLIP residual   = %.5f' % (dEst - PBE_REF))
print('dZPE            = %.5f' % dZPE)
print('dG(0 K)         = %.5f' % dG_0K)
print('closure         = %.5f' % closure)
print('dG(300 K)       = %.5f' % dG_300K)
print('-TdS(300 K)     = %.5f' % dS_term)
print('sum of rows     = %.5f  (diff to dG300 = %.6f)'
      % (EXP_E + (PBE_REF - EXP_E) + (dEst - PBE_REF) + dZPE + closure + dS_term,
         EXP_E + (PBE_REF - EXP_E) + (dEst - PBE_REF) + dZPE + closure + dS_term - dG_300K))
print('ratio dG300/Pt300 = %.4f GPa/(meV/atom)' % (dG_300K / rows[300]['Pt_GPa']))
