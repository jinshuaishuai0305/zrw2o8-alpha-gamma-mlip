#!/usr/bin/env python3
"""Fig. 4 - computed alpha/gamma boundary and thermal expansion, V11 data.

Data sources:
  deposit/data/json/pt_line.json  rows[*].{T,Pt_GPa}
  deposit/data/json/cte.json      phases.{alpha,gamma}.{T,alpha_V_K}
  experimental onset 0.210 GPa at 300 K (experimental input, not computed here)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QHA = os.path.join(HERE, '..', 'data', 'json') + os.sep
OUT = os.path.join(HERE, '..', 'figures')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 6.8,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_A, C_G, C_EXP = '#0072B2', '#D55E00', '#000000'

pt = json.load(open(QHA + 'pt_line.json'))
cte = json.load(open(QHA + 'cte.json'))
Tmax = pt['Pt_valid_T_max_K']

T = np.array([r['T'] for r in pt['rows']], float)
P = np.array([r['Pt_GPa'] for r in pt['rows']], float)
o = np.argsort(T); T, P = T[o], P[o]

fig = plt.figure(figsize=(7.2, 2.7))
gs = fig.add_gridspec(1, 2, wspace=0.27, left=0.075, right=0.985, top=0.93, bottom=0.215)

# ------------------------------------------------ (a) coexistence line
ax = fig.add_subplot(gs[0, 0])
m = T <= Tmax
ax.plot(T[m], P[m], '-', color=C_A, lw=1.4, label=r'QHA, full minimisation')
ax.axvspan(Tmax, 660, color='0.9', lw=0)
ax.text(0.5 * (Tmax + 660), 0.35, 'no QHA\nsolution', ha='center', fontsize=6.6, color='0.35')
ax.plot(300, 0.210, marker='*', ms=10, color=C_EXP, label='experiment (onset)')
ax.annotate('', xy=(300, 0.210), xytext=(300, np.interp(300, T, P)),
            arrowprops=dict(arrowstyle='<->', lw=0.8, color='0.25'))
ax.text(312, 0.62, '$1.31$ GPa', fontsize=6.8, color='0.25')
ax.set_xlabel('$T$ (K)'); ax.set_ylabel('$P_t$ (GPa)')
ax.set_xlim(0, 660); ax.set_ylim(0, 2.45)
ax.legend(frameon=False, loc='upper left')
ax.set_title('(a)', loc='left', fontsize=8, fontweight='bold')

# ------------------------------------------------ (b) volumetric CTE
ax = fig.add_subplot(gs[0, 1])
for phase, c, lab in (('alpha', C_A, r'$\alpha$'), ('gamma', C_G, r'$\gamma$')):
    b = cte['phases'][phase]
    t = np.array(b['T'], float) / 1000.0
    a = np.array(b['alpha_V_K'], float) * 1e5
    k = np.isfinite(a) & (np.array(b['T'], float) <= Tmax)   # QHA validity window
    ax.plot(t[k], a[k], 'o-', ms=2.6, lw=1.0, color=c, label=lab)
ax.axhline(0, color='k', lw=0.7, ls='--')
ax.set_xlabel('$T$ (K)'); ax.set_ylabel(r'$\alpha_V$ ($10^{-5}$ K$^{-1}$)')
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xticklabels(['0', '100', '200', '300', '400', '500', '600'])
ax.set_xlim(0, 0.62); ax.set_ylim(-8.2, 6.2)
ax.annotate(r'$-2.62\times10^{-5}$', xy=(0.30, -2.62), xytext=(0.20, -6.6),
            fontsize=6.6, color=C_A,
            arrowprops=dict(arrowstyle='-', lw=0.6, color=C_A))
ax.annotate(r'$+1.79\times10^{-5}$', xy=(0.30, 1.79), xytext=(0.045, 1.25),
            fontsize=6.6, color=C_G,
            arrowprops=dict(arrowstyle='-', lw=0.6, color=C_G))
ax.legend(frameon=False, loc='lower left')
ax.set_title('(b)', loc='left', fontsize=8, fontweight='bold')

fig.savefig(os.path.join(OUT, 'fig4.pdf'))
print('fig4.pdf written; Pt(0)=%.4f, valid T<=%.0f' % (np.interp(0, T, P), Tmax))
