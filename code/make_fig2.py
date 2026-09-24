#!/usr/bin/env python3
"""Fig. 2 (manuscript) - Convergence and stability diagnostics: where the error is NOT."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/jss/share/AlZrW2O8_MACE/diag_precision/json/'
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 6.8,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_2 = '#D55E00'; C_3 = '#0072B2'; C_SIG = '#009E73'

gc = json.load(open(ROOT + 'gamma_333_phonons.json'))['phonons']
V0 = gc['eq']['V']
x3 = [gc[k]['V'] / V0 for k in ['c0915', 'c0925', 'c0935', 'c0945', 'eq']]
y3 = [gc[k]['fmin'] for k in ['c0915', 'c0925', 'c0935', 'c0945', 'eq']]
tc = json.load(open(ROOT + 'thermal_compression.json'))
x2 = [d['V'] / V0 for d in tc][::-1]; y2 = [d['fmin'] for d in tc][::-1]

fig = plt.figure(figsize=(7.2, 2.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.14], wspace=0.42,
                      left=0.075, right=0.985, top=0.865, bottom=0.235)

# ---------------------------------------------------------------- (a)
ax = fig.add_subplot(gs[0, 0])
ax.axhline(0, color='#888888', lw=0.7)
ax.axhspan(-1.15, 0, color='#f2dede', alpha=0.7, lw=0, zorder=0)
for i, (lo, hi) in enumerate([(0.905, 0.945), (0.945, 1.0)]):
    pass
ax.plot(x2, y2, 's--', ms=3.4, color=C_2, lw=1.0, label=r'$2\times2\times1$ (528 at.)')
ax.plot(x3, y3, 'o-', ms=3.6, color=C_3, lw=1.2, label=r'$3\times3\times3$ (3564 at.)')
ax.axvline(0.945, color='#999999', lw=0.7, ls=':')
ax.text(0.9415, -0.13, r'$0.945\,V_0$', fontsize=6.3, color='#555555',
        ha='right', va='center')
ax.plot(1.0, y3[-1], 'o', ms=3.6, color=C_3)
ax.text(0.995, 0.30, r'equilibrium', fontsize=6.3, color=C_3, ha='center')
ax.set_xlabel(r'$\gamma$-phase volume $V/V_0$')
ax.set_ylabel(r'softest mode (THz)')
ax.set_xlim(0.90, 1.03); ax.set_ylim(-1.15, 0.85)
ax.text(0.0, 1.025, '(a)', transform=ax.transAxes, va='bottom', ha='left',
        fontsize=9, fontweight='bold')
ax.text(0.03, 0.93, r'$a$-axis $+1\,\%$ strain:' + '\n' + r'$2\times2\times1$: $-3.01$ THz, 509 imag.' +
        '\n' + r'$3\times3\times3$: $+0.25$ THz, 0 imag.', transform=ax.transAxes,
        va='top', fontsize=6.2, color='#333333')
ax.legend(frameon=False, loc='lower right', handletextpad=0.5, labelspacing=0.25,
          borderpad=0.2)

# ---------------------------------------------------------------- (b)
ax2 = fig.add_subplot(gs[0, 1])
items = [(r'$q$-mesh', 1.0e-4, 1.0e-4, C_3),
         (r'supercell size', 4.3e-3, 4.3e-3, C_3),
         (r'float32 forces', 2.0e-2, 2.0e-2, C_3),
         (r'float32 energies', 1.0e-4, 1.0e-4, C_3),
         (r'float32 CUDA$^\dagger$', 1.5, 4.6, C_2)]
ypos = np.arange(len(items))[::-1]
for y, (lab, lo, hi, col) in zip(ypos, items):
    ax2.barh(y, hi, height=0.5, color=col, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax2.text(hi * 1.5, y, r'$%.0e$' % hi if hi < 1 else r'$%.1f$' % hi, va='center',
             fontsize=6.2)
ax2.axvline(1.70, color=C_SIG, lw=1.2, ls='--')
ax2.text(0.015, 0.972, r'physical signal: 1.70 meV/atom' + '\n' + r'($0.30$ GPa in $P_t$)',
         transform=ax2.transAxes, ha='left', va='top', fontsize=6.4, color=C_SIG,
         linespacing=1.25)
ax2.set_yticks(ypos); ax2.set_yticklabels([it[0] for it in items], fontsize=6.6)
ax2.set_xscale('log'); ax2.set_xlim(1e-5, 60)
ax2.set_xlabel(r'numerical drift (meV/atom)')
ax2.set_ylim(-1.05, len(items) + 1.25)
ax2.text(0.0, 1.025, '(b)', transform=ax2.transAxes, va='bottom', ha='left',
         fontsize=9, fontweight='bold')
fig.savefig(os.path.join(OUT, 'fig2.pdf'))
fig.savefig(os.path.join(OUT, 'fig2.png'), dpi=300)
print('fig2: 3x3x3 fmin', ['%.3f' % v for v in y3])
print('      2x2x1 fmin', ['%.3f' % v for v in y2])
print('      V/V0 3x3x3', ['%.4f' % v for v in x3])
