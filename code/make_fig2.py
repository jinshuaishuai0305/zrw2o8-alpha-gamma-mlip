#!/usr/bin/env python3
"""Fig. 2 - dynamical-stability windows of both phases, V11 data.

Data source: deposit/data/json/phonons_all.json
             phases.{alpha,gamma}.vol_<s>/ {n_imag, fmin_THz}

The old panel (b) (numerical-drift sticks: 6^3->10^3 mesh, 2x2x1->3x3x3 supercell,
float32/float64, plus the 0.30 GPa signal line) depended entirely on V6/V9 measurements
and is NOT reproduced; the V6 script is kept in `code/superseded_v6/` for lineage only.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- deposit/ adaptation: results read from ../data/json, figures written to ../figures ---
HERE = os.path.dirname(os.path.abspath(__file__))
QHA = os.path.join(HERE, '..', 'data', 'json') + os.sep
OUT = os.path.join(HERE, '..', 'figures')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_A, C_G = '#0072B2', '#D55E00'

ph = json.load(open(QHA + 'phonons_all.json'))['phases']

def series(name):
    s, f, n = [], [], []
    for tag, r in ph[name].items():
        s.append(float(tag.split('_')[1]))
        f.append(r['fmin_THz'])
        n.append(r['n_imag'])
    o = np.argsort(s)
    return np.array(s)[o], np.array(f)[o], np.array(n)[o]

fig, ax = plt.subplots(figsize=(7.2, 2.7))
fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.215)

for name, c, lab, mk in (('alpha', C_A, r'$\alpha$ (44 atoms)', 'o'),
                         ('gamma', C_G, r'$\gamma$ (132 atoms)', 's')):
    s, f, n = series(name)
    ax.plot(s, f, marker=mk, ms=3.6, lw=1.0, color=c, label=lab)
    bad = n > 0
    ax.plot(s[bad], f[bad], marker=mk, ms=6.2, mfc='none', mec=c, mew=0.9, ls='none')
    for x, y, k in zip(s[bad], f[bad], n[bad]):
        ax.annotate('%d' % k, (x, y), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=6.2, color=c)

ax.axhline(0.0, color='0.4', lw=0.8, ls='--')
ax.axvspan(0.935, 0.975, color='0.88', lw=0, zorder=0)
ax.axvspan(1.035, 1.065, color='0.93', lw=0, zorder=0)
ax.text(0.9375, -1.72, 'unstable\n(QHA invalid)', ha='left', va='bottom', fontsize=6.4, color='0.3')
ax.text(1.048, -1.72, r'WO$_4\to$WO$_5$' + '\nreconstruction', ha='center', va='bottom', fontsize=6.4, color='0.3')
ax.set_xlabel(r'$V/V_0$')
ax.set_ylabel(r'lowest phonon frequency (THz)')
ax.set_xlim(0.935, 1.065); ax.set_ylim(-1.85, 0.42)
ax.set_xticks(np.arange(0.94, 1.061, 0.02))
ax.legend(frameon=False, loc='upper left')
ax.text(0.972, 0.40, 'open symbols: imaginary modes\n(count given)', 
        fontsize=6.4, color='0.25', ha='left', va='top')

fig.savefig(os.path.join(OUT, 'fig2.pdf'))
print('fig2.pdf written')
