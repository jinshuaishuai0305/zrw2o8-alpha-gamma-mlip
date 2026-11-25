#!/usr/bin/env python3
"""Fig. 3 - origin of the zero-point difference, V11 data.

Data source: deposit/data/json/zpe_phdos.json
  phases.{alpha,gamma}.phdos_representative.{grid,dos}   (both at V/V0 = 1.0000)
  phases.{alpha,gamma}.bands_detail[tag=vol_1.0000].bands  (per-band ZPE)

The band deltas are the arithmetic difference of the two phases' per-band ZPE values at
the common sampling volume V/V0 = 1; they sum to +0.7626 meV/atom
(zpe_mesh_check.json:delta.same_mesh_24_meV_at.value). The value used in the text
(+0.7907) is at the respective Birch-Murnaghan volumes.
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
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.1,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_A, C_G = '#0072B2', '#D55E00'

d = json.load(open(QHA + 'zpe_phdos.json'))

def phdos(phase):
    r = d['phases'][phase]['phdos_representative']
    return np.array(r['grid'], float), np.array(r['dos'], float)

def bands(phase):
    for e in d['phases'][phase]['bands_detail']:
        if e['tag'] == 'vol_1.0000':
            return e['bands']
    raise KeyError(phase)

ba, bg = bands('alpha'), bands('gamma')
# tick labels must not carry the LaTeX-style '--' (matplotlib prints it verbatim)
groups = [('$0$–$4$', [0, 1, 2]), ('$4$–$16$', [3, 4]), ('$>16$', [5])]
dz = np.array([sum(bg[i]['zpe'] - ba[i]['zpe'] for i in idx) for _, idx in groups])
fr_a = np.array([sum(ba[i]['frac'] for i in idx) for _, idx in groups]) * 100
fr_g = np.array([sum(bg[i]['frac'] for i in idx) for _, idx in groups]) * 100
tot = dz.sum()
share = dz / tot * 100

fig = plt.figure(figsize=(7.2, 2.6))
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.062, right=0.988, top=0.92, bottom=0.235,
                      width_ratios=[1.25, 0.85, 0.85])

ax = fig.add_subplot(gs[0, 0])
for phase, c, lab in (('alpha', C_A, r'$\alpha$'), ('gamma', C_G, r'$\gamma$')):
    g, dos = phdos(phase)
    ax.plot(g, dos, color=c, lw=1.0, label=lab)
ax.set_xlim(0, 32); ax.set_ylim(0, 0.68)
ax.set_xlabel('frequency (THz)'); ax.set_ylabel('PHDOS (states/THz/atom)')
ax.legend(frameon=False, loc='upper right')
ax.set_title('(a)', loc='left', fontsize=8, fontweight='bold')

ax = fig.add_subplot(gs[0, 1])
x = np.arange(3)
ax.bar(x, dz, width=0.62, color=['#4477AA', '#CC6677', '#88CCEE'], edgecolor='k', lw=0.5)
ax.axhline(0, color='k', lw=0.7)
for i, v in enumerate(dz):
    ax.text(i, v + (0.09 if v > 0 else -0.20), '%+.3f' % v, ha='center', fontsize=6.6)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_xlabel('band (THz)'); ax.set_ylabel(r'$\Delta$ZPE (meV/atom)')
ax.set_ylim(-1.55, 2.75)
ax.set_title('(b)', loc='left', fontsize=8, fontweight='bold')

ax = fig.add_subplot(gs[0, 2])
cum = np.concatenate([[0.0], np.cumsum(dz)])
ax.plot(np.arange(4), cum, 'o-', ms=3.5, color='#332288', lw=1.1)
ax.axhline(tot, color='0.5', ls='--', lw=0.7)
ax.text(0.06, tot + 0.13, r'net $= +%.3f$' % tot, fontsize=6.8, color='0.3')
ax.set_xticks(range(4)); ax.set_xticklabels(['start'] + [g[0] for g in groups])
ax.set_xlabel('band (THz)'); ax.set_ylabel('cumulative (meV/atom)')
ax.set_ylim(-1.7, 2.75)
ax.set_title('(c)', loc='left', fontsize=8, fontweight='bold')

fig.savefig(os.path.join(OUT, 'fig3.pdf'))
print('fig3.pdf written; dZPE bands =', np.round(dz, 4), 'sum', round(tot, 4),
      'shares', np.round(share, 0), 'frac alpha', np.round(fr_a, 2), 'gamma', np.round(fr_g, 2))
