#!/usr/bin/env python3
"""Fig. 4 - Computed alpha/gamma boundary and thermal expansion against experiment."""
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
C_EXP = '#000000'; C_MLIP = '#0072B2'; C_PBE = '#D55E00'

raw = json.load(open(ROOT + 'pt_line_v6_raw.json'))
dft = json.load(open(ROOT + 'pt_line_dft_aligned.json'))
def curve(d, tmax=None):
    T = np.array([r['T'] for r in d['rows']], float)
    P = np.array([r['Pt'] for r in d['rows']], float)
    o = np.argsort(T); T, P = T[o], P[o]
    if tmax is not None:
        m = T <= tmax; T, P = T[m], P[m]
    return T, P

fig = plt.figure(figsize=(7.2, 2.85))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.26,
                      left=0.072, right=0.985, top=0.90, bottom=0.245)

# ---------------------------------------------------------------- (a)
ax = fig.add_subplot(gs[0, 0])
Plim = 1.90
ax.axhspan(Plim, 3.2, color='#dddddd', alpha=0.55, lw=0)
for y in np.arange(Plim + 0.06, 3.1, 0.16):
    ax.plot([0, 700], [y, y], color='#bbbbbb', lw=0.5, ls='-', zorder=0)
ax.text(690, Plim + 0.10, r'$\gamma$ dynamically unstable', fontsize=6.4, color='#666666',
        ha='right', va='bottom')
T4, P4 = curve(raw, 400); T5, P5 = curve(raw)
ax.plot(T5[T5 >= 400], P5[T5 >= 400], '--', color=C_MLIP, lw=1.2)
ax.plot(T4, P4, '-', color=C_MLIP, lw=1.6, label=r'this work (V6), full QHA')
Td, Pd = curve(dft, 700)
ax.plot(Td, Pd, ':', color=C_PBE, lw=1.4, label=r'same PBE functional, labels corrected')
ax.plot(300, 0.21, '*', ms=8, color=C_EXP, zorder=6, label=r'experiment (onset)')
ax.annotate('', xy=(300, 0.21), xytext=(300, 1.374),
            arrowprops=dict(arrowstyle='<->', color='#333333', lw=0.8))
ax.plot([300, 300], [0.21, 1.374], 'o', ms=2.6, color='#333333')
ax.text(312, 0.72, r'$1.16$ GPa', fontsize=6.6, color='#333333')
ax.plot([400, 400], [0, 3.0], color='#999999', lw=0.7, ls='-')
ax.text(388, 2.30, r'valid range $\leq 400$ K', fontsize=6.3, color='#555555',
        rotation=90, va='top', ha='right')
ax.set_xlabel(r'$T$ (K)'); ax.set_ylabel(r'$P_t$ (GPa)')
ax.set_xlim(0, 700); ax.set_ylim(0, 3.0)
ax.legend(frameon=False, loc='upper left', handletextpad=0.5, labelspacing=0.28,
          borderpad=0.2, bbox_to_anchor=(0.055, 1.005))
ax.text(0.02, 0.97, '(a)', transform=ax.transAxes, va='top', fontsize=9, fontweight='bold')

# ---------------------------------------------------------------- (b)
ax2 = fig.add_subplot(gs[0, 1])
ath = json.load(open(ROOT + 'alpha_thermo_v6.json'))
Ta = np.array(ath['T'], float); aa = np.array(ath['alpha_ppm'], float)
m = (Ta >= 50) & (Ta <= 550)
ax2.plot(Ta[m], aa[m], '-', color=C_MLIP, lw=1.4, label=r'this work (V6), $\alpha$')
tt = json.load(open(ROOT + 'thermo_table.json'))['gamma']
Tg = np.array(tt['T'], float); Vg = np.array(tt['V'], float)
alph = np.gradient(Vg, Tg) / (3.0 * Vg) * 1e6
mg = (Tg >= 50) & (Tg <= 550)
ax2.plot(Tg[mg], alph[mg], '-', color=C_PBE, lw=1.4, label=r'this work (V6), $\gamma$')
ax2.axhline(-8.9, color=C_EXP, lw=1.0, ls='--')
ax2.text(555, -8.9 + 0.7, r'experiment $\alpha$ ($-8.9$ ppm/K)', fontsize=6.4,
         color=C_EXP, ha='right')
ax2.plot(300, -1.0, 'D', ms=3.8, color=C_EXP, zorder=6)
ax2.text(310, -2.6, r'experiment $\gamma$', fontsize=6.4, color=C_EXP)
ax2.axhline(0, color='#bbbbbb', lw=0.6)
ax2.set_xlabel(r'$T$ (K)'); ax2.set_ylabel(r'linear expansion $\alpha_L$ (ppm/K)')
ax2.set_xlim(0, 560); ax2.set_ylim(-24, 22)
ax2.legend(frameon=False, loc='lower left', handletextpad=0.5, labelspacing=0.25,
           borderpad=0.2)
ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, va='top', fontsize=9, fontweight='bold')
fig.savefig(os.path.join(OUT, 'fig4.pdf'))
fig.savefig(os.path.join(OUT, 'fig4.png'), dpi=300)
print('fig4: V6 line 0/300/400 K = %.3f/%.3f/%.3f | PBE 0 K = %.3f' %
      (raw['rows'][0]['Pt'], [r for r in raw['rows'] if r['T'] == 300][0]['Pt'],
       [r for r in raw['rows'] if r['T'] == 400][0]['Pt'], dft['rows'][0]['Pt']))
sel = (Ta >= 300) & (Ta <= 430)
print('     alpha CTE 300-430 K = %.3f ppm/K (exp -8.9)' % aa[sel].mean())
print('     gamma CTE at 300 K = %.3f ppm/K' % alph[np.argmin(abs(Tg - 300))])
