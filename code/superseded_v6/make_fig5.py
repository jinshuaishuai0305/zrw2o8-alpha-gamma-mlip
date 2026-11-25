#!/usr/bin/env python3
"""Fig. 5 (manuscript) - Clausius-Clapeyron amplification and the tripartite error ledger."""
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
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.0,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.1,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_EXP = '#000000'; C_MLIP = '#0072B2'; C_PBE = '#D55E00'; C_ZPE = '#009E73'; C_LBL = '#E69F00'
C_RES = '#56B4E9'
k = 0.1791                      # GPa per (meV/atom), dP_t/d(Delta E)
rows = {r['T']: r for r in json.load(open(ROOT + 'pt_line_v6_raw.json'))['rows']}
Tsel = [0, 100, 200, 300, 400]

fig = plt.figure(figsize=(7.2, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.02], wspace=0.30,
                      left=0.072, right=0.985, top=0.90, bottom=0.29)

# ------------------------------------------------------------------ (a)
ax = fig.add_subplot(gs[0, 0])
P = np.linspace(0.0, 2.6, 300)
cmap = plt.get_cmap('viridis')
for i, T in enumerate(Tsel):
    r = rows[T]
    s = (r['Va'] - r['Vg']) / 160.2177 * 1000.0        # meV/atom per GPa
    col = cmap(0.06 + 0.80 * i / (len(Tsel) - 1))
    ax.plot(P, s * (r['Pt'] - P), '-', color=col, lw=1.2, label=r'$%d$ K' % T)
    ax.plot(r['Pt'], 0.0, 'o', ms=3.3, mfc='white', mec=col, mew=0.9, zorder=5)

# 0 K with the zero-point term removed
s0 = (rows[0]['Va'] - rows[0]['Vg']) / 160.2177 * 1000.0
Pt_nz = rows[0]['Pt'] - 0.984 * k
ax.plot(P, s0 * (Pt_nz - P), '--', color=C_ZPE, lw=1.1, label=r'$0$ K, no ZPE')
ax.plot(Pt_nz, 0.0, 's', ms=3.3, mfc='white', mec=C_ZPE, mew=0.9, zorder=5)
ax.annotate('', xy=(rows[0]['Pt'], 2.35), xytext=(Pt_nz, 2.35),
            arrowprops=dict(arrowstyle='<->', color=C_ZPE, lw=0.8))
ax.text(0.5 * (rows[0]['Pt'] + Pt_nz), 3.05, r'$+0.176$ GPa', ha='center',
        fontsize=6.9, color=C_ZPE)
# experimental onset at 300 K
ax.plot(0.210, 0.0, 'D', ms=3.6, color=C_EXP, zorder=6)
ax.annotate('experiment\n(onset, 300 K)', xy=(0.235, -0.3), xytext=(0.07, -6.0),
            fontsize=6.4, color=C_EXP, ha='left', va='top',
            arrowprops=dict(arrowstyle='-', color=C_EXP, lw=0.6, shrinkA=1, shrinkB=2))
ax.annotate(r'$P_t$ (0 K)', xy=(0.53, 0.1), xytext=(0.80, 1.30), fontsize=6.5, color='#333333',
            arrowprops=dict(arrowstyle='-', color='#333333', lw=0.6, shrinkA=0, shrinkB=1))
ax.text(0.03, 0.03, r'$P_t = 0.51,\ 0.61,\ 0.91,\ 1.37,\ 1.78$ GPa' + '\n' + r'for $T = 0$-$400$ K',
        transform=ax.transAxes, fontsize=6.2, color='#666666', va='bottom')
ax.axhline(0, color='#aaaaaa', lw=0.6, zorder=0)
ax.set_xlabel(r'$P$ (GPa)')
ax.set_ylabel(r'$G_\gamma - G_\alpha$ (meV/atom)')
ax.set_xlim(0, 2.6); ax.set_ylim(-14.5, 12.5)
ax.set_yticks([-10, -5, 0, 5, 10])
ax.legend(frameon=False, loc='upper right', handletextpad=0.4, labelspacing=0.22,
          borderpad=0.15, ncol=2, columnspacing=0.8)
ax.text(0.02, 0.97, '(a)', transform=ax.transAxes, va='top', fontsize=9, fontweight='bold')

# ------------------------------------------------------------------ (b)
ax2 = fig.add_subplot(gs[0, 1])
steps = [(r'experiment' + '\n' + r'(onset)', 1.173, C_EXP),
         (r'PBE' + '\n' + r'$\delta_{\rm DFT}$', 6.257, C_PBE),
         (r'label shift' + '\n' + r'$\Delta E_{\rm shift}$', -4.891, C_LBL),
         (r'MLIP' + '\n' + r'$\delta_{\rm MLIP}$', -0.616, C_RES),
         (r'zero-point' + '\n' + r'$\Delta$ZPE', 0.984, C_ZPE)]
cum = 1.173
for i, (name, d, col) in enumerate(steps):
    if i == 0:
        ax2.bar(i, cum * k, width=0.62, color=col, edgecolor='k', linewidth=0.5)
        ax2.text(i, cum * k + 0.03, r'$0.210$', ha='center', fontsize=6.5)
        continue
    bottom, top = cum * k, (cum + d) * k
    ax2.bar(i, abs(top - bottom), bottom=min(bottom, top), width=0.62,
            color=col, edgecolor='k', linewidth=0.5)
    ax2.plot([i - 0.31, i + 0.31], [bottom, bottom], color='k', lw=0.5, ls=':')
    ax2.text(i, max(bottom, top) + 0.035, r'$%+.3f$' % (d * k), ha='center', fontsize=6.5)
    ax2.text(i, 0.045, r'$%+.3f$' % d, ha='center', fontsize=6.0, color='#444444')
    cum += d
ax2.bar(len(steps), cum * k, width=0.62, color='#777777', edgecolor='k', linewidth=0.5)
ax2.text(len(steps), cum * k + 0.035, r'$%.3f$' % (cum * k), ha='center', fontsize=6.5)
ax2.set_xticks(list(range(len(steps))) + [len(steps)])
ax2.set_xticklabels([s[0] for s in steps] + [r'this work' + '\n' + r'(V6)'],
                    fontsize=6.6, rotation=18, ha='right')
ax2.set_ylabel(r'$P_t$ (GPa)')
ax2.set_ylim(0, 1.66); ax2.set_xlim(-0.6, 5.6)
ax2.axhline(0.210, color=C_EXP, lw=0.6, ls='--')
ax2.text(5.45, 0.245, 'experiment', fontsize=6.3, color=C_EXP, ha='right')
ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, va='top', fontsize=9, fontweight='bold')
ax2.text(0.99, 0.965, r'$89\,\%$ data engineering' + '\n' + r'$11\,\%$ model residual',
         transform=ax2.transAxes, ha='right', va='top', fontsize=6.8, color='#333333')
ax2.text(0.015, 0.895, r'upper: GPa' + '\n' + r'lower: meV atom$^{-1}$', transform=ax2.transAxes,
         ha='left', va='top', fontsize=5.8, color='#666666', linespacing=1.3)
fig.savefig(os.path.join(OUT, 'fig5.pdf'))
fig.savefig(os.path.join(OUT, 'fig5.png'), dpi=300)
print('fig5: ledger closes at %.3f GPa (full QHA 0.509); no-ZPE P_t=%.3f' % (cum * k, Pt_nz))
for T in Tsel:
    r = rows[T]; s = (r['Va'] - r['Vg']) / 160.2177 * 1000.0
    print('   T=%3d  dV=%.4f  slope=%.3f meV/GPa  dG(0)=%.2f meV  Pt=%.3f' %
          (T, r['Va'] - r['Vg'], s, s * r['Pt'], r['Pt']))
