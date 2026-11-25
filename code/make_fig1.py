#!/usr/bin/env python3
"""Fig. 1 - near-degeneracy and the response coefficient, present (MACE) potential.

Data sources (all read-only):
  deposit/data/json/eos.json   phases.{alpha,gamma}.{table,BM}, delta
  deposit/data/json/pt_line.json consistency

Panel (a): fixed-cell relaxation points plus the Birch-Murnaghan fit that was made to
them. The gamma table has 13 volumes of which 10 are used in the fit (n_points_used); the
three at V/V0 >= 1.04 are the WO4 -> WO5 reconstruction and are drawn as open symbols,
excluded from the fit and from the curve, but kept visible so that the reconstruction is
shown rather than hidden.
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
GPA_A3_PER_EV = 160.2177

eos = json.load(open(QHA + 'eos.json'))
pt = json.load(open(QHA + 'pt_line.json'))
C = pt['consistency']['coef_variants']['v11_fit_dV']['coef_GPa_per_meV_atom']   # 0.1755
EXP_P = 0.210                                                                   # GPa, onset
EXP_E = EXP_P / C                                                               # 1.1966 meV/atom


def bm_meV(V, BM):
    """Birch-Murnaghan (3rd order) energy in meV/atom above E0; V in A^3/atom."""
    eta = (BM['V0_A3_per_atom'] / V) ** (2.0 / 3.0)
    eV = 9.0 * BM['V0_A3_per_atom'] * BM['B0_GPa'] / 16.0 * (
        (eta - 1.0) ** 3 * BM['B0p'] + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)) * 0.0062415
    return eV * 1000.0


fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(1, 2, wspace=0.27, left=0.075, right=0.985, top=0.93, bottom=0.20)

# ------------------------------------------------ (a) E(V) of the two phases
ax = fig.add_subplot(gs[0, 0])
xwin = (16.65, 18.45)
for ph, c, lab, mk in (('alpha', C_A, r'$\alpha$', 'o'), ('gamma', C_G, r'$\gamma$', 's')):
    BM = eos['phases'][ph]['BM']
    E0 = BM['E0_eV_per_atom'] * 1000.0
    t = eos['phases'][ph]['table']
    V = np.array([r['V_at'] for r in t], float)
    dE = np.array([r['E_meV_at'] for r in t], float) - E0
    good = np.array([not r['restructured'] for r in t], bool)
    Vf = np.linspace(*xwin, 400)
    ax.plot(Vf, bm_meV(Vf, BM), '-', color=c, lw=1.1, label=lab)
    ax.plot(V[good], dE[good], mk, ms=3.4, color=c, mec=c, mew=0.8, zorder=3)
    if (~good).any():
        idx = np.argsort(V[~good])
        Vs, Es = V[~good][idx], dE[~good][idx]
        ax.plot(Vs, Es, mk, ms=3.8, mfc='none', mec=c, mew=0.9, ls='none', zorder=4)
        ax.plot(Vs, Es, ':', color=c, lw=0.9, zorder=2)

V0a = eos['phases']['alpha']['BM']['V0_A3_per_atom']
V0g = eos['phases']['gamma']['BM']['V0_A3_per_atom']
ax.set_xlabel(r'$V$ ($\mathrm{\AA^3}$/atom)')
ax.set_ylabel(r'$E_{\mathrm{st}} - E_0$ (meV/atom)')
ax.set_xlim(*xwin); ax.set_ylim(-2.5, 8.6)
ax.axvline(V0a, color=C_A, lw=0.6, ls=':'); ax.axvline(V0g, color=C_G, lw=0.6, ls=':')
ax.annotate('', xy=(V0g, 6.9), xytext=(V0a, 6.9),
            arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.3'))
ax.text(0.5 * (V0a + V0g), 7.25, r'$\Delta V/V = -5.06\,\%$', ha='center', fontsize=7)
ax.text(17.98, -2.05, 'WO$_4\\to$WO$_5$ reconstruction', fontsize=6.4, color='0.3',
        ha='center', va='center')
ax.legend(frameon=False, loc='upper left', handletextpad=0.4)
ax.set_title('(a)', loc='left', fontsize=8, fontweight='bold')

# ------------------------------------------------ (b) energy -> pressure axis
ax = fig.add_subplot(gs[0, 1])
dEst = eos['delta']['dEst_meV_per_atom']                     # 2.7064
dG = pt['consistency']['sum_meV_at']                         # 3.4971
pts = [(EXP_E, 'experiment\n(onset, $\\leq$1.20)', C_EXP, 'D'),
       (2.5247, 'PBE, relaxed\n(2.52 $\\pm$ 0.20)', '#D55E00', 's'),
       (dEst, 'MACE (EOS)', C_A, 'o'),
       (dG, 'MACE + ZPE', '#009E73', '^')]
for x, lab, c, m in pts:
    ax.plot([0, x], [x * C, x * C], color=c, lw=0.6, ls=':')
    ax.plot(x, x * C, marker=m, ms=4.5, color=c, label=lab)
ax.plot([0, 4.0], [0, 4.0 * C], color='0.5', lw=0.9)
ax.set_xlim(0, 4.0); ax.set_ylim(0, 0.72)
ax.set_xlabel(r'$\Delta G(0\,\mathrm{K}) = \Delta E_{\mathrm{st}} + \Delta$ZPE (meV/atom)')
ax.set_ylabel(r'$P_t$ (GPa)')
ax.text(0.09, 0.66, r'slope $=%.4f$ GPa/(meV/atom)' % C, fontsize=7, color='0.25')
ax.annotate('', xy=(dG, 0.11), xytext=(dG, dG * C),
            arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.3'))
ax.text(dG - 0.08, 0.36, '$3.0\\times$', ha='right', fontsize=7, rotation=90)
ax.legend(frameon=False, loc='lower right', handletextpad=0.3, labelspacing=0.25)
ax.set_title('(b)', loc='left', fontsize=8, fontweight='bold')

fig.savefig(os.path.join(OUT, 'fig1.pdf'))
print('fig1.pdf written; C = %.6f, EXP_E = %.4f, dEst = %.4f, dG = %.4f' % (C, EXP_E, dEst, dG))
n_a = eos['phases']['alpha']['n_points_used']
n_g = eos['phases']['gamma']['n_points_used']
print('points used in the fits: alpha %d, gamma %d (gamma total %d)'
      % (n_a, n_g, len(eos['phases']['gamma']['table'])))
