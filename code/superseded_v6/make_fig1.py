#!/usr/bin/env python3
"""Fig. 1 — Motivated example: near-degeneracy of the alpha/gamma transition.

(a) ball-and-stick rendering of both phases (matplotlib 3D; fully colour-controlled)
(b) static equation of state E(V) for both phases, with the volume collapse marked
(c) sensitivity ruler: the same energy axis mapped to coexistence pressure

Data: diag_precision/json/*.json + _alpha_ref.vasp / _gamma_ref.vasp  (no new calculation)
Output: fig1.pdf (vector) + fig1.png (preview)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import CubicSpline

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- typography
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.1,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_EXP = '#000000'; C_MLIP = '#0072B2'; C_PBE = '#D55E00'; C_ZPE = '#009E73'
C_LBL = '#E69F00'
EL = {'Zr': ('#2E3D8F', 0.95), 'W': ('#E59F00', 0.85), 'O': ('#BFBFBF', 0.42)}

# ----------------------------------------------------------------------- data
ev = json.load(open(ROOT + 'diag_precision/json/ev_for_figures.json'))
rows = ev['alpha'] + ev['gamma']

st = {}
for tag in ('alpha', 'gamma'):
    lines = open(ROOT + ('_alpha_ref.vasp' if tag == 'alpha' else '_gamma_ref.vasp')).read().split('\n')
    scale = float(lines[1]); cell = np.array([[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)]) * scale
    syms = lines[5].split(); counts = [int(x) for x in lines[6].split()]
    n = sum(counts); start = 8
    frac = np.array([[float(x) for x in lines[start + i].split()[:3]] for i in range(n)])
    labels = []
    for s, c in zip(syms, counts):
        labels += [s] * c
    st[tag] = dict(cell=cell, frac=frac, labels=labels, cart=frac @ cell)

V_A = [r for r in rows if r['tag'].startswith('vol') and r['n'] == 44 and r['tag'] in
       [x['tag'] for x in ev['alpha']]]
V_G = ev['gamma']

# =====================================================================  FIGURE
fig = plt.figure(figsize=(7.0, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 1.0], wspace=0.42,
                      left=0.055, right=0.985, top=0.86, bottom=0.16)

# ------------------------------------------------- (a) two structures, 3D
def draw_crystal(ax, tag, title):
    d = st[tag]; cart = d['cart']; lab = d['labels']
    cell = d['cell']
    # replicate neighbours so that the cell is filled
    reps = []
    rng = range(-1, 2) if tag == 'gamma' else range(-1, 2)
    for i in rng:
        for j in rng:
            for k in rng:
                shift = i * cell[0] + j * cell[1] + k * cell[2]
                reps.append(cart + shift)
    P = np.vstack(reps); L = lab * len(reps)
    # keep a spherical region for a clean look
    c0 = cell.sum(0) * 0.5
    keep = np.linalg.norm(P - c0, axis=1) < 0.62 * np.linalg.norm(cell, axis=1).mean()
    P = P[keep]; L = [l for l, k in zip(L, keep) if k]
    # bonds: W-O < 2.0, Zr-O < 2.3
    pairs = []
    idxO = [i for i, l in enumerate(L) if l == 'O']
    for i, l in enumerate(L):
        if l == 'O':
            continue
        cut = 2.0 if l == 'W' else 2.35
        for j in idxO:
            if np.linalg.norm(P[i] - P[j]) < cut:
                pairs.append((i, j))
    for i, j in pairs:
        ax.plot(*zip(P[i], P[j]), color='#9a9a9a', lw=0.35, zorder=1)
    order = {'O': 0, 'W': 1, 'Zr': 2}
    for el in ('O', 'W', 'Zr'):
        m = np.array([l == el for l in L])
        if not m.any():
            continue
        col, rad = EL[el]
        ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=(rad * 26) ** 2 * 0.42,
                   c=col, edgecolors='#333333', linewidths=0.25,
                   depthshade=True, zorder=2 + order[el])
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=14, azim=-58)
    lo = P.min(0); hi = P.max(0); mid = (lo + hi) / 2; rr = (hi - lo).max() / 2
    ax.set_xlim(mid[0] - rr, mid[0] + rr); ax.set_ylim(mid[1] - rr, mid[1] + rr)
    ax.set_zlim(mid[2] - rr, mid[2] + rr)
    ax.set_title(title, pad=1.5, fontsize=8)

ax_a = fig.add_subplot(gs[0, 0], projection='3d')
# draw gamma as the "collapsed" phase; alpha separately is expensive -> draw side by side
draw_crystal(ax_a, 'alpha', r'$\alpha$-ZrW$_2$O$_8$  (P2$_1$3)')
ax_a.text2D(0.5, -0.02, r'$V_0=18.041$ Å$^3$/atom', transform=ax_a.transAxes,
            ha='center', va='top', fontsize=7.2)
ax_a.text2D(0.02, 0.98, '(a)', transform=ax_a.transAxes, va='top', fontsize=9, fontweight='bold')
ax_a.legend(handles=[Line2D([], [], marker='o', ls='', color=EL['Zr'][0], ms=4.2, label='Zr'),
                     Line2D([], [], marker='o', ls='', color=EL['W'][0], ms=3.8, label='W'),
                     Line2D([], [], marker='o', ls='', color=EL['O'][0], ms=2.6, label='O')],
            loc='lower left', frameon=False, ncol=3, handletextpad=0.15, columnspacing=0.6,
            bbox_to_anchor=(-0.02, 0.0))

# ------------------------------------------------------------ (b) E(V) curves
ax_b = fig.add_subplot(gs[0, 1])
Va = np.array([r['V'] / 44 for r in V_A]); Ea = np.array([r['E'] / 44 for r in V_A]) * 1000
Vg = np.array([r['V'] / 132 for r in V_G]); Eg = np.array([r['E'] / 132 for r in V_G]) * 1000
# each phase referenced to its own minimum so both curvatures are visible
Ea = Ea - Ea.min(); Eg = Eg - Eg.min()
sa = CubicSpline(Va, Ea); sg = CubicSpline(Vg, Eg)
xa = np.linspace(Va.min(), Va.max(), 300); xg = np.linspace(Vg.min(), Vg.max(), 300)
ax_b.plot(xa, sa(xa), '-', color=C_MLIP, lw=1.2)
ax_b.plot(xg, sg(xg), '-', color=C_PBE, lw=1.2)
ax_b.plot(Va, Ea, 'o', color=C_MLIP, ms=2.6, mec='w', mew=0.3)
ax_b.plot(Vg, Eg, 's', color=C_PBE, ms=2.6, mec='w', mew=0.3)
V0a = Va[np.argmin(Ea)]; V0g = Vg[np.argmin(Eg)]
ax_b.axvline(V0a, color=C_MLIP, lw=0.6, ls=':')
ax_b.axvline(V0g, color=C_PBE, lw=0.6, ls=':')
ax_b.annotate('', xy=(V0g, 5.35), xytext=(V0a, 5.35),
              arrowprops=dict(arrowstyle='<->', color='k', lw=0.7))
ax_b.text((V0a + V0g) / 2, 5.55, r'$\Delta V/V = -4.96\ \%$', ha='center', fontsize=7.2)
ax_b.set_xlabel(r'$V$ (Å$^3$/atom)')
ax_b.set_ylabel(r'$E - E_{\min}$ (meV/atom)')
ax_b.set_ylim(-0.4, 7.6); ax_b.set_xlim(16.3, 19.9)
ax_b.set_yticks([0, 2, 4, 6])
ax_b.text(0.97, 0.03, '(b)', transform=ax_b.transAxes, ha='right', va='bottom', fontsize=9,
          fontweight='bold')
ax_b.legend(handles=[Line2D([], [], color=C_MLIP, marker='o', ms=2.8, label=r'$\alpha$ (P2$_1$3)'),
                     Line2D([], [], color=C_PBE, marker='s', ms=2.8, label=r'$\gamma$ (P2$_1$2$_1$2$_1$)')],
            loc='upper left', frameon=False, bbox_to_anchor=(-0.01, 1.04),
            handletextpad=0.4, labelspacing=0.25)
ax_b.text(0.98, 0.30, 'each curve referenced\nto its own minimum',
          transform=ax_b.transAxes, fontsize=6.0, color='#666666', ha='right', va='bottom')

# ------------------------------------------------------ (c) sensitivity ruler
ax_c = fig.add_subplot(gs[0, 2])
k = 0.1791
dE = np.array([1.173, 2.539, 1.923, 2.907, 7.430])
pts = [('experiment\n(onset)', 1.173, C_EXP, 'o', 'k'),
       ('label-\nshifted PBE', 2.539, C_LBL, '^', C_LBL),
       ('this work\n(V6)', 1.923, C_MLIP, 'o', C_MLIP),
       ('+ $\\Delta$ZPE', 2.907, C_ZPE, 's', C_ZPE),
       ('PBE\nreference', 7.430, C_PBE, '^', C_PBE)]
xs = np.linspace(0, 8.2, 50)
ax_c.plot(xs, xs * k, '-', color='#777777', lw=0.9, zorder=1)
for name, e, col, mk, mec in pts:
    ax_c.plot(e, e * k, mk, ms=5.2, mfc='white' if name.startswith('this') else col,
              mec=mec, mew=1.0, zorder=3)
lab_xy = {'experiment\n(onset)': (1.20, -0.52), 'label-\nshifted PBE': (3.45, -0.52),
          'this work\n(V6)': (1.55, 0.34), '+ $\\Delta$ZPE': (4.55, 0.46),
          'PBE\nreference': (6.2, -0.52)}
for name, e, col, mk, mec in pts:
    dx, dy = lab_xy[name]
    ax_c.annotate(name, (e, e * k), xytext=(dx, e * k + dy), fontsize=6.8, ha='center',
                  va='center', color=mec,
                  arrowprops=dict(arrowstyle='-', color=mec, lw=0.5, shrinkA=0, shrinkB=3))
ax_c.set_xlabel(r'$\Delta E(\gamma-\alpha)$ (meV/atom)')
ax_c.set_ylabel(r'$P_t$ (GPa)', color='#333333')
ax_c.text(0.035, 0.90, r'slope $=\ 0.1791$ GPa' + '\n' + r'per (meV/atom)',
          transform=ax_c.transAxes, ha='left', va='top', fontsize=6.6, color='#333333',
          linespacing=1.25)
ax_c.set_xlim(0, 8.2); ax_c.set_ylim(-0.60, 1.85)
ax_c.text(0.02, 0.97, '(c)', transform=ax_c.transAxes, va='top', fontsize=9, fontweight='bold')

fig.savefig(os.path.join(OUT, 'fig1.pdf'))
fig.savefig(os.path.join(OUT, 'fig1.png'), dpi=300)
print('-> fig1.pdf / fig1.png')
print('V0(alpha)=%.3f  V0(gamma)=%.3f  collapse=%.2f %%' %
      (V0a, V0g, 100 * (V0g - V0a) / V0a))
print('pressure check: 1.173 meV/at -> %.3f GPa ; 7.430 -> %.3f GPa'
      % (1.173 * k, 7.430 * k))
