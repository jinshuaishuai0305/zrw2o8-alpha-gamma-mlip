#!/usr/bin/env python3
"""Fig. 3 - Phonon DOS and the band-resolved zero-point energy mechanism."""
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
C_A = '#0072B2'; C_G = '#D55E00'; C_POS = '#009E73'; C_NEG = '#CC79A7'
THz2meV = 4.135667
ph = json.load(open(ROOT + 'phdos_zpe.json'))
a, g = ph['alpha'], ph['gamma']
fg = np.array(a['dos_grid'])
da = np.array(a['dos']); dg = np.array(g['dos'])      # both integrate to 3 states/THz/atom
df = fg[1] - fg[0]

fig = plt.figure(figsize=(7.2, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.0, 1.0], wspace=0.36,
                      left=0.068, right=0.985, top=0.90, bottom=0.22)

# ---- (a) PHDOS
ax = fig.add_subplot(gs[0, 0])
for x in (4, 8, 16):
    ax.axvline(x, color='#c8c8c8', lw=0.6, ls=':')
def smooth(y, sig=0.25):
    n = int(4 * sig / df) | 1
    x = (np.arange(n) - n // 2) * df
    w = np.exp(-0.5 * (x / sig) ** 2); w /= w.sum()
    return np.convolve(y, w, mode='same')
da_s, dg_s = smooth(da), smooth(dg)
ax.fill_between(fg, 0, dg_s, color=C_G, alpha=0.14, lw=0, zorder=1)
ax.fill_between(fg, 0, da_s, color=C_A, alpha=0.14, lw=0, zorder=1)
ax.plot(fg, dg_s, '-', color=C_G, lw=0.9, label=r'$\gamma$', zorder=3)
ax.plot(fg, da_s, '-', color=C_A, lw=0.9, label=r'$\alpha$', zorder=4)
ax.set_xlim(0, 32); ax.set_ylim(0, 0.66)
ax.set_xlabel(r'Frequency (THz)'); ax.set_ylabel(r'PHDOS (states THz$^{-1}$ atom$^{-1}$)')
ax.legend(frameon=False, loc='upper right', handletextpad=0.4, labelspacing=0.2,
          borderpad=0.1)
ax.text(0.03, 0.96, '(a)', transform=ax.transAxes, va='top', fontsize=9, fontweight='bold')

# ---- (b) band-resolved Delta ZPE
ax2 = fig.add_subplot(gs[0, 1])
bands = [(r'0-4', 0.0, 4.0), (r'4-8', 4.0, 8.0), (r'8-16', 8.0, 16.0), (r'$>$16', 16.0, 40.0)]
def band_zpe(d, lo, hi):
    s = 0.0
    for b in d['bands']:
        bh = 40.0 if b['hi'] is None else b['hi']
        ov = max(0.0, min(bh, hi) - max(b['lo'], lo))
        s += b['zpe'] * ov / (bh - b['lo'])
    return s
vals = [band_zpe(g, lo, hi) - band_zpe(a, lo, hi) for _, lo, hi in bands]
tot = sum(vals)
for i, v in enumerate(vals):
    ax2.bar(i, v, width=0.6, color=(C_POS if v > 0 else C_NEG), edgecolor='k', linewidth=0.5)
    ax2.text(i, v + (0.08 if v > 0 else -0.10), r'$%+.2f$' % v, ha='center', fontsize=6.5,
             va='bottom' if v > 0 else 'top')
ax2.axhline(0, color='#888888', lw=0.6)
ax2.axhline(tot, color='#333333', lw=0.7, ls='--')
ax2.text(3.45, tot + 0.07, r'net $+0.99$', fontsize=6.4, ha='right', color='#333333')
ax2.set_xticks(range(len(bands))); ax2.set_xticklabels([b[0] for b in bands], fontsize=7)
ax2.set_xlabel(r'Frequency band (THz)'); ax2.set_ylabel(r'$\Delta$ZPE (meV/atom)')
ax2.set_ylim(-1.55, 2.05)
ax2.text(0.03, 0.96, '(b)', transform=ax2.transAxes, va='top', fontsize=9, fontweight='bold')

# ---- (c) cumulative Delta ZPE
ax3 = fig.add_subplot(gs[0, 2])
# staircase built from the archived band table (1 THz bins below 2 THz)
bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 40.0)]
xs, ys = [0.0], [0.0]
run = 0.0
for lo, hi in bins:
    run += band_zpe(g, lo, hi) - band_zpe(a, lo, hi)
    xs += [lo, hi]; ys += [run, run]
ax3.plot(xs, ys, '-', color='#333333', lw=1.2)
ax3.plot(xs[1:], ys[1:], '.', ms=2.2, color='#333333')
for x in (4, 8, 16):
    ax3.axvline(x, color='#c8c8c8', lw=0.6, ls=':')
ax3.axhline(0, color='#bbbbbb', lw=0.6)
ax3.annotate(r'$+0.99$', xy=(28.0, run), xytext=(17.5, 1.35),
             fontsize=6.8, arrowprops=dict(arrowstyle='-', lw=0.6, color='#333333'))
ax3.annotate(r'$-0.35$', xy=(4.0, ys[xs.index(4.0) + 1]), xytext=(0.6, -1.6),
             fontsize=6.8, arrowprops=dict(arrowstyle='-', lw=0.6, color='#333333'))
ax3.set_xlim(0, 32); ax3.set_ylim(-2.2, 2.4)
ax3.set_xlabel(r'Frequency (THz)'); ax3.set_ylabel(r'cumulative $\Delta$ZPE (meV/atom)')
ax3.text(0.03, 0.96, '(c)', transform=ax3.transAxes, va='top', fontsize=9, fontweight='bold')
ax3.text(0.98, 0.05, r'soft modes $<$2 THz:' + '\n' + r'$\alpha$ 8.7 %, $\gamma$ 6.3 %',
         transform=ax3.transAxes, ha='right', va='bottom', fontsize=6.2, color='#555555')

fig.savefig(os.path.join(OUT, 'fig3.pdf'))
fig.savefig(os.path.join(OUT, 'fig3.png'), dpi=300)
print('fig3 bands', ['%+.3f' % v for v in vals], 'total %+.4f' % tot)
print('     staircase end %+.4f (archive net %+.4f)' % (run, g['zpe_total'] - a['zpe_total']))
print('     mean freq alpha %.3f gamma %.3f THz' % (np.sum(fg*da)/np.sum(da), np.sum(fg*dg)/np.sum(dg)))
