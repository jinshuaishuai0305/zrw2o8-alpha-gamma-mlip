#!/usr/bin/env python3
"""列出 α-ZrW₂O₈（V6）QHA 到底算出了哪些热力学量，并做一次自洽的重新提取。

数据源：qha/vol_*/ 的 FORCE_SETS + thermal_properties.yaml（phonopy 4.5.0，3×3×3 超胞 1188 原子）
       qha/relax.json 的静态 E(V)
输出：每个体积点的 ZPE / F_vib(T) / S(T) / Cv(T)，以及自洽的 V(T)、α_L(T)、B(T)
"""
import json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np, yaml
from scipy.optimize import curve_fit

ROOT = '/home/jss/share/AlZrW2O8_MACE/qha/'
N = 44

def vinet(V, E0, B0, Bp, V0):
    x = (V / V0) ** (1 / 3.)
    return E0 + 2 * B0 * V0 / (Bp - 1) ** 2 * (2 - (5 + 3 * Bp * (x - 1) - 3 * x) * np.exp(-1.5 * (Bp - 1) * (x - 1)))

summ = json.load(open(ROOT + 'phonon_summary.json'))
relax = {v['dir']: v for v in json.load(open(ROOT + 'relax.json'))}
use = [s for s in summ if s['n_imag'] == 0]

print('=== 1. 每个体积点已有的热力学量（phonopy thermal_properties.yaml）===')
print('%-9s %9s %11s %9s %9s %10s %10s' % ('vol', 'V(Å³)', 'ZPE(kJ/mol)', 'ZPE/at', 'F300/at', 'S300', 'Cv300'))
print('%-9s %9s %11s %9s %9s %10s %10s' % ('', '', '', 'meV/at', 'meV/at', 'J/K/mol', 'J/K/mol'))
tab = {}
for s in use:
    d = s['dir']
    y = yaml.safe_load(open(ROOT + d + '/thermal_properties.yaml'))
    tp = y['thermal_properties']
    T = np.array([e['temperature'] for e in tp])
    F = np.array([e['free_energy'] for e in tp])      # kJ/mol (phonon free energy, incl. ZPE)
    S = np.array([e['entropy'] for e in tp])
    Cv = np.array([e['heat_capacity'] for e in tp])
    i300 = int(np.argmin(abs(T - 300)))
    tab[d] = dict(V=relax[d]['V'], E=relax[d]['E'], T=T, F=F, S=S, Cv=Cv,
                  zpe=y['zero_point_energy'])
    kJ2meV = 1000.0 / 96.485 / N
    print('%-9s %9.2f %11.3f %9.3f %9.3f %10.2f %10.2f' %
          (d, relax[d]['V'], y['zero_point_energy'], y['zero_point_energy'] * kJ2meV,
           F[i300] * kJ2meV, S[i300], Cv[i300]))

print('\n=== 2. 自洽重提取：F(V,T) = E_static(V) + F_vib(V,T)，再对 V 做 Vinet 拟合 ===')
dirs = [s['dir'] for s in use]
Vs = np.array([tab[d]['V'] for d in dirs])
Es = np.array([tab[d]['E'] for d in dirs])
Tgrid = tab[dirs[0]]['T']
kJ2eV = 1.0 / 96.485

VT, BT, ST, CVT, GT, ZPT = [], [], [], [], [], []
for it, T in enumerate(Tgrid):
    Ftot = np.array([Es[k] + tab[d]['F'][it] / 96.485 / N for k, d in enumerate(dirs)])   # eV/atom
    p, _ = curve_fit(vinet, Vs, Ftot, p0=[Ftot.min(), 0.5, 4, Vs[Ftot.argmin()]], maxfev=200000)
    VT.append(p[3]); BT.append(p[1] * 160.21766208)
    GT.append(p[0])   # eV/atom
    ZPT.append(np.mean([tab[d]['zpe'] for d in dirs]) / 96.485 / N)             # eV/atom
    ST.append(np.mean([tab[d]['S'][it] for d in dirs]) / 96.485 / N * 1000)     # meV/at/K
    CVT.append(np.mean([tab[d]['Cv'][it] for d in dirs]) / 96.485 / N * 1000)
VT = np.array(VT); BT = np.array(BT)

print('\n%-8s %10s %10s %10s %13s %13s' % ('T(K)', 'V(Å³)', 'B(GPa)', 'α_L(ppm/K)', 'Cv(meV/at/K)', 'S(meV/at/K)'))
for t in (0, 100, 200, 300, 400, 500, 600, 700):
    i = int(np.argmin(abs(Tgrid - t)))
    a = np.gradient(VT) / np.gradient(Tgrid) / (3 * VT) * 1e6
    print('%-8d %10.3f %10.2f %10.2f %13.4f %13.4f' % (Tgrid[i], VT[i], BT[i], a[i], CVT[i], ST[i]))

a = np.gradient(VT) / np.gradient(Tgrid) / (3 * VT) * 1e6
print('\n=== 3. 分温区 α_L ===')
for lo, hi in [(100, 200), (200, 300), (300, 400), (300, 430), (400, 500), (500, 700)]:
    m = (Tgrid >= lo) & (Tgrid <= hi)
    print('  %3d–%3d K : α_L = %+7.2f ppm/K' % (lo, hi, a[m].mean()))

print('\n=== 4. 绝对热力学量（每个 44 原子胞，含 ZPE 的振动部分 + 静态能 −78808.8984 eV）===')
print('  ZPE = %.4f eV/atom = %.2f meV/atom' % (ZPT[0], ZPT[0] * 1000))
for t in (0, 300, 700):
    i = int(np.argmin(abs(Tgrid - t)))
    Fa = GT[i] * N
    print('  T=%3d K : F = %.4f eV/cell (%.6f eV/atom), G(0 GPa)=同值, S = %.4f meV/(atom K), Cv = %.4f meV/(atom K)'
          % (t, Fa, GT[i], ST[i], CVT[i]))
json.dump(dict(T=Tgrid.tolist(), V=VT.tolist(), B_GPa=BT.tolist(), alpha_ppm=a.tolist(),
               Cv_meV_at_K=CVT, S_meV_at_K=ST, G_eV_cell=GT, ZPE_eV=ZPT[0]),
          open(ROOT + '../diag_precision/json/alpha_thermo_v6.json', 'w'), indent=1)
print('\n写出 diag_precision/json/alpha_thermo_v6.json')
