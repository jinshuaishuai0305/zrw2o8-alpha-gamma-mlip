#!/usr/bin/env python3
"""γ 相在固定体积下放开晶胞形状 (b/a, c/a) 是否还有能量收益？

背景：归档的 stage2_relax.py 只弛豫离子、晶胞写死；实测 γ 的 5 个体积点
      b/a=1.0159、c/a=3.0288 完全相同 → 形状从未随体积放开。
若最优形状随压缩偏移，G_γ 被抬高，公切线交点被推向高压。

做法：以归档弛豫胞为起点，对每个体积点做 3×3 二维扫描
      (fb, fc) ∈ {1−δ, 1, 1+δ}²，**固定总体积**（a' 由 a'=1/(b'c') 反算，正交晶系下 a·b·c=V），
      每个点只弛豫离子，取 E。中心点即归档值。
"""
import os, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import FIRE
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
ap = argparse.ArgumentParser()
ap.add_argument('--delta', type=float, default=0.01)
ap.add_argument('--vols', default='vol_0985,vol_1000,vol_1015')
ap.add_argument('--src', default='qha_gamma')
a = ap.parse_args()
calc = MACECalculator(model_paths=[ROOT + 'AlZrW_v6_stagetwo.model'],
                      device='cuda', default_dtype='float32')
res = {}
for tag in a.vols.split(','):
    at0 = read('%s/%s/%s/POSCAR_relaxed' % (ROOT, a.src, tag))
    C = at0.cell[:]
    diag = np.diag(C)
    aligned = np.count_nonzero(C) == 3          # a,b,c 是否在坐标轴上
    V = at0.get_volume()
    if aligned:
        a0 = diag[0]; b0 = diag[1]; c0 = diag[2]
    else:                                        # (a, b, c) 在 x,y,z 乱序，按长度排序
        L = np.sort(at0.cell.lengths())
        a0, b0, c0 = L[0], L[1], L[2]
    print('\n%s V=%.4f  aligned=%s  a,b,c = %.4f %.4f %.4f' % (tag, V, aligned, a0, b0, c0), flush=True)
    E = np.zeros((3, 3)); VV = np.zeros((3, 3))
    t0 = time.time()
    for i, fb in enumerate((1 - a.delta, 1.0, 1 + a.delta)):
        for j, fc in enumerate((1 - a.delta, 1.0, 1 + a.delta)):
            bb, cc = b0 * fb, c0 * fc
            aa = V / (bb * cc)                   # 体积守恒
            at = Atoms(symbols=at0.get_chemical_symbols(), positions=at0.get_positions(),
                       cell=np.diag([aa, bb, cc]), pbc=True)
            at.calc = calc
            FIRE(at, logfile=None).run(fmax=1e-4, steps=3000)
            at.calc = calc
            E[i, j] = float(at.get_potential_energy()); VV[i, j] = at.get_volume()
            print('   fb=%.3f fc=%.3f  V=%.3f  E=%.6f eV (%.6f eV/at)  (%.0fs)'
                  % (fb, fc, VV[i, j], E[i, j], E[i, j] / len(at), time.time() - t0), flush=True)
    E0 = E[1, 1]
    print('  中心 E=%.6f eV (%.6f eV/at)' % (E0, E0 / 132))
    print('  ΔE(b−,b+) = %+.3f, %+.3f meV/at' % ((E[1, 0] - E0) * 1000 / 132, (E[1, 2] - E0) * 1000 / 132))
    print('  ΔE(c−,c+) = %+.3f, %+.3f meV/at' % ((E[0, 1] - E0) * 1000 / 132, (E[2, 1] - E0) * 1000 / 132))
    res[tag] = dict(V=float(V), a0=float(a0), b0=float(b0), c0=float(c0), delta=a.delta,
                    E=E.tolist(), dE_b=[float((E[1, 0] - E0) / 132), float((E[1, 2] - E0) / 132)],
                    dE_c=[float((E[0, 1] - E0) / 132), float((E[2, 1] - E0) / 132)])
json.dump(res, open(ROOT + 'diag_precision/json/gamma_shape_relax.json', 'w'), indent=1)
print('\n-> diag_precision/json/gamma_shape_relax.json')
