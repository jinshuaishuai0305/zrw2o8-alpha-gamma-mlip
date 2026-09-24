#!/usr/bin/env python3
"""V6 的 0 K 相稳定性独立复核（float64 + 自由弛豫 + 固定体积 EOS）。
不用 ΔE/ΔV 的一阶构造，而是在两相各自的 E(V) 上求公切线（等压共存），
再报 0 K 的 ΔE(V0)、ΔV/V0、以及共存压力 P_eq。

注意：原 eos_probe/eos2.py 用的是 default_dtype='float32'，
本脚本改用 float64（CUDA 上 float64 与 CPU float64 结果一致，见 diag_precision）。
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from mace.calculators import MACECalculator
from scipy.optimize import curve_fit, fsolve

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
model = sys.argv[1] if len(sys.argv) > 1 else ROOT + 'AlZrW_v6_stagetwo.model'
TAG = os.path.basename(model).replace('.model', '')
DTYPE = os.environ.get('DTYPE', 'float64')
DEV = os.environ.get('DEV', 'cuda')
calc = MACECalculator(model_paths=[model], device=DEV, default_dtype=DTYPE)
print('model=%s dtype=%s device=%s' % (TAG, DTYPE, DEV), flush=True)

ref = {'a': (ROOT + 'elastic/_alpha_lmps.vasp', 44),
       'g': (ROOT + 'elastic/_gamma_lmps.vasp', 132)}

def relax_free(at, fmax=0.005, steps=2000):
    at.calc = calc
    FIRE(FrechetCellFilter(at, scalar_pressure=0.0), logfile=None).run(fmax=fmax, steps=steps)
    return at

def relax_fixedV(at, Vtot, fmax=0.005, steps=2000):
    at.calc = calc
    at.set_cell(at.cell[:] * (Vtot / at.get_volume()) ** (1 / 3.), scale_atoms=True)
    f = FrechetCellFilter(at, scalar_pressure=0.0, constant_volume=True, hydrostatic_strain=True)
    FIRE(f, logfile=None).run(fmax=fmax, steps=steps)
    return at

def bm(V, E0, B0, Bp, V0):
    x = (V0 / V) ** (2 / 3.)
    return E0 + 9 * V0 * B0 / 16 * ((x - 1) ** 3 * Bp + (x - 1) ** 2 * (6 - 4 * x))

out = {}
for k, (f, n) in ref.items():
    a0 = read(f)
    amin = relax_free(a0.copy())
    Vmin = amin.get_volume()
    print('[%s] %s: free relax V/at=%.5f  E/at=%.6f  cell=%s' %
          (TAG, k, Vmin / n, amin.get_potential_energy() / n, np.round(amin.cell.lengths(), 4)), flush=True)
    rows = []
    for s in np.linspace(0.94, 1.06, 9):
        at = relax_fixedV(read(f), Vmin * s ** 3)
        rows.append((at.get_volume() / n, at.get_potential_energy() / n))
        print('      scale %.3f -> V/at=%.5f  E/at=%.7f' % (s, rows[-1][0], rows[-1][1]), flush=True)
    rows.append((Vmin / n, amin.get_potential_energy() / n))
    rows = sorted(rows)
    V = np.array([r[0] for r in rows]); E = np.array([r[1] for r in rows])
    p, cov = curve_fit(bm, V, E, p0=[E.min(), 0.7, 4.0, V[E.argmin()]], maxfev=100000)
    msg = np.sqrt(np.diag(cov)) / np.abs(p)
    out[k] = dict(rows=[[float(x), float(y)] for x, y in rows],
                  bm=[float(x) for x in p],
                  bm_err=[float(x) for x in msg])
    print('[%s] %s: BM E0=%.6f eV/at  B0=%.1f+-%.1f GPa  Bp=%.2f+-%.2f  V0=%.5f+-%.5f' %
          (TAG, k, p[0], p[1] * 160.2177, msg[1] * p[1] * 160.2177, p[2], msg[2] * p[2], p[3], msg[3] * p[3]), flush=True)

pa = np.array(out['a']['bm']); pg = np.array(out['g']['bm'])
dE = (pg[0] - pa[0]) * 1000.0
dV = pg[3] - pa[3]
print('\n[%s] 0 K 静态结果' % TAG)
print('  V0(a)=%.4f  V0(g)=%.4f A^3/at   dV/at=%+.4f  (%.2f %% 收缩)' % (pa[3], pg[3], dV, 100 * dV / pa[3]))
print('  E0(a)=%.6f  E0(g)=%.6f eV/at   dE(g-a)=%+.3f meV/at' % (pa[0], pg[0], dE))
print('  一阶 P_t = dE/dV = %+.3f GPa   (实验 0.21, DFT/PBE 1.331)' % (dE * 1e-3 / dV * 160.2177))

# 公切线（两相共存压力）
def dbm(p, V):
    h = 1e-5
    return (bm(V + h, *p) - bm(V - h, *p)) / (2 * h)

def eqs(x):
    Va, Vg = x
    return [dbm(pa, Va) - dbm(pg, Vg),
            (bm(Va, *pa) - Va * dbm(pa, Va)) - (bm(Vg, *pg) - Vg * dbm(pg, Vg))]

try:
    sol, info, ier, m = fsolve(eqs, [pa[3], pg[3]], full_output=True)
    P = -dbm(pa, sol[0]) * 160.2177
    print('  公切线: Va=%.4f Vg=%.4f A^3/at  P_eq=%+.4f GPa   ier=%d' % (sol[0], sol[1], P, ier))
except Exception as e:  # noqa: BLE001
    print('  公切线求解失败:', e)

json.dump(out, open(ROOT + 'diag_precision/json/eos_%s_%s.json' % (TAG, DTYPE), 'w'), indent=1)
