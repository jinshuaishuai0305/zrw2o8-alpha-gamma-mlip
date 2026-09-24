#!/usr/bin/env python3
"""V6 势函数：α 与 γ-ZrW2O8 的弹性常数 C_ij（应力-应变有限差分，float64 + GPU）。

协议（写死，以便与后续 300 K 结果对比）：
  1. 用 FrechetCellFilter 在给定外压下把晶胞+离子一起弛豫到 fmax
  2. 在这个平衡胞上施加 ±delta 的 6 个 Voigt 应变（scale_atoms=True），
     固定晶胞、只弛豫离子，取 ±两侧应力差
  3. C[:,j] = (σ(+δ) − σ(−δ)) / (2δ)

用法: python 15_elastic_v6.py <alpha|gamma> [delta]
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.units import GPa
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
REF = {'alpha': (ROOT + 'elastic/_alpha_lmps.vasp', 44),
       'gamma': (ROOT + 'elastic/_gamma_lmps.vasp', 132)}
VOIGT_IJ = {3: (1, 2), 4: (0, 2), 5: (0, 1)}
OUT = ROOT + 'diag_precision/json/'

phase = sys.argv[1] if len(sys.argv) > 1 else 'alpha'
delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
DTYPE = os.environ.get('DTYPE', 'float64')

calc = MACECalculator(model_paths=[ROOT + 'AlZrW_v6_stagetwo.model'],
                      device='cuda', default_dtype=DTYPE)
print('phase=%s delta=%.4f dtype=%s' % (phase, delta, DTYPE), flush=True)


def strain_tensor(j, s):
    e = np.zeros((3, 3))
    if j < 3:
        e[j, j] = s
    else:
        i, k = VOIGT_IJ[j]
        e[i, k] = e[k, i] = s / 2.0
    return e


def relax_cell(atoms, fmax=0.002, steps=2000):
    a = atoms.copy(); a.calc = calc
    FIRE(FrechetCellFilter(a, scalar_pressure=0.0), logfile=None).run(fmax=fmax, steps=steps)
    return a


t0 = time.time()
f, n = REF[phase]
at0 = read(f)
at = relax_cell(at0)
print('  relax: V=%.3f A^3 (%.4f/atom, %+.2f%%)  E=%.6f eV/atom  cell=%s  (%.0fs)' %
      (at.get_volume(), at.get_volume() / n, (at.get_volume() / at0.get_volume() - 1) * 100,
       at.get_potential_energy() / n, np.round(at.cell.lengths(), 4), time.time() - t0), flush=True)

base_cell = at.get_cell()
C = np.zeros((6, 6))
for j in range(6):
    sig = {}
    for sign in (+1, -1):
        a = at.copy(); a.calc = calc
        a.set_cell(base_cell @ (np.eye(3) + strain_tensor(j, sign * delta)), scale_atoms=True)
        FIRE(a, logfile=None).run(fmax=0.002, steps=2000)
        sig[sign] = a.get_stress(voigt=True) / GPa
    C[:, j] = (sig[+1] - sig[-1]) / (2.0 * delta)
    print('  strain %d done (%.0fs)' % (j, time.time() - t0), flush=True)

C = 0.5 * (C + C.T)      # 对称化
names = ['C11', 'C22', 'C33', 'C12', 'C13', 'C23', 'C44', 'C55', 'C66']
idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2), (3, 3), (4, 4), (5, 5)]
vals = {nm: float(C[i, j]) for nm, (i, j) in zip(names, idx)}

c11, c22, c33 = C[0, 0], C[1, 1], C[2, 2]
c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]
c44, c55, c66 = C[3, 3], C[4, 4], C[5, 5]
Bv = ((c11 + c22 + c33) + 2 * (c12 + c13 + c23)) / 9.0
Gv = ((c11 + c22 + c33) - (c12 + c13 + c23) + 3 * (c44 + c55 + c66)) / 15.0
# Reuss（正交晶系）
s = np.linalg.inv(C[:3, :3])
Br = 1.0 / np.sum(s)
Gr = 15.0 / (4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
            + 3 * (1 / c44 + 1 / c55 + 1 / c66))
Bh, Gh = (Bv + Br) / 2, (Gv + Gr) / 2
E = 9 * Bh * Gh / (3 * Bh + Gh)
nu = (3 * Bh - 2 * Gh) / (2 * (3 * Bh + Gh))
# 声速（用 Hill 值）
rho = sum(at.get_masses()) / at.get_volume() * 1.66053906660      # g/cm^3 (amu/A^3 -> g/cm^3)
vl = np.sqrt((Bh + 4 * Gh / 3) / rho) * 1e3      # m/s  (GPa/(g/cm^3) -> km/s)
vt = np.sqrt(Gh / rho) * 1e3
born = bool(c11 > 0 and c22 > 0 and c33 > 0 and c44 > 0 and c55 > 0 and c66 > 0
            and c11 + c22 - 2 * c12 > 0 and c11 + c33 - 2 * c13 > 0 and c22 + c33 - 2 * c23 > 0
            and c11 + c22 + c33 + 2 * (c12 + c13 + c23) > 0)
zener = 2 * c44 / (c11 - c12) if abs(c11 - c12) > 1e-9 else None

print('\n  C_ij (GPa, 0 K, 外压 0):')
print('        ' + ' '.join('%8s' % n for n in names))
for i in range(6):
    print('   row%d ' % (i + 1) + ' '.join('%8.1f' % C[i, j] for j in range(6)))
print('  Voigt  B=%.1f G=%.1f | Reuss B=%.1f G=%.1f | Hill B=%.1f G=%.1f GPa'
      % (Bv, Gv, Br, Gr, Bh, Gh))
print('  E=%.1f GPa  nu=%.3f  rho=%.3f g/cm3  v_L=%.0f m/s  v_T=%.0f m/s  Debye~%.0f K'
      % (E, nu, rho, vl, vt, 0))
print('  Born 力学稳定性: %s' % ('通过' if born else '不通过'))
if zener:
    print('  Zener A = %.3f' % zener)

res = dict(phase=phase, natoms=n, delta=delta, dtype=DTYPE,
           V_relaxed=at.get_volume(), E_per_atom=at.get_potential_energy() / n,
           C=C.tolist(), C_ij=vals,
           B_Voigt=Bv, G_Voigt=Gv, B_Reuss=Br, G_Reuss=Gr, B_Hill=Bh, G_Hill=Gh,
           E_GPa=E, nu=nu, rho_g_cm3=rho, v_L_m_s=vl, v_T_m_s=vt,
           born_stable=born, Zener_A=zener)
json.dump(res, open(OUT + 'elastic_v6_%s_0K.json' % phase, 'w'), indent=1)
print('  -> %selastic_v6_%s_0K.json' % (OUT, phase))
