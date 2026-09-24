#!/usr/bin/env python3
"""V6：α / γ-ZrW2O8 在给定（有限温度）平衡体积下的弹性常数。

物理约定：固体的 C_ij(T) 主要由热膨胀带来的体积变化决定（准谐近似），
所以这里在 **V(T) 冻结** 的胞上做应力-应变，得到 C_ij(T) 的准谐值。

用法: python 16_elastic_v6_atV.py <alpha|gamma> <Vtot> [delta]
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.units import GPa
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
REF = {'alpha': (ROOT + 'elastic/_alpha_lmps.vasp', 44, 790.30),
       'gamma': (ROOT + 'elastic/_gamma_lmps.vasp', 132, 2279.42)}
VOIGT_IJ = {3: (1, 2), 4: (0, 2), 5: (0, 1)}
OUT = ROOT + 'diag_precision/json/'

phase = sys.argv[1]
Vtot = float(sys.argv[2]) if len(sys.argv) > 2 else REF[phase][2]
delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
tag = os.environ.get('TAG', '300K')
calc = MACECalculator(model_paths=[ROOT + 'AlZrW_v6_stagetwo.model'],
                      device='cuda', default_dtype=os.environ.get('DTYPE', 'float64'))
f, n, _ = REF[phase]

def strain_tensor(j, s):
    e = np.zeros((3, 3))
    if j < 3:
        e[j, j] = s
    else:
        i, k = VOIGT_IJ[j]
        e[i, k] = e[k, i] = s / 2.0
    return e

t0 = time.time()
at = read(f)
at.set_cell(at.cell[:] * (Vtot / at.get_volume()) ** (1 / 3.), scale_atoms=True)
at.calc = calc
FIRE(at, logfile=None).run(fmax=0.002, steps=3000)     # 固定胞、只弛豫离子
print('phase=%s V_target=%.2f -> V=%.3f A^3 (%.4f/atom)  E=%.6f eV/atom  (%.0fs)'
      % (phase, Vtot, at.get_volume(), at.get_volume() / n, at.get_potential_energy() / n, time.time() - t0), flush=True)

base_cell = at.get_cell()
C = np.zeros((6, 6))
for j in range(6):
    sig = {}
    for sign in (+1, -1):
        a = at.copy(); a.calc = calc
        a.set_cell(base_cell @ (np.eye(3) + strain_tensor(j, sign * delta)), scale_atoms=True)
        FIRE(a, logfile=None).run(fmax=0.002, steps=3000)
        sig[sign] = a.get_stress(voigt=True) / GPa
    C[:, j] = (sig[+1] - sig[-1]) / (2.0 * delta)
    print('  strain %d done (%.0fs)' % (j, time.time() - t0), flush=True)
C = 0.5 * (C + C.T)

names = ['C11', 'C22', 'C33', 'C12', 'C13', 'C23', 'C44', 'C55', 'C66']
idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2), (3, 3), (4, 4), (5, 5)]
vals = {nm: float(C[i, j]) for nm, (i, j) in zip(names, idx)}
c11, c22, c33 = C[0, 0], C[1, 1], C[2, 2]
c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]
c44, c55, c66 = C[3, 3], C[4, 4], C[5, 5]
Bv = ((c11 + c22 + c33) + 2 * (c12 + c13 + c23)) / 9.0
Gv = ((c11 + c22 + c33) - (c12 + c13 + c23) + 3 * (c44 + c55 + c66)) / 15.0
s = np.linalg.inv(C[:3, :3])
Br = 1.0 / np.sum(s)
Gr = 15.0 / (4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
            + 3 * (1 / c44 + 1 / c55 + 1 / c66))
Bh, Gh = (Bv + Br) / 2, (Gv + Gr) / 2
E = 9 * Bh * Gh / (3 * Bh + Gh)
nu = (3 * Bh - 2 * Gh) / (2 * (3 * Bh + Gh))
rho = sum(at.get_masses()) / at.get_volume() * 1.66053906660
print('\n  C_ij (GPa, V = %s 平衡体积):' % tag)
print('        ' + ' '.join('%8s' % nm for nm in names))
for i in range(6):
    print('   row%d ' % (i + 1) + ' '.join('%8.1f' % C[i, j] for j in range(6)))
print('  Hill: B=%.1f  G=%.1f GPa | Voigt B=%.1f G=%.1f | Reuss B=%.1f G=%.1f'
      % (Bh, Gh, Bv, Gv, Br, Gr))
print('  E=%.1f GPa  nu=%.3f  rho=%.3f g/cm3' % (E, nu, rho))
json.dump(dict(phase=phase, tag=tag, V=at.get_volume(), natoms=n, delta=delta,
               C=C.tolist(), C_ij=vals, B_Hill=Bh, G_Hill=Gh, B_Voigt=Bv, G_Voigt=Gv,
               B_Reuss=Br, G_Reuss=Gr, E_GPa=E, nu=nu, rho_g_cm3=rho),
          open(OUT + 'elastic_v6_%s_%s.json' % (phase, tag), 'w'), indent=1)
print('  -> %selastic_v6_%s_%s.json' % (OUT, phase, tag))
