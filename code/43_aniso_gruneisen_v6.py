#!/usr/bin/env python3
"""V6 势函数的 γ 相轴向模 Grüneisen（用于"γ 的 CTE 符号反"这一已知局限的证据链）。

背景：现在报告里引用的 "+17 ~ +25 的正 Grüneisen" 来自 **v5b** 模型，
      而整篇工作用的是 V6 —— 属版本张冠李戴，必须在 V6 上重算。

做法（与 qha_gamma/aniso_run.py 同法，但用 V6 + 本地 GPU）：
  6 个轴向应变态（a/b/c 轴各 ±1%），每个：
    固定胞离子弛豫 -> 2×2×1 超胞 + 198 位移 -> V6 算力 -> FC -> mesh 8³ 频率
  轴向模 Grüneisen：γ_i = −d ln ω_i / d ε_j = −ln(ω_i^+/ω_i^−)/(2ε)

用法: python 43_aniso_gruneisen_v6.py
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_SETS
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'diag_precision/aniso_v6/'
os.makedirs(OUT, exist_ok=True)
DIM = [3, 3, 3]; MESH = [8, 8, 8]; SP = 0.05; DISP = 0.01; EPS = 0.01
DTYPE = os.environ.get("DTYPE", "float32")

calc = MACECalculator(model_paths=ROOT + 'AlZrW_v6_stagetwo.model',
                      device='cuda', default_dtype=DTYPE)
base = read(ROOT + 'elastic/_gamma_lmps.vasp')
print('%s  dtype=%s  V=%.3f' % (os.path.basename(ROOT + 'elastic/_gamma_lmps.vasp'),
                                DTYPE, base.get_volume()), flush=True)

W = {}
for j, ax in enumerate('abc'):
    for sgn, tag in ((+1, ax + 'p'), (-1, ax + 'm')):
        t0 = time.time()
        e = np.zeros(3); e[j] = sgn * EPS
        a = base.copy()
        a.set_cell(base.cell[:] @ np.diag(1 + e), scale_atoms=True)
        a.calc = calc
        FIRE(a, logfile=None).run(fmax=1e-5 if DTYPE == 'float64' else 1e-4, steps=8000)
        a.calc = calc
        write('%s/%s_rel.vasp' % (OUT, tag), a, format='vasp', direct=True)

        pa = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                          scaled_positions=a.get_scaled_positions())
        ph = Phonopy(pa, supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=SP)
        ph.generate_displacements(distance=DISP)
        forces = []
        for sc in ph.supercells_with_displacements:
            at = Atoms(symbols=list(sc.symbols), cell=np.array(sc.cell),
                       scaled_positions=np.array(sc.scaled_positions), pbc=True)
            at.calc = calc
            forces.append(at.get_forces())
        ph.forces = forces
        write_FORCE_SETS(ph.dataset, filename='%s/%s_FORCE_SETS' % (OUT, tag))
        ph.produce_force_constants()
        ph.run_mesh(MESH, with_eigenvectors=False)
        w = np.array(ph.get_mesh_dict()['frequencies']).ravel()
        W[tag] = w
        print('  %s: V=%.2f  频率 %d 个  最低 %+.4f THz  虚频 %d  (%.0fs)'
              % (tag, a.get_volume(), w.size, w.min(), int((w < -1e-3).sum()), time.time() - t0),
              flush=True)

np.savez(OUT + 'aniso_freqs_v6.npz', **W)
res = {}
print('\n=== 轴向模 Grüneisen  γ_i = −ln(ω⁺/ω⁻)/(2ε) ===')
for ax in 'abc':
    wp, wm = W[ax + 'p'], W[ax + 'm']
    ok = (wp > 1e-3) & (wm > 1e-3)
    g = -np.log(wp[ok] / wm[ok]) / (2 * EPS)
    freq = W[ax + 'p'][ok]
    res[ax] = dict(gamma_mean=float(g.mean()), gamma_std=float(g.std()),
                   n_modes=int(ok.sum()),
                   gamma_low=float(g[freq < 1.0].mean()) if (freq < 1.0).any() else None,
                   gamma_rum=float(g[(freq >= 0.2) & (freq < 0.8)].mean())
                   if ((freq >= 0.2) & (freq < 0.8)).any() else None,
                   n_rum=int(((freq >= 0.2) & (freq < 0.8)).sum()))
    print('  沿 %s 轴: <γ>=%+.3f (±%.3f, %d 个模)  |  <1 THz 段 %+.3f  |  0.2–0.8 THz(RUM) %+.3f (%d 个模)'
          % (ax, g.mean(), g.std(), ok.sum(),
             res[ax]['gamma_low'] if res[ax]['gamma_low'] is not None else float('nan'),
             res[ax]['gamma_rum'] if res[ax]['gamma_rum'] is not None else float('nan'),
             res[ax]['n_rum']))
json.dump(dict(dtype=DTYPE, eps=EPS, mesh=MESH, per_axis=res), 
          open(ROOT + 'diag_precision/json/aniso_gruneisen_v6.json', 'w'), indent=1)
print('\n-> diag_precision/json/aniso_gruneisen_v6.json')
