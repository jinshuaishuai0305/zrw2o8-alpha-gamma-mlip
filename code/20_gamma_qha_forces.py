#!/usr/bin/env python3
"""γ-ZrW2O8：13 个体积点（0.955–1.045 V₀）的声子力 —— V6 势，2×2×1 超胞（528 原子）。

目的：补齐 vol_0955/0965/0975/1025/1035/1045 六个点，使 γ 的 QHA 体积网格
覆盖 QHA 自身的平衡体积，从而能做 G_α(T,P)=G_γ(T,P) 的相变线（含 ZPE）。

协议（写死，全 13 点一致）：
  1. _gamma_ref.vasp 用 spglib(0.05) 对称化成 P2₁2₁2₁，再按 r^(1/3) 缩放到目标体积
  2. MACE V6 固定胞弛豫离子（FIRE, fmax=1e-4 eV/Å），记录 E(V)
  3. phonopy 2×2×1 超胞 + 198 个对称约化位移（0.01 Å）
  4. V6 算每个位移超胞的力 -> FORCE_SETS
  5. 热学量 mesh 8×8×8（γ 正交胞）

用法: python 20_gamma_qha_forces.py [--only 0955,0965] [--mesh 8 8 8]
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
import spglib
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_SETS
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUTDIR = ROOT + 'qha_gamma/'
SP = 0.05
RATIOS = [0.955, 0.965, 0.975, 0.985, 0.990, 0.995, 1.000,
          1.005, 1.010, 1.015, 1.025, 1.035, 1.045]

ap = argparse.ArgumentParser()
ap.add_argument('--only', default='')
ap.add_argument('--mesh', type=int, nargs=3, default=[8, 8, 8])
ap.add_argument('--dtype', default='float32')
a_ = ap.parse_args()
only = set(x.strip() for x in a_.only.split(',') if x.strip())

calc = MACECalculator(model_paths=[ROOT + 'AlZrW_v6_stagetwo.model'],
                      device='cuda', default_dtype=a_.dtype)
print('V6 QHA-γ 声子力：13 个体积点，2x2x1 超胞，mesh %s，dtype %s'
      % (a_.mesh, a_.dtype), flush=True)

# ---- 参考胞对称化 ----
at0 = read(ROOT + '_gamma_ref.vasp')
cell = (at0.cell[:], at0.get_scaled_positions(), at0.numbers)
ds = spglib.get_symmetry_dataset(cell, symprec=SP)
lat, pos, num = spglib.refine_cell(cell, symprec=SP)
sym = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
V_ref = sym.get_volume()
print('参考: %s (#%d)  V=%.4f Å³  N=%d' % (ds.international, ds.number, V_ref, len(sym)), flush=True)

summary, relax_rec = [], []
t_all = time.time()
for r in RATIOS:
    tag = 'vol_%04d' % int(round(r * 1000))
    if only and tag.replace('vol_', '') not in only:
        continue
    d = OUTDIR + tag
    os.makedirs(d, exist_ok=True)
    t0 = time.time()

    # 1) 缩放到目标体积
    s = r ** (1 / 3)
    a = sym.copy()
    a.set_cell(sym.cell[:] * s, scale_atoms=False)
    a.calc = calc
    write(d + '/POSCAR', a, format='vasp', direct=True)

    # 2) 固定胞弛豫离子
    FIRE(a, logfile=None).run(fmax=1e-4, steps=5000)
    a.calc = calc
    E = float(a.get_potential_energy())
    fmax = float(np.abs(a.get_forces()).max())
    write(d + '/POSCAR_relaxed', a, format='vasp', direct=True)
    relax_rec.append(dict(dir=tag, ratio=r, V=float(a.get_volume()), E=E, fmax=fmax))
    print('  %s V=%.3f  E=%.6f (%.6f eV/at)  max|F|=%.1e  (%.0fs)'
          % (tag, a.get_volume(), E, E / len(a), fmax, time.time() - t0), flush=True)

    # 3) 生成位移
    pa = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                      scaled_positions=a.get_scaled_positions())
    ph = Phonopy(pa, supercell_matrix=np.diag([2, 2, 1]), primitive_matrix='auto', symprec=SP)
    ph.generate_displacements(distance=0.01)
    scs = ph.supercells_with_displacements
    print('     位移 %d 个，超胞 %d 原子' % (len(scs), len(ph.supercell)), flush=True)
    try:
        ph.save(d + '/phonopy_disp.yaml')
    except AttributeError:
        ph.write_yaml(d + '/phonopy_disp.yaml')

    # 4) 算力
    forces = []
    for k, sc in enumerate(scs, 1):
        at = Atoms(symbols=list(sc.symbols), cell=np.array(sc.cell),
                   scaled_positions=np.array(sc.scaled_positions), pbc=True)
        at.calc = calc
        forces.append(at.get_forces())
        if k % 50 == 0:
            print('       %d/%d  (%.0fs)' % (k, len(scs), time.time() - t0), flush=True)
    ph.forces = forces
    write_FORCE_SETS(ph.dataset, filename=d + '/FORCE_SETS')

    # 5) 热学量
    ph.produce_force_constants()
    ph.run_mesh(a_.mesh, with_eigenvectors=False)
    fr = np.array(ph.get_mesh_dict()['frequencies'])
    n_imag = int((fr < -1e-3).sum())
    ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
    ph.write_yaml_thermal_properties(filename=d + '/thermal_properties.yaml')
    tp = ph.get_thermal_properties_dict()
    T = np.array(tp['temperatures']); F = np.array(tp['free_energy'])
    i300 = int(np.argmin(abs(T - 300)))
    zpe = float(ph.zero_point_energy) if hasattr(ph, 'zero_point_energy') else None
    summary.append(dict(dir=tag, V=float(a.get_volume()), E=E, n_imag=n_imag,
                        fmin=float(fr.min()), F300=float(F[i300] * 1000 / len(ph.primitive))))
    print('     done %s: 虚频 %d  最低模 %.3f THz  F(300K)=%.4f kJ/mol  (%.0fs)'
          % (tag, n_imag, fr.min(), F[i300], time.time() - t0), flush=True)

json.dump(relax_rec, open(OUTDIR + 'relax.json', 'w'), indent=2)
json.dump(summary, open(OUTDIR + 'phonon_summary.json', 'w'), indent=2)
print('\n总耗时 %.0f s；写出 relax.json / phonon_summary.json' % (time.time() - t_all))
