#!/usr/bin/env python3
"""γ-ZrW2O8：13 个体积点（0.955–1.045 V₀）的**统一** QHA 数据（V6 势，float64）。

为什么全跑而不是只补 6 点：
  归档的 7 点是集群上用 float64 算的；补的 6 点若用 float32，
  同一势的 E(V) 会出现 0.4–3 meV/atom 的接缝，而相变线要分辨的正是 1 meV/atom。
  α 相也是 float64，两相必须同精度，否则 G_α 与 G_γ 不可比。

协议（全 13 点一致，可复现）：
  1. _gamma_ref.vasp -> spglib(0.05) 对称化 P2₁2₁2₁（与集群脚本同法，含 origin 回移）
  2. 缩放 (r)^(1/3) 到目标体积；V6 **float64** 固定胞弛豫离子（FIRE, fmax=1e-5）
  3. phonopy 2×2×1 超胞（528 原子）+ 198 个对称约化位移（0.01 Å）
  4. V6 **float64** 算力 -> FORCE_SETS
  5. mesh 8×8×8 热学量 -> thermal_properties.yaml；记录 ZPE、虚频数

用法: python 21_gamma_qha_v6_f64.py [--mesh 8 8 8]
"""
import os, json, time, argparse, warnings
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
OUT = ROOT + 'qha_gamma_v6f64/'
SP = 0.05
RATIOS = [0.955, 0.965, 0.975, 0.985, 0.990, 0.995, 1.000,
          1.005, 1.010, 1.015, 1.025, 1.035, 1.045]

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', type=int, nargs=3, default=[8, 8, 8])
ap.add_argument('--fmax', type=float, default=1e-5)
a_ = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

calc = MACECalculator(model_paths=[ROOT + 'AlZrW_v6_stagetwo.model'],
                      device='cuda', default_dtype='float64')
print('V6 / float64 / 13 个体积点 / 2x2x1 超胞 / mesh %s' % a_.mesh, flush=True)

# ---- 参考胞：与集群 qha_gamma/stage1_scale.py 完全相同的对称化流程 ----
at0 = read(ROOT + '_gamma_ref.vasp')
cell = (at0.cell[:], at0.get_scaled_positions(), at0.numbers)
ds = spglib.get_symmetry_dataset(cell, symprec=SP)
assert ds.number == 19, ds.international
lat, pos, num = spglib.refine_cell(cell, symprec=SP)
sym = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
d = sym.get_scaled_positions() - at0.get_scaled_positions()
d -= np.round(d)
ang = 2 * np.pi * d
shift = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0)) / (2 * np.pi)
sym.set_scaled_positions((sym.get_scaled_positions() - shift) % 1.0)
V_ref = sym.get_volume()
print('参考胞 %s (#%d) V=%.4f Å³ N=%d' % (ds.international, ds.number, V_ref, len(sym)), flush=True)

relax_rec, summ = [], []
t_all = time.time()
for r in RATIOS:
    tag = 'vol_%04d' % int(round(r * 1000))
    d_ = OUT + tag
    os.makedirs(d_, exist_ok=True)
    t0 = time.time()

    s = r ** (1 / 3)
    a = sym.copy()
    a.set_cell(sym.cell[:] * s, scale_atoms=False)
    a.calc = calc
    write(d_ + '/POSCAR', a, format='vasp', direct=True)
    FIRE(a, logfile=None).run(fmax=a_.fmax, steps=8000)
    a.calc = calc
    E = float(a.get_potential_energy()); fmax = float(np.abs(a.get_forces()).max())
    write(d_ + '/POSCAR_relaxed', a, format='vasp', direct=True)
    relax_rec.append(dict(dir=tag, ratio=r, V=float(a.get_volume()), E=E, fmax=fmax))
    print('  %-9s V=%8.3f  E=%.6f (%.6f eV/at)  |F|max=%.1e  (%.0fs)'
          % (tag, a.get_volume(), E, E / len(a), fmax, time.time() - t0), flush=True)

    pa = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                      scaled_positions=a.get_scaled_positions())
    ph = Phonopy(pa, supercell_matrix=np.diag([2, 2, 1]), primitive_matrix='auto', symprec=SP)
    ph.generate_displacements(distance=0.01)
    scs = ph.supercells_with_displacements
    try:
        ph.save(d_ + '/phonopy_disp.yaml')
    except AttributeError:
        ph.write_yaml(d_ + '/phonopy_disp.yaml')

    forces = []
    for k, sc in enumerate(scs, 1):
        at = Atoms(symbols=list(sc.symbols), cell=np.array(sc.cell),
                   scaled_positions=np.array(sc.scaled_positions), pbc=True)
        at.calc = calc
        forces.append(at.get_forces())
        if k % 66 == 0:
            print('       %3d/%d (%.0fs)' % (k, len(scs), time.time() - t0), flush=True)
    ph.forces = forces
    write_FORCE_SETS(ph.dataset, filename=d_ + '/FORCE_SETS')

    ph.produce_force_constants()
    ph.run_mesh(a_.mesh, with_eigenvectors=False)
    fr = np.array(ph.get_mesh_dict()['frequencies'])
    n_imag = int((fr < -1e-3).sum())
    ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
    ph.write_yaml_thermal_properties(filename=d_ + '/thermal_properties.yaml')
    tp = ph.get_thermal_properties_dict()
    T = np.array(tp['temperatures']); F = np.array(tp['free_energy'])
    i300 = int(np.argmin(abs(T - 300)))
    summ.append(dict(dir=tag, V=float(a.get_volume()), E=E, n_imag=n_imag,
                     fmin=float(fr.min()),
                     F300=float(F[i300] * 1000 / len(ph.primitive)),
                     zpe_kJ=float(tp['zero_point_energy']) if 'zero_point_energy' in tp else None))
    print('     %-9s 虚频 %4d  最低模 %+.3f THz  F(300K)=%8.4f kJ/mol  (%.0fs)'
          % (tag, n_imag, fr.min(), F[i300], time.time() - t0), flush=True)

json.dump(relax_rec, open(OUT + 'relax.json', 'w'), indent=2)
json.dump(summ, open(OUT + 'phonon_summary.json', 'w'), indent=2)
print('\n总耗时 %.0f s -> %s{relax.json,phonon_summary.json}' % (time.time() - t_all, OUT))
