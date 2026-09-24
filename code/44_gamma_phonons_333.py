#!/usr/bin/env python3
"""γ 相声子：**统一 3×3×3 超胞**，一个构型一个任务（集群 mace 环境，float32）。

为什么用 3×3×3：实测同一个 γ 构型（a 轴 +1% 应变）
   2×2×1 超胞 -> 最低模 −3.01 THz，509 个虚频
   3×3×3 超胞 -> 最低模 +0.25 THz，0 个虚频
   ⟹ 2×2×1 会制造**假虚频**，而归档的 10 个 γ QHA 点与各向异性 QHA 都用的是 2×2×1，
      因此必须用 3×3×3 复核。

任务清单由 JOBS 环境变量给出，格式：  tag:type:path
   type=relax  -> path 是 POSCAR，固定胞弛豫后生成位移（用于应变态/压缩点）
   type=direct -> path 是已弛豫好的 POSCAR_relaxed，直接生成位移

每个任务输出：<outdir>/<tag>_ph.json（含 ZPE、最低模、虚频数、F(300K)）+ FORCE_SETS
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_SETS
from mace.calculators import MACECalculator

ROOT = os.environ.get('ROOT', '/home/jinshuaishuai/workdir_tyut-youzhiyong/'
                             'jinshuaishuai/AlZrW2O8_MACE/')
OUT = os.environ.get('OUT', ROOT + 'gamma_phonons_333/')
DIM = [3, 3, 3]
MESH = [8, 8, 8]
SP = 0.05
DISP = 0.01
DT = os.environ.get('DTYPE', 'float32')
os.makedirs(OUT, exist_ok=True)

import shlex
spec = os.environ.get("JOBSPEC", "")          # tag:type:path
parts = spec.split(":")
tag, typ = parts[0], parts[1]
path = ":".join(parts[2:])
print('task %s  type=%s  path=%s  DIM=%s  dtype=%s' % (tag, typ, path, DIM, DT), flush=True)

calc = MACECalculator(model_paths=[ROOT + 'mace_train/AlZrW_v6_stagetwo.model'],
                      device='cuda', default_dtype=DT)
t0 = time.time()
a = read(path)
if typ == 'relax':
    a.calc = calc
    FIRE(a, logfile=None).run(fmax=1e-4, steps=8000)
    a.calc = calc
    E = float(a.get_potential_energy())
else:
    E = float('nan')
d = os.path.join(OUT, tag)
os.makedirs(d, exist_ok=True)
write(d + '/POSCAR_relaxed', a, format='vasp', direct=True)
sym = 'n/a'
print('  弛豫后 V=%.3f  E=%s  (%.0fs)' % (a.get_volume(), E, time.time() - t0), flush=True)

pa = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                  scaled_positions=a.get_scaled_positions())
ph = Phonopy(pa, supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=SP)
ph.generate_displacements(distance=DISP)
scs = ph.supercells_with_displacements
try:
    ph.save(d + '/phonopy_disp.yaml')
except AttributeError:
    ph.write_yaml(d + '/phonopy_disp.yaml')
forces = []
for i, sc in enumerate(scs, 1):
    at = Atoms(symbols=list(sc.symbols), cell=np.array(sc.cell),
               scaled_positions=np.array(sc.scaled_positions), pbc=True)
    at.calc = calc
    forces.append(at.get_forces())
    if i % 50 == 0:
        print('    %d/%d  (%.0fs)' % (i, len(scs), time.time() - t0), flush=True)
ph.forces = forces
write_FORCE_SETS(ph.dataset, filename=d + '/FORCE_SETS')
ph.produce_force_constants()
ph.run_mesh(MESH, with_eigenvectors=False)
fr = np.array(ph.get_mesh_dict()['frequencies'])
ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
ph.write_yaml_thermal_properties(filename=d + '/thermal_properties.yaml')
tp = ph.get_thermal_properties_dict()
T = np.array(tp['temperatures']); F = np.array(tp['free_energy']) / 96.485 / len(ph.primitive)
i300 = int(np.argmin(abs(T - 300)))
rec = dict(tag=tag, type=typ, sym=sym, V=float(a.get_volume()), E=E,
           n_supercell=len(ph.supercell), n_disp=len(scs),
           fmin=float(fr.min()), n_imag=int((fr < -1e-3).sum()),
           zpe=float(F[0]), F300=float(F[i300]), dtype=DT)
json.dump(rec, open(os.path.join(OUT, tag + '_ph.json'), 'w'), indent=1)
print('DONE %s  V=%.3f  %s  最低模 %+.4f THz  虚频 %d  ZPE=%.3f meV/at  F300=%+.3f  (%.0fs)'
      % (tag, a.get_volume(), fr.min(), rec['n_imag'],
         rec['zpe'] * 1000, rec['F300'] * 1000, time.time() - t0), flush=True)
