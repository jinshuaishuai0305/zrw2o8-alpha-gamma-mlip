#!/usr/bin/env python3
"""γ-ZrW2O8 QHA：**单个**体积点的弛豫 + 声子力 + 热学量（V6, float64）。
设计成 job array 的一个 task：一切参数由命令行给出，不依赖其它 task。
用法: python 23_gamma_vol_one.py <ratio> <outdir> [mesh...]
"""
import os, sys, json, time, warnings
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
SP = 0.05
ratio = float(sys.argv[1]); OUT = sys.argv[2].rstrip('/')
mesh = [int(x) for x in sys.argv[3:6]] or [8, 8, 8]
tag = 'vol_%04d' % int(round(ratio * 1000))
d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
t0 = time.time()

calc = MACECalculator(model_paths=[ROOT + 'AlZrW_v6_stagetwo.model'],
                      device=os.environ.get('DEV', 'cuda'), default_dtype='float64')
at0 = read(ROOT + '_gamma_ref.vasp')
cell = (at0.cell[:], at0.get_scaled_positions(), at0.numbers)
ds = spglib.get_symmetry_dataset(cell, symprec=SP)
lat, pos, num = spglib.refine_cell(cell, symprec=SP)
sym = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
dd = sym.get_scaled_positions() - at0.get_scaled_positions(); dd -= np.round(dd)
ang = 2 * np.pi * dd
shift = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0)) / (2 * np.pi)
sym.set_scaled_positions((sym.get_scaled_positions() - shift) % 1.0)

s = ratio ** (1 / 3)
a = sym.copy(); a.set_cell(sym.cell[:] * s, scale_atoms=False); a.calc = calc
write(d + '/POSCAR', a, format='vasp', direct=True)
FIRE(a, logfile=None).run(fmax=1e-4, steps=8000)
a.calc = calc
E = float(a.get_potential_energy()); fmax = float(np.abs(a.get_forces()).max())
write(d + '/POSCAR_relaxed', a, format='vasp', direct=True)
print('%s V=%.3f E=%.6f |F|max=%.1e relax=%.0fs' % (tag, a.get_volume(), E, fmax, time.time() - t0), flush=True)

pa = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                  scaled_positions=a.get_scaled_positions())
ph = Phonopy(pa, supercell_matrix=np.diag([2, 2, 1]), primitive_matrix='auto', symprec=SP)
ph.generate_displacements(distance=0.01)
scs = ph.supercells_with_displacements
try: ph.save(d + '/phonopy_disp.yaml')
except AttributeError: ph.write_yaml(d + '/phonopy_disp.yaml')
forces = []
for k, sc in enumerate(scs, 1):
    at = Atoms(symbols=list(sc.symbols), cell=np.array(sc.cell),
               scaled_positions=np.array(sc.scaled_positions), pbc=True)
    at.calc = calc
    forces.append(at.get_forces())
    if k % 50 == 0: print('   %d/%d %.0fs' % (k, len(scs), time.time() - t0), flush=True)
ph.forces = forces
write_FORCE_SETS(ph.dataset, filename=d + '/FORCE_SETS')
ph.produce_force_constants()
ph.run_mesh(mesh, with_eigenvectors=False)
fr = np.array(ph.get_mesh_dict()['frequencies'])
n_imag = int((fr < -1e-3).sum())
ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
ph.write_yaml_thermal_properties(filename=d + '/thermal_properties.yaml')
tp = ph.get_thermal_properties_dict()
T = np.array(tp['temperatures']); F = np.array(tp['free_energy'])
i300 = int(np.argmin(abs(T - 300)))
rec = dict(dir=tag, ratio=ratio, V=float(a.get_volume()), E=E, fmax=fmax,
           n_imag=n_imag, fmin=float(fr.min()),
           F300=float(F[i300] * 1000 / len(ph.primitive)))
json.dump(rec, open(d + '/summary.json', 'w'), indent=1)
print('DONE %s 虚频=%d 最低模=%+.3f THz F300=%.4f  total=%.0fs'
      % (tag, n_imag, fr.min(), F[i300], time.time() - t0), flush=True)
