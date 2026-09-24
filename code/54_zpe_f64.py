#!/usr/bin/env python3
"""用 float64 的力重算 ZPE，与 float32 结果比较（收敛表的第三段）。"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.vasp import read_vasp

ROOT = os.environ['ROOT']
CASES = [('alpha', ROOT + '/qha_f32_v6/qha_f32/vol_1000', 'disp_info.dat', [12, 12, 12]),
         ('gamma', ROOT + '/qha_gamma_333/vol_1000', 'displacements.json', [8, 8, 8])]
out = {}
for tag, d, fmt, mesh in CASES:
    if fmt == 'disp_info.dat':
        man = []
        for l in open(d + '/disp_info.dat'):
            f = l.split()
            if len(f) >= 5:
                man.append(dict(atom=int(f[1]), disp=[float(f[2]), float(f[3]), float(f[4])],
                                dir='disp-%03d' % int(f[0])))
    else:
        man = json.load(open(d + '/displacements.json'))
    cell = read_vasp(d + '/POSCAR')
    ph = phonopy.Phonopy(cell, supercell_matrix=np.diag([3, 3, 3]),
                         primitive_matrix='auto', symprec=0.05)
    ph.generate_displacements(distance=0.01)
    nsc = len(ph.supercell)
    assert len(man) == len(ph.supercells_with_displacements)
    rec = {}
    for dt in ('float32', 'float64'):
        calc = MACECalculator(model_paths=ROOT + '/mace_train/AlZrW_v6_stagetwo.model',
                              device='cuda', default_dtype=dt)
        forces = []
        for m in man:
            at = read('%s/%s/POSCAR' % (d, m['dir'])); at.calc = calc
            forces.append(at.get_forces())
        ph.forces = forces
        ph.produce_force_constants()
        ph.run_mesh(mesh, with_eigenvectors=False)
        fr = np.array(ph.get_mesh_dict()['frequencies'])
        ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
        tp = ph.get_thermal_properties_dict()
        T = np.array(tp['temperatures']); F = np.array(tp['free_energy'])
        i300 = int(np.argmin(abs(T - 300)))
        n = len(ph.primitive)
        rec[dt] = dict(zpe=F[0] / n * 1000 / 96.485, F300=F[i300] / n * 1000 / 96.485,
                       n_imag=int((fr < -1e-3).sum()), fmin=float(fr.min()))
        print('  %-6s %-8s ZPE=%.4f meV/at  F300=%.4f meV/at  虚频=%d  最低模=%+.4f THz'
              % (tag, dt, rec[dt]['zpe'], rec[dt]['F300'], rec[dt]['n_imag'], rec[dt]['fmin']),
              flush=True)
    print('    ΔZPE(f32−f64) = %+.5f meV/at   ΔF300 = %+.5f meV/at'
          % (rec['float32']['zpe'] - rec['float64']['zpe'],
             rec['float32']['F300'] - rec['float64']['F300']), flush=True)
    out[tag] = rec
json.dump(out, open(os.environ.get('OUTF', '/tmp/zpe_f64.json'), 'w'), indent=1)
print('写出', os.environ.get('OUTF'))
