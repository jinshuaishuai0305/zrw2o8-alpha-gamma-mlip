#!/usr/bin/env python3
"""精度检验（集群 V100 32 GB）：同一几何下 float32 vs float64 的 ZPE。
α: 3x3x3 1188 原子（12^3 mesh）；γ: 2x2x1 528 原子（8^3 mesh）。
注意：目的只是"精度"对照，故 γ 用可放下的 2x2x1；正文的 γ 声子仍以 3x3x3 为准。
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.vasp import read_vasp

ROOT = os.environ['ROOT']
CASES = [('alpha', ROOT + '/qha_f32_v6/qha_f32/vol_1000', 'disp_info.dat', [12, 12, 12], 1188),
         ('gamma', ROOT + '/qha_gamma/vol_1000', 'disp_info.dat', [8, 8, 8], 528)]
out = {}
for tag, d, fmt, mesh, nsc in CASES:
    if fmt == 'disp_info.dat':
        man = []
        for l in open(d + '/disp_info.dat'):
            f = l.split()
            if len(f) >= 5:
                man.append(dict(dir='disp-%03d' % int(f[0])))
    else:
        man = json.load(open(d + '/displacements.json'))
    cell = read_vasp(d + '/POSCAR')
    rec = {}
    for dt in ('float32', 'float64'):
        calc = MACECalculator(model_paths=ROOT + '/mace_train/AlZrW_v6_stagetwo.model',
                              device='cuda', default_dtype=dt)
        ph = phonopy.Phonopy(cell, supercell_matrix=np.diag([3, 3, 3]),
                             primitive_matrix='auto', symprec=0.05)
        ph.generate_displacements(distance=0.01)
        assert len(ph.supercells_with_displacements) == len(man), \
            (tag, len(ph.supercells_with_displacements), len(man))
        t0 = time.time(); fs = []
        for m in man:
            at = read('%s/%s/POSCAR' % (d, m['dir'])); at.calc = calc
            fs.append(at.get_forces())
        ph.forces = fs
        ph.produce_force_constants()
        ph.run_mesh(mesh, with_eigenvectors=False)
        fr = np.array(ph.get_mesh_dict()['frequencies'])
        ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
        tp = ph.get_thermal_properties_dict()
        T = np.array(tp['temperatures']); F = np.array(tp['free_energy'])
        i300 = int(np.argmin(abs(T - 300)))
        rec[dt] = dict(zpe_kJ=float(F[0]), F300_kJ=float(F[i300]),
                       n_imag=int((fr < -1e-3).sum()), t=time.time() - t0)
        print('  %-6s %-8s ZPE=%.4f kJ/mol  F300=%.4f  虚频=%d  (%.0fs)'
              % (tag, dt, F[0], F[i300], rec[dt]['n_imag'], time.time() - t0), flush=True)
    dZ = rec['float32']['zpe_kJ'] - rec['float64']['zpe_kJ']
    print('    ΔZPE(f32−f64) = %+.5f kJ/mol' % dZ, flush=True)
    rec['dZPE_kJ'] = dZ
    out[tag] = rec
json.dump(out, open(os.environ['OUTF'], 'w'), indent=1)
print('写出', os.environ['OUTF'])
