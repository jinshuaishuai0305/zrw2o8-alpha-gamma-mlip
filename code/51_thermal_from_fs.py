#!/usr/bin/env python3
"""集群侧（**phonopy 环境**）：从 POSCAR + displacements.json + FORCE_SETS 建 FC，
输出 thermal_properties.yaml —— **不依赖 ase**（phonopy 环境没有 ase）。

用法: WORKROOT=<含 vol_*/ 的目录> TAGS="vol_1000,..." [MESH="8 8 8"] python 51_thermal_from_fs.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.vasp import read_vasp

WORKROOT = os.environ['WORKROOT']
TAGS = [t for t in os.environ['TAGS'].split(',') if t]
MESH = [int(x) for x in os.environ.get('MESH', '8 8 8').split()]
SYMPREC = float(os.environ.get('SYMPREC', '0.05'))
DIM = [int(x) for x in os.environ.get('DIM', '3 3 3').split()]


def parse_force_sets(path, nsc, ndisp):
    toks = [l for l in open(path).read().splitlines() if l.strip()]
    assert int(toks[0]) == nsc, '%s 首行 %s != %d' % (path, toks[0], nsc)
    assert int(toks[1]) == ndisp, '%s 次行 %s != %d' % (path, toks[1], ndisp)
    k, out = 2, []
    for _ in range(ndisp):
        int(toks[k]); k += 1
        [float(x) for x in toks[k].split()]; k += 1
        out.append(np.array([[float(x) for x in toks[k + i].split()] for i in range(nsc)]))
        k += nsc
    return out


for tag in TAGS:
    d = os.path.join(WORKROOT, tag)
    t0 = time.time()
    cell = read_vasp(d + '/POSCAR')                      # 晶胞（phonopy 自带 VASP 读取器）
    # 位移清单：两种格式都兼容
    #   displacements.json: [{"atom":1-based,"disp":[...],"dir":"disp-001"}, ...]
    #   disp_info.dat:      每行 "序号 原子(1基) dx dy dz"
    if os.path.exists(d + '/displacements.json'):
        man = json.load(open(d + '/displacements.json'))
    elif os.path.exists(d + '/disp_info.dat'):
        man = []
        for l in open(d + '/disp_info.dat'):
            f = l.split()
            if len(f) < 5:
                continue
            man.append(dict(atom=int(f[1]), disp=[float(f[2]), float(f[3]), float(f[4])],
                            dir='disp-%03d' % int(f[0])))
    else:
        raise SystemExit('%s 缺位移清单' % d)
    ph = phonopy.Phonopy(cell, supercell_matrix=np.diag(DIM), primitive_matrix='auto',
                         symprec=SYMPREC)
    # 必须先 generate_displacements：它建立 _dataset（否则 ph.forces= 会报
    # TypeError: argument of type 'NoneType' is not iterable）
    ph.generate_displacements(distance=0.01)
    nsc = len(ph.supercell)
    assert len(ph.supercells_with_displacements) == len(man), \
        '%s 位移数不符: %d vs %d' % (tag, len(ph.supercells_with_displacements), len(man))
    forces = parse_force_sets(d + '/FORCE_SETS', nsc, len(man))
    ph.forces = forces
    ph.produce_force_constants()
    ph.run_mesh(MESH, with_eigenvectors=False)
    fr = np.array(ph.get_mesh_dict()['frequencies'])
    n_imag = int((fr < -1e-3).sum())
    ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
    ph.write_yaml_thermal_properties(filename=d + '/thermal_properties.yaml')
    tp = ph.get_thermal_properties_dict()
    T = np.array(tp['temperatures']); F = np.array(tp['free_energy'])
    i300 = int(np.argmin(abs(T - 300)))
    print('  %-10s V=%9.3f N=%3d 超胞%5d 最低模 %+.4f THz 虚频 %4d  ZPE=%.4f meV/at  F300=%.4f kJ/mol  (%.0fs)'
          % (tag, cell.volume, len(cell), nsc, fr.min(), n_imag,
             F[0] / len(cell) * 1000 / 96.485, F[i300], time.time() - t0), flush=True)
print('完成 %d 个点' % len(TAGS))
