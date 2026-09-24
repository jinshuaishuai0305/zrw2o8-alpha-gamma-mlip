#!/usr/bin/env python3
"""集群侧（**phonopy 环境**，有 phonopy+spglib+numpy、无 ase）：
把弛豫胞对称化到 P2₁2₁2₁ 并生成 3×3×3 位移结构。

环境分工（skill references/qha-workflow.md §1）：
  phonopy env : phonopy + spglib + numpy   -> 本脚本（对称化 + 生成位移）
  mace env    : ase + torch + mace         -> 45_forces_only.py（算力）

手工解析 POSCAR 的原因：phonopy 环境没有 ase。VASP5 格式 + Direct 坐标，解析量很小。

用法: SRC=<POSCAR_relaxed> TAG=<tag> OUTROOT=<dir> python 50_cluster_symmetrize_disp.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import spglib
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

SYMS = ['Zr', 'W', 'O', 'Al']
SRC = os.environ['SRC']
TAG = os.environ['TAG']
OUTROOT = os.environ.get('OUTROOT', './' + TAG)
DIM = [int(x) for x in os.environ.get('SUPER', '3 3 3').split()]
SP = float(os.environ.get('SYMPREC', '0.05'))
DISP = float(os.environ.get('DISP', '0.01'))


def read_poscar(path):
    L = [l.rstrip('\n') for l in open(path)]
    scale = float(L[1])
    cell = np.array([[float(x) for x in L[2 + i].split()[:3]] for i in range(3)]) * scale
    k = 5
    if not L[k].strip() or all(not c.isalpha() for c in L[k].strip()):
        symbols = None                       # VASP4：无元素行
    else:
        symbols = L[k].split(); k += 1
    counts = [int(x) for x in L[k].split()]; k += 1
    mode = L[k].strip().lower(); k += 1
    n = sum(counts)
    pos = np.array([[float(x) for x in L[k + i].split()[:3]] for i in range(n)])
    if mode.startswith('c') or mode.startswith('k'):
        pos = pos @ np.linalg.inv(cell)
    if symbols is None:
        symbols = ['X'] * len(counts)
    labels = []
    for s, c in zip(symbols, counts):
        labels += [s] * c
    return cell, pos, labels


cell, pos, labels = read_poscar(SRC)
print('[%s] 读入 %s: %d 原子 V=%.3f  元素顺序 %s'
      % (TAG, os.path.basename(SRC), len(pos), abs(np.linalg.det(cell)),
         sorted(set(labels))), flush=True)
nums = np.array([SYMS.index(s) + 1 for s in labels])

lat, spos, snum = spglib.standardize_cell((cell, pos, nums), to_primitive=False,
                                          no_idealize=False, symprec=SP)
ds = spglib.get_symmetry_dataset((lat, spos, snum), symprec=SP)
print('[%s] 对称化后 %s (#%d)' % (TAG, ds.international, ds.number), flush=True)
if ds.number != 19:
    sys.exit('[%s] 不是 P2₁2₁2₁（%s），终止' % (TAG, ds.international))

sym_labels = [SYMS[i - 1] for i in snum]
os.makedirs(OUTROOT, exist_ok=True)
os.chdir(OUTROOT)
ph = Phonopy(PhonopyAtoms(symbols=sym_labels, cell=lat, scaled_positions=spos),
             supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=SP)
ph.generate_displacements(distance=DISP)
scs = ph.supercells_with_displacements
print('[%s] 超胞 %d 原子，位移 %d 个' % (TAG, len(ph.supercell), len(scs)), flush=True)

def write_poscar(path, cell_, spos_, labels_):
    with open(path, 'w') as fh:
        fh.write(' '.join(sorted(set(labels_))) + '\n1.0\n')
        for v in cell_:
            fh.write('  %.12f %.12f %.12f\n' % tuple(v))
        order = sorted(set(labels_))
        fh.write(' '.join(order) + '\n')
        fh.write(' '.join(str(labels_.count(s)) for s in order) + '\n')
        fh.write('Direct\n')
        # VASP 要求按元素分组
        for s in order:
            for i, l in enumerate(labels_):
                if l == s:
                    fh.write('  %.12f %.12f %.12f\n' % tuple(spos_[i]))

write_poscar('POSCAR_unit', lat, spos, sym_labels)
for i, sc in enumerate(scs, 1):
    d = 'disp-%03d' % i
    os.makedirs(d, exist_ok=True)
    write_poscar(d + '/POSCAR', np.array(sc.cell), np.array(sc.scaled_positions), list(sc.symbols))
man = [dict(atom=int(r[0]) + 1, disp=[float(r[1]), float(r[2]), float(r[3])], dir='disp-%03d' % i)
       for i, r in enumerate(ph.displacements, 1)]
json.dump(man, open('displacements.json', 'w'), indent=2)
# 算力脚本需要晶胞（3×3×3 超胞本身，用于核对原子数）
write_poscar('POSCAR', np.array(ph.supercell.cell), np.array(ph.supercell.scaled_positions),
             list(ph.supercell.symbols))
json.dump(dict(tag=TAG, V=float(abs(np.linalg.det(lat))), sym=ds.international,
               n_disp=len(scs), n_supercell=len(ph.supercell)),
          open('prep.json', 'w'), indent=2)
print('[%s] DONE 位移结构生成完毕 -> %s' % (TAG, os.getcwd()), flush=True)
