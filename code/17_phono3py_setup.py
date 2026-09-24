#!/usr/bin/env python3
"""为 V6 的 α-ZrW2O8 建立 phono3py 三声子（热导率）计算：
2x2x2 超胞 + cutoff_pair_distance，输出位移结构与 cell 列表（供 GPU 算力）。
"""
import warnings, os, json, time
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from phonopy.structure.atoms import PhonopyAtoms
from phono3py import Phono3py

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'phono3py/'
os.makedirs(OUT, exist_ok=True)
DIM = [2, 2, 2]
CUT = 6.0
DISP = 0.03

t0 = time.time()
a = read(ROOT + 'elastic/_alpha_lmps.vasp')
uc = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                  scaled_positions=a.get_scaled_positions())
ph3 = Phono3py(uc, supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=0.05)
print('超胞 %d 原子，cell %s' % (len(ph3.supercell), np.round(ph3.supercell.cell, 3)), flush=True)
ph3.generate_displacements(distance=DISP, cutoff_pair_distance=CUT)
sup = ph3.supercells_with_displacements
print('三阶位移数 = %d  (cutoff_pair=%.1f A, disp=%.2f A)  (%.0fs)'
      % (len(sup), CUT, DISP, time.time() - t0), flush=True)
cells = []
for i, s in enumerate(sup):
    cells.append(dict(index=i,
                      symbols=list(s.symbols),
                      cell=np.array(s.cell).tolist(),
                      positions=np.array(s.scaled_positions).tolist()))
json.dump(cells, open(OUT + 'alpha_cells.json', 'w'))
print('写出 alpha_cells.json (%d 个超胞)' % len(cells))
try:
    ph3.save(OUT + 'alpha_phono3py.yaml')
except Exception as e:
    print('save 失败', e)
print('完成 (%.0fs)' % (time.time() - t0))
