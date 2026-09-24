#!/usr/bin/env python3
"""本地：把归档的 10 个 γ QHA 体积点重做成 **3×3×3 超胞**位移结构。

理由：归档 γ 点是 2×2×1，而 α 是 3×3×3；且已实测 2×2×1 在应变构型上会造假虚频
      （a 轴 +1%：−3.01 THz / 509 个虚频 vs 3×3×3 的 +0.25 THz / 0 个）。
      虽然平衡点 ZPE 两者只差 0.004 meV/atom，但两相超胞一致才是可辩护的做法。

流程：归档弛豫胞 → spglib 对称化(0.05) → 生成 3×3×3 位移(198 个)
输出：qha_gamma_333/<tag>/{POSCAR, displacements.json, disp-XXX/POSCAR}
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
import spglib
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'qha_gamma_333/'
DIM = [3, 3, 3]
SP = 0.05
SRC = {
    'vol_0955': ROOT + 'qha_gamma_f32/vol_0955/POSCAR_relaxed',
    'vol_0965': ROOT + 'qha_gamma_f32/vol_0965/POSCAR_relaxed',
    'vol_0975': ROOT + 'qha_gamma_f32/vol_0975/POSCAR_relaxed',
    'vol_0985': ROOT + 'qha_gamma/vol_0985/POSCAR_relaxed',
    'vol_0990': ROOT + 'qha_gamma/vol_0990/POSCAR_relaxed',
    'vol_0995': ROOT + 'qha_gamma/vol_0995/POSCAR_relaxed',
    'vol_1000': ROOT + 'qha_gamma/vol_1000/POSCAR_relaxed',
    'vol_1005': ROOT + 'qha_gamma/vol_1005/POSCAR_relaxed',
    'vol_1010': ROOT + 'qha_gamma/vol_1010/POSCAR_relaxed',
    'vol_1015': ROOT + 'qha_gamma/vol_1015/POSCAR_relaxed',
}
os.makedirs(OUT, exist_ok=True)
recs = []
for tag, src in SRC.items():
    d = OUT + tag
    if os.path.exists(d + '/displacements.json'):
        print('  %-9s 已存在，跳过' % tag, flush=True); continue
    if not os.path.exists(src):
        print('  !! %s 缺源文件 %s' % (tag, src), flush=True); continue
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    a = read(src)
    lat, pos, num = spglib.standardize_cell((a.cell[:], a.get_scaled_positions(), a.numbers),
                                            to_primitive=False, no_idealize=False, symprec=SP)
    sym = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
    ds = spglib.get_symmetry_dataset((sym.cell[:], sym.get_scaled_positions(), sym.numbers),
                                     symprec=SP)
    if ds.number != 19:
        print('  !! %s 对称化后 %s，跳过' % (tag, ds.international), flush=True); continue
    write(d + '/POSCAR', sym, format='vasp', direct=True)
    ph = Phonopy(PhonopyAtoms(symbols=sym.get_chemical_symbols(), cell=np.array(sym.cell),
                              scaled_positions=sym.get_scaled_positions()),
                 supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=SP)
    ph.generate_displacements(distance=0.01)
    scs = ph.supercells_with_displacements
    for i, sc in enumerate(scs, 1):
        sub = '%s/disp-%03d' % (d, i)
        os.makedirs(sub, exist_ok=True)
        write(sub + '/POSCAR', Atoms(numbers=sc.numbers, scaled_positions=sc.scaled_positions,
                                     cell=sc.cell, pbc=True), format='vasp', direct=True)
    man = [dict(atom=int(r[0]) + 1, disp=[float(r[1]), float(r[2]), float(r[3])],
                dir='disp-%03d' % i) for i, r in enumerate(ph.displacements, 1)]
    json.dump(man, open(d + '/displacements.json', 'w'), indent=2)
    recs.append(dict(tag=tag, V=float(sym.get_volume()), n_disp=len(scs)))
    print('  %-9s V=%9.3f  %s  超胞 %4d  位移 %3d  (%.0fs)'
          % (tag, sym.get_volume(), ds.international, len(ph.supercell), len(scs),
             time.time() - t0), flush=True)
json.dump(recs, open(OUT + 'prep_summary.json', 'w'), indent=2)
print('\n完成 %d 个点 -> %s' % (len(recs), OUT))
