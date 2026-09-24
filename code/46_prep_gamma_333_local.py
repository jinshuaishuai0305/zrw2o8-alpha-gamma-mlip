#!/usr/bin/env python3
"""本地（phonopy 环境）：为 11 个 γ 构型生成 **3×3×3 超胞**位移结构，供集群算力。

为什么是 11 个构型、为什么用 3×3×3：
  ① 复核 2×2×1 是否制造假虚频 —— 实测同一构型（a 轴 +1% 应变）
     2×2×1 → 最低模 −3.01 THz、509 个虚频；3×3×3 → +0.25 THz、0 个虚频。
     归档的 10 个 γ QHA 点与各向异性 QHA 都用 2×2×1，必须复核。
  ② 6 个轴向应变态（a/b/c 各 ±1%）+ 平衡点 + 4 个压缩点（0.945/0.935/0.925/0.915）

流程（每一步都要有，缺一不可）：
  应变/缩放 → **固定胞弛豫离子**（恢复 P2₁2₁2₁）→ 对称化(0.05) → 生成位移
  注意：对已对称化的 γ 胞直接各向同性缩放会破坏 P2₁2₁2₁（偏差随压缩线性增长），
        必须在目标体积/应变下弛豫离子后才能恢复。

输出（集群读这个布局）：
  <outdir>/<tag>/POSCAR                      晶胞（3×3×3 超胞）
  <outdir>/<tag>/displacements.json          [{"atom":1-based,"disp":[...],"dir":"disp-001"}, ...]
  <outdir>/<tag>/<dir>/POSCAR                每个位移的超胞
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
import spglib
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'gamma_phonons_333/'
DIM = [3, 3, 3]
SP = 0.05
DISP = 0.01
DTYPE = os.environ.get('DTYPE', 'float64')

# 6 个轴向应变态 + 平衡点 + 4 个压缩点
JOBS = []
for ax, j in (('a', 0), ('b', 1), ('c', 2)):
    for sgn, s in (('p', +0.01), ('m', -0.01)):
        JOBS.append(dict(tag=ax + sgn, src=ROOT + 'elastic/_gamma_lmps.vasp',
                         strain=(j, s), ratio=None,
                         note='轴向应变 %s %+.0f%%' % (ax, s * 100)))
JOBS.append(dict(tag='eq', src=ROOT + 'qha_gamma/vol_1000/POSCAR_relaxed',
                 strain=None, ratio=None, note='平衡点'))
for r in (0.945, 0.935, 0.925, 0.915):
    JOBS.append(dict(tag='c%04d' % int(round(r * 1000)),
                     src=ROOT + 'qha_gamma_f32/vol_%04d/POSCAR_relaxed' % int(round(r * 1000))
                     if os.path.exists(ROOT + 'qha_gamma_f32/vol_%04d/POSCAR_relaxed' % int(round(r * 1000)))
                     else ROOT + 'qha_gamma_f32/vol_%04d/POSCAR' % int(round(r * 1000)),
                     strain=None, ratio=r, note='压缩点 %.3f V₀' % r))

os.environ.setdefault('MACE_MODEL', ROOT + 'AlZrW_v6_stagetwo.model')
from mace.calculators import MACECalculator          # noqa: E402
calc = MACECalculator(model_paths=ROOT + 'AlZrW_v6_stagetwo.model',
                      device='cuda', default_dtype=DTYPE)
os.makedirs(OUT, exist_ok=True)
print('本地生成 3×3×3 位移结构：%d 个构型  dtype=%s' % (len(JOBS), DTYPE), flush=True)

recs = []
for job in JOBS:
    tag = job['tag']
    d = OUT + tag
    # 断点续跑：已生成完整就跳过
    if os.path.exists(d + '/displacements.json') and os.path.exists(d + '/POSCAR'):
        n = len(json.load(open(d + '/displacements.json')))
        print('  %-7s 已存在（%d 位移），跳过' % (tag, n), flush=True)
        recs.append(dict(tag=tag, n_disp=n, note=job['note'], skipped=True)); continue
    os.makedirs(d, exist_ok=True)
    t0 = time.time()

    a = read(job['src'])
    if job['strain'] is not None:
        j, s = job['strain']
        e = np.zeros(3); e[j] = s
        a.set_cell(a.cell[:] @ np.diag(1 + e), scale_atoms=True)
    a.calc = calc
    FIRE(a, logfile=None).run(fmax=1e-5, steps=8000)
    a.calc = calc
    E = float(a.get_potential_energy())

    lat, pos, num = spglib.standardize_cell((a.cell[:], a.get_scaled_positions(), a.numbers),
                                            to_primitive=False, no_idealize=False, symprec=SP)
    sym = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
    ds = spglib.get_symmetry_dataset((sym.cell[:], sym.get_scaled_positions(), sym.numbers),
                                     symprec=SP)
    if ds.number != 19:
        print('  !! %s 弛豫后非 P2₁2₁2₁（得到 %s），跳过' % (tag, ds.international), flush=True)
        continue
    write(d + '/POSCAR', sym, format='vasp', direct=True)

    pa = PhonopyAtoms(symbols=sym.get_chemical_symbols(), cell=np.array(sym.cell),
                      scaled_positions=sym.get_scaled_positions())
    ph = Phonopy(pa, supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=SP)
    ph.generate_displacements(distance=DISP)
    scs = ph.supercells_with_displacements
    man = []
    for i, sc in enumerate(scs, 1):
        name = 'disp-%03d' % i
        sub = '%s/%s' % (d, name)
        os.makedirs(sub, exist_ok=True)
        at = Atoms(numbers=sc.numbers, scaled_positions=sc.scaled_positions, cell=sc.cell, pbc=True)
        write(sub + '/POSCAR', at, format='vasp', direct=True)
    man = []
    for i, row in enumerate(ph.displacements, 1):
        man.append(dict(atom=int(row[0]) + 1, disp=[float(row[1]), float(row[2]), float(row[3])],
                        dir='disp-%03d' % i))
    json.dump(man, open(d + '/displacements.json', 'w'), indent=2)
    rec = dict(tag=tag, note=job['note'], V=float(sym.get_volume()), E=E,
               sym=ds.international, n_supercell=len(ph.supercell), n_disp=len(scs))
    recs.append(rec)
    print('  %-7s %-18s V=%9.3f  %s  超胞 %4d  位移 %3d  (%.0fs)'
          % (tag, job['note'], sym.get_volume(), ds.international, len(ph.supercell),
             len(scs), time.time() - t0), flush=True)

json.dump(recs, open(OUT + 'prep_summary.json', 'w'), indent=2)
print('\n完成 %d 个构型 -> %s' % (len(recs), OUT))
