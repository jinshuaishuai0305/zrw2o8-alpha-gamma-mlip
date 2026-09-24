#!/usr/bin/env python3
"""γ 相压缩端 5 个 QHA 体积点（0.945–0.905 V₀）的位移结构生成。

背景：要把 P_t(T) 的可信上限从 400 K 推到 700 K，需覆盖 >2.5 GPa，
      γ 网格必须延伸到 <16.55 Å³/at（现网格最小 16.546）。

关键发现（本脚本处理的那个坑）：
  对已对称化的 γ 胞做**各向同性缩放会破坏 P2₁2₁2₁**（群检验偏差随压缩线性增长：
  0.99→0.14 Å，0.955→0.64 Å）。但**在固定体积下弛豫离子后对称性完全恢复**
  （实测 0.945/0.935/0.925/0.915 四点全部回到 P2₁2₁2₁ 且位移数 198）。
  所以流程必须是：缩放 → 弛豫离子 → 对称化 → 生成位移。

起点用归档的 0.955 胞（已弛豫、P2₁2₁2₁），逐点向下缩。

用法: python 40_gamma_compression_prep.py
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
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'qha_gamma_f32/'          # 与既有 10 个 γ 点同目录、同格式
SP = 0.05
RATIOS = [0.945, 0.935, 0.925, 0.915, 0.905]
START = ROOT + 'qha_gamma/vol_0955/POSCAR_relaxed'   # 0.955，已弛豫
START_R = 0.955

os.makedirs(OUT, exist_ok=True)
calc = MACECalculator(model_paths=ROOT + 'AlZrW_v6_stagetwo.model',
                      device='cuda', default_dtype='float32')
base = read(START)
print('起点 %s V=%.3f (%.4f Å³/at)' % (os.path.basename(START),
      base.get_volume(), base.get_volume() / 132), flush=True)

recs = []
prev, prev_r = base, START_R
t_all = time.time()
for r in RATIOS:
    tag = 'vol_%04d' % int(round(r * 1000))
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()

    # 1) 从上一个点缩放（体积比 → 线性缩放因子）
    s = (r / prev_r) ** (1 / 3)
    a = prev.copy()
    a.set_cell(prev.cell[:] * s, scale_atoms=False)

    # 2) 固定体积弛豫离子 —— 这一步恢复对称性
    a.calc = calc
    FIRE(a, logfile=None).run(fmax=1e-4, steps=8000)
    a.calc = calc
    E = float(a.get_potential_energy())
    fmax = float(np.abs(a.get_forces()).max())

    # 3) 对称化（离子已弛豫，P2₁2₁2₁ 可恢复）
    lat, pos, num = spglib.standardize_cell((a.cell[:], a.get_scaled_positions(), a.numbers),
                                            to_primitive=False, no_idealize=False, symprec=SP)
    sym = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
    ds = spglib.get_symmetry_dataset((sym.cell[:], sym.get_scaled_positions(), sym.numbers),
                                     symprec=SP)
    if ds.number != 19:
        raise SystemExit('%s 弛豫后仍非 P2₁2₁2₁（得到 %s）' % (tag, ds.international))
    write(d + '/POSCAR_relaxed', sym, format='vasp', direct=True)
    write(d + '/POSCAR', a, format='vasp', direct=True)

    # 4) 生成位移
    pa = PhonopyAtoms(symbols=sym.get_chemical_symbols(), cell=np.array(sym.cell),
                      scaled_positions=sym.get_scaled_positions())
    ph = Phonopy(pa, supercell_matrix=np.diag([2, 2, 1]), primitive_matrix='auto', symprec=SP)
    ph.generate_displacements(distance=0.01)
    scs = ph.supercells_with_displacements
    try:
        ph.save(d + '/phonopy_disp.yaml')
    except AttributeError:
        ph.write_yaml(d + '/phonopy_disp.yaml')
    man = []
    for i, sc in enumerate(scs, 1):
        sub = '%s/disp-%03d' % (d, i)
        os.makedirs(sub, exist_ok=True)
        at = Atoms(numbers=sc.numbers, scaled_positions=sc.scaled_positions, cell=sc.cell, pbc=True)
        write(sub + '/POSCAR', at, format='vasp', direct=True)
        man.append(dict(index=i, dir=sub))
    json.dump(man, open(d + '/disp.json', 'w'), indent=2)
    with open(d + '/disp_info.dat', 'w') as fh:
        for k, row in enumerate(ph.displacements, 1):
            fh.write('%d %d %.10f %.10f %.10f\n'
                     % (k, int(row[0]) + 1, float(row[1]), float(row[2]), float(row[3])))

    print('  %-9s V=%9.3f (%.4f Å³/at)  %s  位移 %3d  E=%.6f  |F|max=%.1e  (%.0fs)'
          % (tag, sym.get_volume(), sym.get_volume() / 132, ds.international, len(scs),
             E, fmax, time.time() - t0), flush=True)
    recs.append(dict(dir=tag, ratio=r, V=float(sym.get_volume()), E=E, fmax=fmax,
                     n_disp=len(scs), n_supercell=528))
    prev, prev_r = sym, r

json.dump(recs, open(OUT + 'relax_compression.json', 'w'), indent=2)
print('\n完成 %d 个压缩端体积点（总 %.0f s）-> %s' % (len(recs), time.time() - t_all, OUT))
