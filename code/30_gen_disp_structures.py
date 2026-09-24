#!/usr/bin/env python3
"""阶段 01+03（本地 phonopy 环境）：为 α 和 γ 的全部 QHA 体积点生成「位移超胞 POSCAR」。

严格按 skill `references/qha-workflow.md` 的接口：本步只产出 POSCAR 文本，
不含任何力；力由集群的 stage2_relax.py / stage4_forces.py（MACE float32）产生。

α: 7 点  0.975 0.980 1.000 1.020 1.040 1.060 1.080 V₀   3×3×3 超胞（1188 原子）
γ: 13 点 0.955 0.965 0.975 0.985 0.990 0.995 1.000 1.005 1.010 1.015 1.025 1.035 1.045 V₀
          2×2×1 超胞（528 原子）

注意：
  · α 必须 spglib 对称化到 P2₁3(#198)——DFT 弛豫残差 0.027 Å 会把对称性降到 P1，
    位移数从 22 涨到 264（qha-workflow.md §2 stage 03 实测）
  · γ 必须 spglib 对称化到 P2₁2₁2₁(#19)，并做 origin 回移（与 qha_gamma/stage1_scale.py 同法）
  · 缩放用 scale_atoms=False（分数坐标不变 → 各体积点内部几何完全相同）

用法: python 30_gen_disp_structures.py <alpha|gamma> [outroot]
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
import spglib
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
JOBS = {
    'alpha': dict(ref=ROOT + '_alpha_ref.vasp', sp=0.08, sg=198, dim=[3, 3, 3],
                  ratios=[0.975, 0.980, 1.000, 1.020, 1.040, 1.060, 1.080],
                  out=ROOT + 'qha_f32/', disp=0.01, origin_shift=False),
    'gamma': dict(ref=ROOT + '_gamma_ref.vasp', sp=0.05, sg=19, dim=[2, 2, 1],
                  ratios=[0.955, 0.965, 0.975, 0.985, 0.990, 0.995, 1.000,
                          1.005, 1.010, 1.015, 1.025, 1.035, 1.045],
                  out=ROOT + 'qha_gamma_f32/', disp=0.01, origin_shift=True),
}

phase = sys.argv[1]
outroot = sys.argv[2] if len(sys.argv) > 2 else JOBS[phase]['out']
cfg = JOBS[phase]
os.makedirs(outroot, exist_ok=True)

at0 = read(cfg['ref'])
cell = (at0.cell[:], at0.get_scaled_positions(), at0.numbers)
ds = spglib.get_symmetry_dataset(cell, symprec=cfg['sp'])
# 必须用 standardize_cell(no_idealize=False)：refine_cell 只理想化晶格、不动原子坐标，
# 残留的 0.03 Å 畸变会把对称性降到 R3/P1，位移数从 22 涨到 88（实测）。
lat, pos, num = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False,
                                        symprec=cfg['sp'])
sym = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
if cfg['origin_shift']:
    d = sym.get_scaled_positions() - at0.get_scaled_positions()
    d -= np.round(d)
    ang = 2 * np.pi * d
    shift = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0)) / (2 * np.pi)
    sym.set_scaled_positions((sym.get_scaled_positions() - shift) % 1.0)
assert ds.number == cfg['sg'], '期望 #%d，实际 %s' % (cfg['sg'], ds.international)
print('%s: %s (#%d)  V=%.4f  N=%d' % (phase, ds.international, ds.number, sym.get_volume(), len(sym)))

vols = []
for r in cfg['ratios']:
    tag = 'vol_%04d' % int(round(r * 1000))
    d_ = os.path.join(outroot, tag)
    os.makedirs(d_, exist_ok=True)
    s = r ** (1 / 3)
    a = sym.copy()
    a.set_cell(sym.cell[:] * s, scale_atoms=False)
    # 缩放后必须复核对称性：均匀缩放会保持对称群，但只有「已对称化」的胞才成立。
    # 若这里再降到 R3/P1，说明上游对称化没生效（会把 22 个位移撑到 88 个）。
    ds2 = spglib.get_symmetry_dataset((a.cell[:], a.get_scaled_positions(), a.numbers),
                                      symprec=cfg['sp'])
    if ds2.number != cfg['sg']:
        raise SystemExit('缩放后对称性降级: %s -> %s (#%d)；拒绝生成错误数据'
                         % (ds.international, ds2.international, ds2.number))
    write(d_ + '/POSCAR', a, format='vasp', direct=True)

    pa = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                      scaled_positions=a.get_scaled_positions())
    ph = Phonopy(pa, supercell_matrix=np.diag(cfg['dim']), primitive_matrix='auto',
                 symprec=cfg['sp'])
    ph.generate_displacements(distance=cfg['disp'])
    scs = ph.supercells_with_displacements
    try:
        ph.save(d_ + '/phonopy_disp.yaml')
    except AttributeError:
        ph.write_yaml(d_ + '/phonopy_disp.yaml')
    man = []
    for i, sc in enumerate(scs, 1):
        sub = '%s/disp-%03d' % (d_, i)
        os.makedirs(sub, exist_ok=True)
        s_at = Atoms(numbers=sc.numbers, scaled_positions=sc.scaled_positions, cell=sc.cell, pbc=True)
        write(sub + '/POSCAR', s_at, format='vasp', direct=True)
        man.append(dict(index=i, dir=sub))
    json.dump(man, open(d_ + '/disp.json', 'w'), indent=2)
    # 与既有 stage3_displace.py 相同的 disp_info.dat（每行: 序号 原子(1基) dx dy dz）
    with open(d_ + '/disp_info.dat', 'w') as fh:
        for k, row in enumerate(ph.displacements, 1):
            ai, dx, dy, dz = int(row[0]), float(row[1]), float(row[2]), float(row[3])
            fh.write('%d %d %.10f %.10f %.10f\n' % (k, ai + 1, dx, dy, dz))
    vols.append(dict(dir=tag, ratio=r, V=float(a.get_volume()), n_disp=len(scs),
                     n_supercell=len(ph.supercell)))
    print('  %-9s V=%8.3f  超胞 %4d 原子  位移 %3d' % (tag, a.get_volume(), len(ph.supercell), len(scs)))

json.dump(dict(phase=phase, ratios=cfg['ratios'], dim=cfg['dim'], sp=cfg['sp'],
               disp=cfg['disp'], volumes=vols), open(outroot + 'volumes.json', 'w'), indent=2)
print('写出 %svolumes.json（%d 个体积点）' % (outroot, len(vols)))
