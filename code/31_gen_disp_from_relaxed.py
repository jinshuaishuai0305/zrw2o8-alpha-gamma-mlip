#!/usr/bin/env python3
"""阶段 03（本地 phonopy 环境）：为 20 个 QHA 体积点生成「位移超胞 POSCAR」（force 之前）。

严格照 skill `references/qha-workflow.md` 的顺序做：
    relax 后的胞  ->  spglib 对称化  ->  生成位移
顺序不能颠倒：DAC 弛豫残留 0.03 Å 畸变，若在对称化之前缩放/生成位移，
P2₁3 会掉到 R3，α 的位移数从 22 涨到 88（本机实测）。

起始几何用「归档的弛豫胞」（已验证为对称化后的纯净胞）：
  α 7 点 : qha/vol_*/POSCAR_relaxed                 (1=O 2=W 3=Zr)
  γ 10 点: qha_gamma/vol_0985..1015/POSCAR_relaxed
           qha_gamma_v6f64/vol_0955..0975/POSCAR_relaxed
  γ 3 点 : 1.025 / 1.035 / 1.045 归档里没有纯净胞，由 _gamma_ref 对称化 + 按体积缩放得到
           （这三点的离子会在集群上按 float32 弛豫：stage2_relax.py）

输出（每个体积点）：POSCAR, phonopy_disp.yaml, disp.json, disp_info.dat, disp-XXX/POSCAR
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
import spglib
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
SP = {'alpha': 0.08, 'gamma': 0.05}
DIM = {'alpha': [3, 3, 3], 'gamma': [2, 2, 1]}
SG = {'alpha': 198, 'gamma': 19}
OUT = {'alpha': ROOT + 'qha_f32/', 'gamma': ROOT + 'qha_gamma_f32/'}

# 起始胞（归档的弛豫胞）
SRC = {'alpha': {r: ROOT + 'qha/vol_%04d/POSCAR_relaxed' % int(round(r * 1000))
                 for r in [0.975, 0.980, 1.000, 1.020, 1.040, 1.060, 1.080]},
       'gamma': {}}
for r in [0.985, 0.990, 0.995, 1.000, 1.005, 1.010, 1.015]:
    SRC['gamma'][r] = ROOT + 'qha_gamma/vol_%04d/POSCAR_relaxed' % int(round(r * 1000))
for r in [0.955, 0.965, 0.975]:
    SRC['gamma'][r] = ROOT + 'qha_gamma_v6f64/vol_%04d/POSCAR_relaxed' % int(round(r * 1000))
NEED_RELAX = {'gamma': [1.025, 1.035, 1.045]}      # 归档缺，需集群弛豫


def symmetrize(a, sp, sg, origin_ref=None):
    lat, pos, num = spglib.standardize_cell((a.cell[:], a.get_scaled_positions(), a.numbers),
                                            to_primitive=False, no_idealize=False, symprec=sp)
    s = Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)
    if origin_ref is not None:
        d = s.get_scaled_positions() - origin_ref.get_scaled_positions()
        d -= np.round(d)
        ang = 2 * np.pi * d
        shift = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0)) / (2 * np.pi)
        s.set_scaled_positions((s.get_scaled_positions() - shift) % 1.0)
    ds = spglib.get_symmetry_dataset((s.cell[:], s.get_scaled_positions(), s.numbers), symprec=sp)
    if ds.number != sg:
        raise SystemExit('对称化后不是 #%d，而是 %s' % (sg, ds.international))
    return s, ds


def emit(phase, ratio, base):
    tag = 'vol_%04d' % int(round(ratio * 1000))
    d = os.path.join(OUT[phase], tag)
    os.makedirs(d, exist_ok=True)
    a, ds = symmetrize(base, SP[phase], SG[phase])
    write(d + '/POSCAR', a, format='vasp', direct=True)
    pa = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                      scaled_positions=a.get_scaled_positions())
    ph = Phonopy(pa, supercell_matrix=np.diag(DIM[phase]), primitive_matrix='auto', symprec=SP[phase])
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
    print('  %-9s %-8s V=%8.3f  超胞 %4d 原子  位移 %3d'
          % (tag, ds.international, a.get_volume(), len(ph.supercell), len(scs)), flush=True)
    return dict(dir=tag, ratio=ratio, V=float(a.get_volume()), n_disp=len(scs),
                n_supercell=len(ph.supercell), need_relax=False)


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    recs = []
    for ph in (['alpha', 'gamma'] if phase == 'all' else [phase]):
        os.makedirs(OUT[ph], exist_ok=True)
        print('%s: 起始几何 = 归档弛豫胞（已对称化）' % ph, flush=True)
        for r, f in sorted(SRC[ph].items()):
            if not os.path.exists(f):
                print('  缺 %s，跳过' % f); continue
            try:
                recs.append(emit(ph, r, read(f)))
            except SystemExit as e:
                print('  !! %s 跳过: %s' % (r, e)); continue
        # 每相结束就写一次，避免后面某点失败导致前面白跑
        json.dump([x for x in recs if x['dir'] in
                   [os.path.basename(d) for d in SRC[ph]] or x.get('need_relax')],
                  open(OUT[ph] + 'volumes.json', 'w'), indent=2)
        for r in NEED_RELAX.get(ph, []):
            ref = read(ROOT + '_gamma_ref.vasp')
            ref_sym, _ = symmetrize(ref, SP[ph], SG[ph], origin_ref=ref)
            s = r ** (1 / 3)
            a = ref_sym.copy(); a.set_cell(ref_sym.cell[:] * s, scale_atoms=False)
            try:
                rec = emit(ph, r, a)
                rec['need_relax'] = True
                recs.append(rec)
            except SystemExit as e:
                print('  !! %s 跳过: %s' % (r, e)); continue
    json.dump(recs, open(ROOT + 'volume_points_f32.json', 'w'), indent=2)
    print('\n共 %d 个体积点 -> volume_points_f32.json' % len(recs))
    print('其中需集群先弛豫的: %s'
          % [r['dir'] for r in recs if r['need_relax']])
