#!/usr/bin/env python3
"""本地 phonopy 后处理：11 个 γ 构型（**3×3×3 超胞**）的声子频率、ZPE、轴向 Grüneisen。

素材：gamma_phonons_333/<tag>/{POSCAR, displacements.json, FORCE_SETS}
  POSCAR          = 晶胞（与生成位移时同一结构，契约①：不要重读别的文件）
  FORCE_SETS      = 集群 MACE V6 float32 算的力
本地重建 FC 不需要 phonopy_disp.yaml —— 可由 (POSCAR, displacements.json, FORCE_SETS) 精确重构，
因为 displacements.json 记录的就是 phonopy 生成位移时的 (原子, 位移矢量)。

输出：json/gamma_333_phonons.json（每个构型的 最低模/虚频数/ZPE/F300/轴向 Grüneisen）
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
D = ROOT + 'gamma_phonons_333/'
DIM = [3, 3, 3]
MESH = [8, 8, 8]
SP = 0.05
DT = 4.135667696        # THz -> meV
TAGS = ['eq', 'ap', 'am', 'bp', 'bm', 'cp', 'cm', 'c0945', 'c0935', 'c0925', 'c0915']


def parse_force_sets(path, nsc, ndisp):
    """顺序解析 FORCE_SETS：首两行 = 超胞原子数/位移数；
    之后每个位移块 = 1 行原子序号 + 1 行位移 + nsc 行受力。"""
    toks = [l for l in open(path).read().splitlines() if l.strip()]
    assert int(toks[0]) == nsc, '首行原子数 %s != %d' % (toks[0], nsc)
    assert int(toks[1]) == ndisp, '次行位移数 %s != %d' % (toks[1], ndisp)
    k = 2
    out = []
    for _ in range(ndisp):
        int(toks[k]); k += 1                      # 原子序号（1-based）
        [float(x) for x in toks[k].split()]; k += 1   # 位移矢量
        out.append(np.array([[float(x) for x in toks[k + i].split()] for i in range(nsc)]))
        k += nsc
    return out

out = {}
for tag in TAGS:
    d = D + tag
    a = read(d + '/POSCAR')
    man = json.load(open(d + '/displacements.json'))
    ph = Phonopy(PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                              scaled_positions=a.get_scaled_positions()),
                 supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=SP)
    # 用生成时的位移顺序重建 dataset
    ph.generate_displacements(distance=0.01)
    assert len(ph.supercells_with_displacements) == len(man), \
        '%s 位移数不符: %d vs %d' % (tag, len(ph.supercells_with_displacements), len(man))
    # 读 FORCE_SETS
    forces = parse_force_sets(d + '/FORCE_SETS', len(ph.supercell), len(man))
    ph.forces = forces
    ph.produce_force_constants()
    ph.run_mesh(MESH, with_eigenvectors=False)
    fr = np.array(ph.get_mesh_dict()['frequencies']).ravel()
    ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
    tp = ph.get_thermal_properties_dict()
    T = np.array(tp['temperatures']); F = np.array(tp['free_energy']) / 96.485 / len(ph.primitive)
    i300 = int(np.argmin(abs(T - 300)))
    out[tag] = dict(V=float(a.get_volume()), fmin=float(fr.min()),
                    n_imag=int((fr < -1e-3).sum()), n_modes=int(fr.size),
                    zpe=float(F[0]), F300=float(F[i300]))
    print('  %-7s V=%8.3f  最低模 %+.4f THz  虚频 %5d  模数 %6d  ZPE=%.3f meV/at  F300=%+.3f'
          % (tag, a.get_volume(), fr.min(), out[tag]['n_imag'], fr.size, F[0] * 1000, F[i300] * 1000),
          flush=True)

# ---- 轴向 Grüneisen（6 个应变态）----
print('\n=== 轴向模 Grüneisen（3×3×3）===')
grun = {}
if all(t in out for t in ('ap', 'am', 'bp', 'bm', 'cp', 'cm')):
    Wf = {}
    for tag in ('ap', 'am', 'bp', 'bm', 'cp', 'cm'):
        a = read(D + tag + '/POSCAR')
        man = json.load(open(D + tag + '/displacements.json'))
        ph = Phonopy(PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=np.array(a.cell),
                                  scaled_positions=a.get_scaled_positions()),
                     supercell_matrix=np.diag(DIM), primitive_matrix='auto', symprec=SP)
        ph.generate_displacements(distance=0.01)
        forces = parse_force_sets(D + tag + '/FORCE_SETS', len(ph.supercell), len(man))
        ph.forces = forces
        ph.produce_force_constants()
        ph.run_mesh(MESH, with_eigenvectors=False)
        Wf[tag] = np.array(ph.get_mesh_dict()['frequencies']).ravel()
    for ax in 'abc':
        wp, wm = Wf[ax + 'p'], Wf[ax + 'm']
        ok = (wp > 1e-3) & (wm > 1e-3)
        g = -np.log(wp[ok] / wm[ok]) / (2 * 0.01)
        f = wp[ok]
        grun[ax] = dict(mean=float(g.mean()), std=float(g.std()), n=int(ok.sum()),
                        low=float(g[f < 1.0].mean()) if (f < 1.0).any() else None,
                        rum=float(g[(f >= 0.2) & (f < 0.8)].mean())
                        if ((f >= 0.2) & (f < 0.8)).any() else None,
                        n_rum=int(((f >= 0.2) & (f < 0.8)).sum()))
        print('  沿 %s 轴: <γ>=%+.3f (±%.3f, %d 模) | <1 THz %+.3f | 0.2–0.8 THz(RUM) %+.3f (%d 模)'
              % (ax, grun[ax]['mean'], grun[ax]['std'], ok.sum(),
                 grun[ax]['low'] if grun[ax]['low'] is not None else float('nan'),
                 grun[ax]['rum'] if grun[ax]['rum'] is not None else float('nan'),
                 grun[ax]['n_rum']))
    np.savez(ROOT + 'diag_precision/json/gamma333_freqs.npz', **Wf)

json.dump(dict(phonons=out, gruneisen=grun, dim=DIM, mesh=MESH),
          open(ROOT + 'diag_precision/json/gamma_333_phonons.json', 'w'), indent=1)
print('\n-> diag_precision/json/gamma_333_phonons.json')
