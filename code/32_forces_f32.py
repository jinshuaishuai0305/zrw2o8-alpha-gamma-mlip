#!/usr/bin/env python3
"""阶段 02+04（集群 mace 环境）：固定胞弛豫 + 逐位移算力，**float32**。

按 skill codes/phonopy.md §3：力这步用 float32（模型权重本身就是 float32，
float64 在 GPU 上慢 4 倍且无实质收益）。

读:  <root>/volumes.json  +  vol_XXXX/{POSCAR, disp_info.dat, disp-XXX/POSCAR}
写:  vol_XXXX/POSCAR_relaxed + relax_<phase>.json   (若 need_relax)
     vol_XXXX/FORCE_SETS                            (总有)
     summary_<phase>.json

与 qha/stage4_forces.py 的差别（那脚本不能直接用）：
  · 超胞原子数从 POSCAR 读，不写死 1188（γ 是 528）
  · relax 列表与 forces 列表各自独立，容许"已弛豫好、只需算力"的点
  · DTYPE 默认 float32，并打印出来供验收

用法: PHASE=gamma ROOT=<dir> python 32_forces_f32.py
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from mace.calculators import MACECalculator

ROOT = os.environ.get('ROOT', '.')
PHASE = os.environ.get('PHASE', 'gamma')
MODEL = os.environ.get('MACE_MODEL', '/home/jinshuaishuai/workdir_tyut-youzhiyong/'
                                         'jinshuaishuai/AlZrW2O8_MACE/mace_train/'
                                         'AlZrW_v6_stagetwo.model')
DEV = os.environ.get('DEV', 'cuda')
DT = os.environ.get('DTYPE', 'float32')
TOPO = os.environ.get('TOPO', 'volumes.json')

if DT != 'float32':
    raise SystemExit('本作业规定 float32（skill codes/phonopy.md §3），收到 DTYPE=%s；'
                     '如确需其它精度请显式改这一行' % DT)
calc = MACECalculator(model_paths=[MODEL], device=DEV, default_dtype=DT)
print('模型 %s\n  device=%s  dtype=%s  phase=%s  root=%s' % (MODEL, DEV, DT, PHASE, ROOT), flush=True)

vols = json.load(open(os.path.join(ROOT, TOPO)))
if isinstance(vols, dict):
    vols = vols.get('volumes', vols)
print('待处理体积点 %d 个' % len(vols), flush=True)

relax_rec = []
for v in vols:
    d = os.path.join(ROOT, v['dir'])
    t0 = time.time()

    # ---- stage 02：固定胞弛豫离子（只在需要时）----
    if v.get('need_relax'):
        a = read(d + '/POSCAR'); a.calc = calc
        FIRE(a, logfile=None).run(fmax=1e-4, steps=8000)
        a.calc = calc
        E = float(a.get_potential_energy()); fmax = float(np.abs(a.get_forces()).max())
        write(d + '/POSCAR_relaxed', a, format='vasp', direct=True)
        relax_rec.append(dict(dir=v['dir'], ratio=v['ratio'], V=float(a.get_volume()),
                              E=E, fmax=fmax))
        print('  %-9s relax V=%.3f E=%.6f |F|max=%.1e (%.0fs)'
              % (v['dir'], a.get_volume(), E, fmax, time.time() - t0), flush=True)

    # ---- stage 04：逐位移算力 -> FORCE_SETS ----
    info = [l.split() for l in open(d + '/disp_info.dat') if l.strip()]
    blk, nsc = [], None
    for row in info:
        k, ai = int(row[0]), int(row[1])
        dx, dy, dz = map(float, row[2:5])
        a = read('%s/disp-%03d/POSCAR' % (d, k)); a.calc = calc
        f = a.get_forces()
        if nsc is None:
            nsc = len(a)
        elif len(a) != nsc:
            raise SystemExit('%s: 超胞原子数不一致' % d)
        blk += [str(ai), '%.10f %.10f %.10f' % (dx, dy, dz)]
        blk += ['%.10f %.10f %.10f' % (fi[0], fi[1], fi[2]) for fi in f]
        blk.append('')
    open(d + '/FORCE_SETS', 'w').write('%d\n%d\n\n' % (nsc, len(info)) + '\n'.join(blk) + '\n')
    print('  %-9s %3d 个位移 (超胞 %4d 原子) -> FORCE_SETS %.2f MB  (%.0fs)'
          % (v['dir'], len(info), nsc, os.path.getsize(d + '/FORCE_SETS') / 1e6,
             time.time() - t0), flush=True)

if relax_rec:
    json.dump(relax_rec, open(os.path.join(ROOT, 'relax_%s.json' % PHASE), 'w'), indent=2)
json.dump(dict(phase=PHASE, dtype=DT, model=MODEL, n_points=len(vols),
               relaxed=[r['dir'] for r in relax_rec]),
          open(os.path.join(ROOT, 'summary_%s.json' % PHASE), 'w'), indent=2)
print('全部完成 phase=%s  点数=%d  其中弛豫 %d 个' % (PHASE, len(vols), len(relax_rec)))
