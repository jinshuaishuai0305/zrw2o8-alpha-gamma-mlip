#!/usr/bin/env python3
"""集群半（mace 环境）：**只算力** —— 读位移结构目录，写 FORCE_SETS。

设计约束（skill references/qha-workflow.md §1）：
  SAI 上没有任何环境同时具备 phonopy 与 MACE
  （mace-main: torch/mace/ase，无 phonopy/spglib；phonopy env: phonopy/spglib，无 ase/torch）。
  因此位移生成与后处理必须在本地 phonopy 环境做，集群只做"读 POSCAR → 算力 → 写文件"。
  本脚本**不 import phonopy / spglib**。

输入目录布局：  <root>/<tag>/POSCAR + <root>/<tag>/displacements.json
  displacements.json = [{"atom": <1-based>, "disp": [dx,dy,dz], "dir": "disp-001"}, ...]
  FORCE_SETS 契约②：原子序号单独一行，位移单独一行。

用法: FORCE_ROOT=<dir> TAGS="a,b,c" python 45_forces_only.py
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

ROOT = os.environ.get('FORCE_ROOT', '.')
TAGS = [t for t in os.environ.get('TAGS', '').split(',') if t]
MODEL = os.environ.get('MACE_MODEL', '/home/jinshuaishuai/workdir_tyut-youzhiyong/'
                                         'jinshuaishuai/AlZrW2O8_MACE/mace_train/'
                                         'AlZrW_v6_stagetwo.model')
DT = os.environ.get('DTYPE', 'float32')
DEV = os.environ.get('DEV', 'cuda')
calc = MACECalculator(model_paths=[MODEL], device=DEV, default_dtype=DT)
print('model=%s device=%s dtype=%s' % (os.path.basename(MODEL), DEV, DT), flush=True)

summary = []
for tag in TAGS:
    d = os.path.join(ROOT, tag)
    man = json.load(open(d + '/displacements.json'))
    t0 = time.time()
    blk, nsc = [], None
    for i, m in enumerate(man, 1):
        at = read('%s/%s/POSCAR' % (d, m['dir'])); at.calc = calc
        f = at.get_forces()
        if nsc is None:
            nsc = len(at)
        elif len(at) != nsc:
            raise SystemExit('%s: 超胞原子数不一致' % d)
        blk += [str(m['atom']), '%.10f %.10f %.10f' % tuple(m['disp'])]
        blk += ['%.10f %.10f %.10f' % tuple(fi) for fi in f]
        blk.append('')
    open(d + '/FORCE_SETS', 'w').write('%d\n%d\n\n' % (nsc, len(man)) + '\n'.join(blk) + '\n')
    print('  %-10s %4d 位移 (超胞 %4d 原子) -> FORCE_SETS %.2f MB  (%.0fs)'
          % (tag, len(man), nsc, os.path.getsize(d + '/FORCE_SETS') / 1e6, time.time() - t0), flush=True)
    summary.append(dict(tag=tag, natoms=nsc, ndisp=len(man)))
json.dump(summary, open(os.path.join(ROOT, 'forces_summary.json'), 'w'), indent=2)
print('全部完成：%d 个构型' % len(summary))
