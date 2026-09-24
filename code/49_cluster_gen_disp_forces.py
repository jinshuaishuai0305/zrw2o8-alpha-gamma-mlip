#!/usr/bin/env python3
"""集群侧（mace 环境）：用 LAMMPS 生成位移 + 调 MACE 算力，补 3 个缺失的 γ 体积点。

不需要 phonopy：`phonopy --lammps` 会写 phonopy_disp.yaml + disp-XXX/POSCAR 并生成
一个把力转成 FORCE_SETS 的脚本；我们再调 MACE（ML-IAP）把力算出来。

用法（在作业目录内）：
  POSCAR_RELAXED=<path> TAG=<tag> python 49_cluster_gen_disp_forces.py
"""
import os, sys, glob, json, time, shutil, subprocess, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

TAG = os.environ['TAG']
SRC = os.environ['POSCAR_RELAXED']
OUT = os.environ.get('OUT', './' + TAG)
MODEL = os.environ.get('MACE_MODEL')
DT = os.environ.get('DTYPE', 'float32')
SUPER = os.environ.get('SUPER', '3 3 3')
DISP = os.environ.get('DISP', '0.01')

os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)
t0 = time.time()

# 1) 把弛豫胞写成 phonopy 能读的 POSCAR（保持 P2₁2₁2₁ 的对称化版本）
a = read(SRC)
write('POSCAR_unit', a, format='vasp', direct=True)
print('[%s] 晶胞 V=%.3f  N=%d' % (TAG, a.get_volume(), len(a)), flush=True)

# 2) phonopy 生成位移（--lammps 会写 FORCE_SETS 骨架脚本）
r = subprocess.run(['phonopy', '--lammps', '-d', '--dim', *SUPER.split(),
                    '--amplitude', DISP, '-c', 'POSCAR_unit'],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit('phonopy 失败')
print('[%s] phonopy 生成位移完成 (%.0fs)' % (TAG, time.time() - t0), flush=True)

# 3) 用 MACE 逐个算力
dirs = sorted(glob.glob('disp-*'))
calc = MACECalculator(model_paths=[MODEL], device='cuda', default_dtype=DT)
print('[%s] %d 个位移超胞，开始算力' % (TAG, len(dirs)), flush=True)
for i, d in enumerate(dirs, 1):
    at = read(d + '/POSCAR'); at.calc = calc
    f = at.get_forces()
    np.savetxt(d + '/forces.npy', f)
    if i % 50 == 0:
        print('   %d/%d (%.0fs)' % (i, len(dirs), time.time() - t0), flush=True)

# 4) 用 phonopy 自带的脚本汇编 FORCE_SETS
r = subprocess.run(['phonopy', '-f', *['%s/forces.npy' % d for d in dirs],
                    '-c', 'POSCAR_unit', '--dim', *SUPER.split()],
                   capture_output=True, text=True)
print('[%s] FORCE_SETS: %s' % (TAG, 'OK' if os.path.exists('FORCE_SETS') else 'FAILED'), flush=True)
if not os.path.exists('FORCE_SETS'):
    print(r.stdout[-1500:]); print(r.stderr[-1500:]); sys.exit('FORCE_SETS 未生成')
json.dump(dict(tag=TAG, V=float(a.get_volume()), n_disp=len(dirs)),
          open('summary.json', 'w'), indent=2)
print('[%s] DONE  V=%.3f  位移 %d  FORCE_SETS %.2f MB  (%.0fs)'
      % (TAG, a.get_volume(), len(dirs), os.path.getsize('FORCE_SETS') / 1e6, time.time() - t0))
