#!/usr/bin/env python3
"""把 5 个压缩端 γ 体积点在 **float64** 下重弛豫离子，取得与归档一致的静态能 E(V)。

原因：float32 弛豫给出系统性偏低 1.2–4.7 meV/atom 的能量（实测），
      而相变线要分辨的信号只有 1–3 meV/atom，必须与归档 float64 同一能量尺度。
几何不变（固定胞、只动离子），因此位移结构无需重新生成。
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'qha_gamma_f32/'
calc = MACECalculator(model_paths=ROOT + 'AlZrW_v6_stagetwo.model',
                      device='cuda', default_dtype='float64')
recs = []
for r in json.load(open(OUT + 'relax_compression.json')):
    d = OUT + r['dir']
    a = read(d + '/POSCAR')          # 缩放后的起点（几何与 float32 版完全一致）
    a.calc = calc
    t0 = time.time()
    FIRE(a, logfile=None).run(fmax=1e-5, steps=8000)
    a.calc = calc
    E = float(a.get_potential_energy()); fmax = float(np.abs(a.get_forces()).max())
    write(d + '/POSCAR_relaxed', a, format='vasp', direct=True)
    recs.append(dict(dir=r['dir'], ratio=r['ratio'], V=float(a.get_volume()), E=E, fmax=fmax))
    print('  %-9s V=%9.3f  E=%.6f  (%.6f eV/at)  |F|max=%.1e  (%.0fs)'
          % (r['dir'], a.get_volume(), E, E / 132, fmax, time.time() - t0), flush=True)
json.dump(recs, open(OUT + 'relax_compression_f64.json', 'w'), indent=2)
print('-> relax_compression_f64.json')
