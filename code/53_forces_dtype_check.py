#!/usr/bin/env python3
"""精度检验：同一构型的力用 float32 与 float64 各算一遍，比较 ZPE / F(300K)。
样本：α vol_1000（22 位移，1188 原子）、γ vol_1000（198 位移，3564 原子）。
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
ROOT = os.environ['ROOT']
CASES = [('alpha', ROOT + '/qha_f32_v6/qha_f32/vol_1000', 'disp_info.dat'),
         ('gamma', ROOT + '/qha_gamma_333/vol_1000', 'displacements.json')]
out = {}
for tag, d, fmt in CASES:
    if fmt == 'disp_info.dat':
        man = []
        for l in open(d + '/disp_info.dat'):
            f = l.split()
            if len(f) >= 5:
                man.append(dict(atom=int(f[1]), dir='disp-%03d' % int(f[0])))
    else:
        man = json.load(open(d + '/displacements.json'))
    rec = {}
    for dt in ('float32', 'float64'):
        calc = MACECalculator(model_paths=ROOT + '/mace_train/AlZrW_v6_stagetwo.model',
                              device='cuda', default_dtype=dt)
        t0 = time.time(); fs = []
        for m in man:
            at = read('%s/%s/POSCAR' % (d, m['dir'])); at.calc = calc
            fs.append(at.get_forces())
        F = np.array(fs)
        rec[dt] = dict(n=len(man), t=time.time() - t0,
                       Fsum=float(np.abs(F).sum()), Fmax=float(np.abs(F).max()))
        print('  %-6s %-8s %3d 位移  %.1fs  Σ|F|=%.6f eV/Å  max|F|=%.6f'
              % (tag, dt, len(man), time.time() - t0, rec[dt]['Fsum'], rec[dt]['Fmax']), flush=True)
    dF = abs(rec['float32']['Fsum'] - rec['float64']['Fsum'])
    rel = dF / rec['float64']['Fsum']
    rec['dFsum'] = dF; rec['rel'] = rel
    print('    Δ(Σ|F|) = %.2e eV/Å   相对差 %.2e   ⟹ 每位移平均力偏差 %.2e eV/Å'
          % (dF, rel, dF / (rec['float64']['n'] * 3)), flush=True)
    out[tag] = rec
json.dump(out, open(os.environ.get('OUTF', '/tmp/dtype_check.json'), 'w'), indent=1)
print('写出', os.environ.get('OUTF', '/tmp/dtype_check.json'))
