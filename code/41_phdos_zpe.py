#!/usr/bin/env python3
"""PHDOS 与分频段 ZPE 分解（α / γ，**统一 16³ mesh**，每原子归一化）。

目的：回答"ΔZPE = +0.984 meV/atom 来自哪个频段"。现有数据只有最低模频率
      （α 0.21 / γ 0.22 THz，几乎相同），不足以支撑"软模导致"的机制陈述。

口径：phonopy 的 frequencies 形状为 (Nq, Nbands)，Nbands = 3 × 每胞原子数
      （即每原子 3 个模）。因此
        ZPE(每原子) = (1/Nq) Σ_{q,ν} hν/2 / natom
        模数占比   = 频段内模数 / 总模数          （与 mesh 无关，可比）
"""
import os, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, phonopy

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
CASES = {'alpha': dict(d=ROOT + 'qha_f32_out/qha_f32/vol_1000/', n=44),
         'gamma': dict(d=ROOT + 'qha_f32_out/qha_gamma_f32/vol_1000/', n=132)}
THZ_TO_MEV = 4.135667696
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 1e9)]

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', type=int, default=16)
a = ap.parse_args()

out = {}
for ph, c in CASES.items():
    p = phonopy.load(phonopy_yaml=c['d'] + 'phonopy_disp.yaml',
                     force_sets_filename=c['d'] + 'FORCE_SETS', produce_fc=True, log_level=0)
    p.run_mesh([a.mesh] * 3, with_eigenvectors=False)
    fr = np.array(p.get_mesh_dict()['frequencies'])       # (Nq, 3*natom) THz
    nq, nb = fr.shape
    fr = fr[fr > 1e-3]                                    # 去平动零模
    half = fr * THZ_TO_MEV / 2.0                          # meV
    zpe = half.sum() / nq / c['n']
    tot = fr.size
    bands = []
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        bands.append(dict(lo=lo, hi=None if hi > 1e8 else hi,
                          n_modes=int(m.sum()), frac=float(m.sum() / tot),
                          zpe=float(half[m].sum() / nq / c['n'])))
    grid = np.arange(0, 40, 0.1)
    dos, _ = np.histogram(fr, bins=np.append(grid, grid[-1] + 0.1))
    dos = dos / nq / c['n'] / 0.1
    out[ph] = dict(natom=c['n'], nq=nq, nbands=nb, mesh=a.mesh,
                   zpe_total=float(zpe), bands=bands,
                   fmin=float(fr.min()), fmax=float(fr.max()),
                   dos_grid=grid.tolist(), dos=dos.tolist())
    print('[%s] mesh %d³  Nq=%d  Nbands=%d  总模数=%d' % (ph, a.mesh, nq, nb, tot))
    print('   ZPE = %.4f meV/atom   fmin=%.3f THz  fmax=%.2f THz' % (zpe, fr.min(), fr.max()))
    for h in bands:
        print('     %5s–%-5s THz : 模数占比 %5.2f %%   ZPE 贡献 %7.4f meV/at (%5.2f %%)'
              % (h['lo'], '∞' if h['hi'] is None else h['hi'], 100 * h['frac'],
                 h['zpe'], 100 * h['zpe'] / zpe))

za, zg = out['alpha']['zpe_total'], out['gamma']['zpe_total']
dZ = zg - za
print('\n=== ΔZPE(γ−α) = %+.4f meV/atom ===' % dZ)
print('%-16s %12s %12s %12s %10s' % ('频段(THz)', 'α ZPE', 'γ ZPE', 'Δ', '占 ΔZPE'))
for hb, hc in zip(out['gamma']['bands'], out['alpha']['bands']):
    lab = '%s–%s' % (hb['lo'], '∞' if hb['hi'] is None else hb['hi'])
    print('%-16s %12.4f %12.4f %+12.4f %9.1f %%'
          % (lab, hc['zpe'], hb['zpe'], hb['zpe'] - hc['zpe'],
             100 * (hb['zpe'] - hc['zpe']) / dZ))
print('\n低频段（<2 THz）模数占比:  α %.3f %%   γ %.3f %%' %
      (100 * (out['alpha']['bands'][0]['frac'] + out['alpha']['bands'][1]['frac']),
       100 * (out['gamma']['bands'][0]['frac'] + out['gamma']['bands'][1]['frac'])))
json.dump(out, open(ROOT + 'diag_precision/json/phdos_zpe.json', 'w'), indent=1)
print('\n-> diag_precision/json/phdos_zpe.json')
