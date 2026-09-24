#!/usr/bin/env python3
"""独立评价 V6：在 v3b 训练集（2160 帧，含界面/Al/远离平衡扰动帧）上
比较 v3b / v5b / V6 的能量与力精度 —— 检验"历代最好"是否只在小位移声子帧上成立。
"""
import warnings, sys, json
warnings.filterwarnings('ignore')
import numpy as np, torch
from ase.io import read
from mace.calculators import MACECalculator

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
MODELS = [('v3b', ROOT + 'AlZrW_v3b_stagetwo.model'),
          ('v5b', ROOT + 'AlZrW_v5b_stagetwo.model'),
          ('v6',  ROOT + 'AlZrW_v6_stagetwo.model')]
FRAMES = ROOT + 'train_mace_v3b.xyz'
NMAX = 300          # 每个 (natoms) 组抽样上限；全部 2160 帧太慢

frames_all = read(FRAMES, index=':')
# 按原子数分组，等间隔抽样，保证各相都覆盖
from collections import defaultdict
groups = defaultdict(list)
for i, a in enumerate(frames_all):
    groups[len(a)].append(i)
sel = []
for n, idx in sorted(groups.items()):
    if len(idx) > NMAX:
        step = len(idx) / NMAX
        idx = [idx[int(k * step)] for k in range(NMAX)]
    sel += idx
sel.sort()
frames = [frames_all[i] for i in sel]
print('评估帧数 %d / %d' % (len(frames), len(frames_all)))
print('分组:', {n: sum(1 for i in sel if len(frames_all[i]) == n) for n in sorted(groups)})

def stats(pred, ref):
    d = np.asarray(pred) - np.asarray(ref)
    return float(np.sqrt((d ** 2).mean())), float(d.mean())

results = {}
for tag, path in MODELS:
    calc = MACECalculator(model_paths=path, device='cuda', default_dtype='float64')
    dE, dF = [], []
    by_n = defaultdict(lambda: [[], []])
    for a in frames:
        r = a.copy()
        r.calc = calc
        e_p = r.get_potential_energy() / len(r)
        e_r = a.get_potential_energy() / len(a)
        f_p = r.get_forces().ravel()
        f_r = a.get_forces().ravel()
        dE.append(e_p - e_r)
        dF.append(f_p - f_r)
        by_n[len(a)][0].append(e_p - e_r)
        by_n[len(a)][1].append((f_p - f_r) ** 2)
    dE = np.array(dE) * 1000.0                 # meV/atom
    dF = np.concatenate(dF) * 1000.0           # meV/A
    rmsE, biasE = float(np.sqrt((dE ** 2).mean())), float(dE.mean())
    rmsF = float(np.sqrt((dF ** 2).mean()))
    per = {}
    for n, (e, f2) in sorted(by_n.items()):
        per[n] = dict(
            nframes=len(e),
            rmse_E=float(np.sqrt(np.mean(np.array(e) ** 2)) * 1000),
            bias_E=float(np.mean(e) * 1000),
            rmse_F=float(np.sqrt(np.mean(np.concatenate(f2))) * 1000),
        )
    results[tag] = dict(rmse_E=rmsE, bias_E=biasE, rmse_F=rmsF, per_natoms=per)
    print('\n%-4s  RMSE_E=%.3f  bias_E=%+.3f meV/atom   RMSE_F=%.2f meV/A' % (tag, rmsE, biasE, rmsF))
    for n, v in per.items():
        print('     N=%3d  n=%3d  RMSE_E=%7.3f  bias=%+7.3f  RMSE_F=%7.2f' %
              (n, v['nframes'], v['rmse_E'], v['bias_E'], v['rmse_F']))

json.dump(results, open(ROOT + 'diag_precision/json/eval_v6_on_v3b.json', 'w'), indent=1)

# 训练集自身的帧标签也提示一下"难度"：平均力幅值
print('\n参考：各 N 的标签 |F| 均值 (meV/A)')
for n, idx in sorted(groups.items()):
    fs = np.concatenate([frames_all[i].get_forces().ravel() for i in idx[::max(1, len(idx) // 50)]]) * 1000
    print('   N=%3d  |F|_rms=%.1f' % (n, float(np.sqrt((fs ** 2).mean()))))
