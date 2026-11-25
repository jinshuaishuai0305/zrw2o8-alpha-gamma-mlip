#!/usr/bin/env python3
"""stage_eos.py — V11 固定胞（只动原子）E_st(V) + γ 硬化/重构闸门。

网格：以 ABACUS 真弛豫 V0 为中心，V/V0 = linspace(0.94, 1.06, N)（等间距）。
每点：读参考胞 -> 等比缩放到目标 V -> FIRE 只弛豫离子（fmax 1e-4，float64）
      -> 记录 E、末态 max|F|、与起始几何的 max|Δpos|（最小像）、W–O 配位数分布。
每个点单独落 eos_point.json（可续跑：REUSE=1 时已有点直接跳过）。

闸门（写进 eos_raw.json，由分析脚本据此剔除）：
  cn_shift   : 2.2 Å 截断下 W 的 CN 直方图与同相 s=1.00 点相比，任一 bin 变化 ≥2 个 W
  dpos_large : max|Δpos| > 0.5 Å（WO4->WO5 的 O 迁移是 0.8–1.5 Å 量级）

用法: QHA_ROOT=<.../cluster_qha_v11> PHASES=alpha,gamma NGRID=13 python stage_eos.py
"""
import os, sys, json, time, hashlib, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

ROOT = os.environ['QHA_ROOT']
MODEL = os.environ.get('MACE_MODEL',
                       '/home/jinshuaishuai/workdir_tyut-youzhiyong/jinshuaishuai/'
                       'cluster_v11/ZrW_v11_seed3_stagetwo.model')
INP = os.path.join(ROOT, 'input')
DEV = os.environ.get('DEV', 'cuda')
DT = os.environ.get('DTYPE', 'float64')
FMAX = float(os.environ.get('FMAX', 1e-4))
STEPS = int(os.environ.get('STEPS', 20000))
NGRID = int(os.environ.get('NGRID', 13))
REUSE = os.environ.get('REUSE', '1') == '1'
PHASES = os.environ.get('PHASES', 'alpha,gamma').split(',')
ONLY_S = [float(x) for x in os.environ['ONLY_S'].split(',')] if os.environ.get('ONLY_S') else None

# ABACUS 真弛豫 V0（ref_audit/abacus_runs/abacus_results.json）
PHASE_INFO = {
    'alpha': dict(ref='_alpha_ref.vasp', natoms=44, V0=793.7913,
                  V0_src='ref_audit/abacus_runs/abacus_results.json:'
                         'statics/alpha_relaxed/volume_A3'),
    'gamma': dict(ref='_gamma_ref.vasp', natoms=132, V0=2263.2869,
                  V0_src='ref_audit/abacus_runs/abacus_results.json:'
                         'statics/gamma_final_100/volume_A3'),
}
CN_CUT = 2.2
DPOS_GATE = 0.5
CN_GATE = 2

calc = MACECalculator(model_paths=[MODEL], device=DEV, default_dtype=DT)
print('model=%s dtype=%s device=%s NGRID=%d fmax=%g reuse=%s'
      % (MODEL, DT, DEV, NGRID, FMAX, REUSE), flush=True)


def wo_cn(at, cut=CN_CUT):
    sym = np.array(at.get_chemical_symbols())
    i, j, d = neighbor_list('ijd', at, cut)
    m = (sym[i] == 'W') & (sym[j] == 'O')
    cnt = Counter(i[m])
    W = np.where(sym == 'W')[0]
    hist = Counter(int(cnt.get(w, 0)) for w in W)
    bw = d[m]
    return dict(cutoff=cut, n_W=int(len(W)),
                hist={str(k): int(v) for k, v in sorted(hist.items())},
                bond_min=float(bw.min()), bond_mean=float(bw.mean()),
                bond_max=float(bw.max()))


def max_dpos(a, b):
    df = a.get_scaled_positions(wrap=False) - b.get_scaled_positions(wrap=False)
    df -= np.round(df)
    return float(np.linalg.norm(df @ a.get_cell()[:], axis=1).max())


out = {'model': MODEL, 'dtype': DT, 'fmax_target': FMAX, 'ngrid': NGRID,
       'grid_rule': 'V/V0 = linspace(0.94, 1.06, NGRID) equally spaced',
       'cn_cutoff_A': CN_CUT, 'gate_rules': dict(cn_shift='>=2 W change in any CN bin vs s=1.00',
                                                 dpos_large='>%.1f A' % DPOS_GATE),
       'phases': {}}

for ph in PHASES:
    info = PHASE_INFO[ph]
    nat = info['natoms']
    base = read(os.path.join(INP, info['ref']))
    d = os.path.join(ROOT, 'eos', ph)
    os.makedirs(d, exist_ok=True)
    pts = []
    grid = np.linspace(0.94, 1.06, NGRID)
    if ONLY_S:
        grid = np.array([g for g in grid if any(abs(g - o) < 1e-6 for o in ONLY_S)])
    for k, s in enumerate(grid):
        tag = 'vol_%0.4f' % s
        wd = os.path.join(d, tag)
        os.makedirs(wd, exist_ok=True)
        rec_f = os.path.join(wd, 'eos_point.json')
        if REUSE and os.path.exists(rec_f) and os.path.exists(os.path.join(wd, 'poscar.json')):
            p = json.load(open(rec_f))
            pts.append(p)
            print('  [%s] %s CACHED E/at=%.8f' % (ph, tag, p['E_per_atom']), flush=True)
            continue
        Vt = info['V0'] * s
        at = base.copy()
        at.set_cell(base.get_cell()[:] * (Vt / base.get_volume()) ** (1. / 3.), scale_atoms=True)
        start = at.copy()
        V = at.get_volume()
        at.calc = calc
        t0 = time.time()
        opt = FIRE(at, logfile=None, loginterval=200, maxstep=0.1)
        opt.run(fmax=FMAX, steps=STEPS)
        dt = time.time() - t0
        at.calc = calc
        E = float(at.get_potential_energy())
        fm = float(np.abs(at.get_forces()).max())
        dp = max_dpos(at, start)
        cn = wo_cn(at)
        write(os.path.join(wd, 'POSCAR_relaxed'), at, format='vasp', direct=True)
        json.dump(dict(cell=np.asarray(at.get_cell()).tolist(),
                       symbols=list(at.get_chemical_symbols()),
                       scaled_positions=np.asarray(at.get_scaled_positions()).tolist(),
                       natoms=nat), open(os.path.join(wd, 'poscar.json'), 'w'))
        p = dict(s=float(s), tag=tag, V_target=float(Vt), V=float(V), natoms=nat,
                 E_eV=E, E_per_atom=E / nat, fmax_end=fm, nsteps=int(opt.nsteps),
                 converged=bool(fm <= FMAX), sec=round(dt, 1),
                 max_dpos_A=dp, wo_cn=cn)
        json.dump(p, open(rec_f, 'w'), indent=1)
        pts.append(p)
        print('  [%s] s=%.3f V=%8.3f E/at=%.8f |F|max=%.2e dpos=%.3f CN=%s (%d steps, %.0fs)'
              % (ph, s, V, E / nat, fm, dp, cn['hist'], opt.nsteps, dt), flush=True)

    i1 = int(np.argmin([abs(p['s'] - 1.0) for p in pts]))
    ref_hist = pts[i1]['wo_cn']['hist']
    keys = set(ref_hist) | {k for p in pts for k in p['wo_cn']['hist']}
    for p in pts:
        h = p['wo_cn']['hist']
        shift = {k: int(h.get(k, 0)) - int(ref_hist.get(k, 0)) for k in keys}
        p['cn_shift_vs_s1'] = shift
        p['gate_cn_shift'] = bool(max(abs(v) for v in shift.values()) >= CN_GATE)
        p['gate_dpos'] = bool(p['max_dpos_A'] > DPOS_GATE)
        p['restructured'] = bool(p['gate_cn_shift'] or p['gate_dpos'])
        p['flag'] = '+'.join([x for x, c in
                              (('cn_shift', p['gate_cn_shift']), ('dpos_large', p['gate_dpos']))
                              if c]) or 'clean'
    out['phases'][ph] = dict(V0_A3=info['V0'], V0_src=info['V0_src'], natoms=nat,
                             ref=os.path.join(INP, info['ref']),
                             cn_ref_hist_at_s1=ref_hist, points=pts)
    print('[%s] 闸门命中 %d / %d' % (ph, sum(p['restructured'] for p in pts), len(pts)),
          flush=True)

json.dump(out, open(os.path.join(ROOT, 'eos', 'eos_raw.json'), 'w'), indent=1)
print('-> %s/eos/eos_raw.json' % ROOT)
