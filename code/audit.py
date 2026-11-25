#!/usr/bin/env python3
"""ZrW2O8 PBE ΔE(γ−α) 口径审计 —— 一次性复现全部数字。

只读 /home/jss/share/submission/AlZrW2O8/train.xyz 与
     /home/jss/share/AlZrW2O8_MACE/_alpha_ref.vasp / _gamma_ref.vasp
写 ref_audit/audit_results.json。

Part A  溯源：-78808.8861118710 / -236425.6776365414 是什么
Part B  口径2：体积分箱 → 最低 max|F| 帧 → Birch–Murnaghan E0 之差
Part C  口径3：E/N vs F_rms² 联合外推到 |F|→0（唯一有足够低力样本的估计量）
"""
import json
import numpy as np
from ase.io import read
from scipy.optimize import least_squares

TRAIN = '/home/jss/share/submission/AlZrW2O8/train.xyz'
REF = {'a': '/home/jss/share/AlZrW2O8_MACE/_alpha_ref.vasp',
       'g': '/home/jss/share/AlZrW2O8_MACE/_gamma_ref.vasp'}
E_HARD = {'a': -78808.8861118710046867, 'g': -236425.6776365413679741}
NAT = {'a': 44, 'g': 132}
VREF = {'a': 793.79 / 44, 'g': 2263.29 / 132}
GPA = 160.2177      # eV/A^3 -> GPa
KC = 0.1791         # 稿件响应系数 GPa per (meV/atom)
OUT = '/home/jss/share/ZrW2O8-beta/ref_audit/audit_results.json'
R = {}

# ---------------------------------------------------------------- Part A
print('=' * 78)
print('Part A — 溯源')
frames = read(TRAIN, index=':')
rec = {'a': [], 'g': []}
for i, at in enumerate(frames):
    n = len(at); F = at.get_forces(); ph = 'a' if n == 44 else 'g'
    rec[ph].append(dict(idx=i, N=n, E=float(at.get_potential_energy()),
                        e=float(at.get_potential_energy()) / n,
                        V=float(at.get_volume()), v=float(at.get_volume()) / n,
                        f2=float((F ** 2).mean()), fmax=float(np.abs(F).max())))

A = {}
for ph in 'ag':
    d = rec[ph]
    mn = min(d, key=lambda z: z['e'])
    off = (E_HARD[ph] - mn['E']) / NAT[ph]
    cell_same = (np.abs(read(REF[ph]).cell.array - frames[mn['idx']].cell.array).max() < 1e-5 and
                 np.abs(read(REF[ph]).get_scaled_positions(wrap=True) -
                        frames[mn['idx']].get_scaled_positions(wrap=True)).max() < 1e-5)
    A[ph] = dict(n=NAT[ph], nframes=len(d), min_idx=mn['idx'], min_E_total=mn['E'],
                 min_e=mn['e'], min_fmax=mn['fmax'], min_v=mn['v'],
                 E_hard=E_HARD[ph], offset_per_atom=off, ref_cell_is_min_frame=bool(cell_same))
    print(f'  {ph}: 最低能帧 #{mn["idx"]}  E_tot={mn["E"]:.10f} eV  E/N={mn["e"]:.6f}  '
          f'max|F|={mn["fmax"]:.3f}  V/N={mn["v"]:.4f}')
    print(f'     硬编码 E_DFT={E_HARD[ph]:.10f}  ⇒ 差 (E_DFT−E_tot)/N = {off:+.7f} eV/atom')
    print(f'     _ref.vasp 与最低能帧同结构？ {cell_same}')

dE_hard = (E_HARD['g'] / NAT['g'] - E_HARD['a'] / NAT['a']) * 1000
dE1 = (min(rec['g'], key=lambda z: z['e'])['e'] - min(rec['a'], key=lambda z: z['e'])['e']) * 1000
dV1 = A['a']['min_v'] - A['g']['min_v']
print(f'  口径0（稿件硬编码单点）ΔE = {dE_hard:+.4f} meV/at   P = {dE_hard*KC:.3f} GPa')
print(f'  口径1（逐相最低能帧）  ΔE = {dE1:+.4f} meV/at   ΔV = {dV1:.5f} A3/at   '
      f'P = {dE1/1000/dV1*GPA:.3f} GPa')
print(f'  ⇒ 两者相差 {dE1-dE_hard:+.4f} meV/at；硬编码值 = 最低能帧 + 常数 {A["a"]["offset_per_atom"]:+.7f} eV/atom')
R['partA'] = dict(per_phase=A, dE_hardcoded=dE_hard, dE_min_frames=dE1, dV_min_frames=dV1,
                  P_hardcoded=dE_hard * KC, P_min_frames=dE1 / 1000 / dV1 * GPA)

# ---------------------------------------------------------------- Part B
print('\n' + '=' * 78)
print('Part B — 口径2：体积分箱(0.15 A3/at) → 每箱最低 max|F| 帧 → BM3 E(V) E0 之差')
def bm3(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))

def bm_fit(pts):
    V = np.array([q['v'] for q in pts]); E = np.array([q['e'] for q in pts])
    i = int(V.argmin())
    sol = least_squares(lambda p: bm3(V, *p) - E, [E[i], V[i], 1.0, 4.0], max_nfev=20000)
    r = bm3(V, *sol.x) - E
    return sol.x, float(np.sqrt((r ** 2).mean()) * 1000)

B = {}
for frac in (0.04, 0.06, 0.08):
    B[f'win{frac}'] = {}
    for ph in 'ag':
        lo, hi = VREF[ph] * (1 - frac), VREF[ph] * (1 + frac)
        sel = [q for q in rec[ph] if lo <= q['v'] <= hi]
        edges = np.arange(lo, hi + 1e-9, 0.15)
        pts = []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            cand = [q for q in sel if a_ <= q['v'] < b_]
            if len(cand) >= 3:
                pts.append(min(cand, key=lambda q: q['fmax']))
        p, rms = bm_fit(pts)
        B[f'win{frac}'][ph] = dict(n_bins=len(pts), E0=float(p[0]), V0=float(p[1]),
                                   B0_GPa=float(p[2]) * GPA, Bp=float(p[3]), rms_meV=rms,
                                   v_window=[lo, hi],
                                   fmax_of_selected=[round(q['fmax'], 2) for q in pts])
    pa, pg = B[f'win{frac}']['a'], B[f'win{frac}']['g']
    d = (pg['E0'] - pa['E0']) * 1000
    print(f'  窗宽 ±{frac*100:.0f}%: α {pa["n_bins"]}箱 V0={pa["V0"]:.4f} B0={pa["B0_GPa"]:.0f} '
          f'rms={pa["rms_meV"]:.2f} | γ {pg["n_bins"]}箱 V0={pg["V0"]:.4f} '
          f'B0={pg["B0_GPa"]:.0f} Bp={pg["Bp"]:.1f} rms={pg["rms_meV"]:.2f} | ΔE0={d:+.2f} meV/at')
B['dE0_by_window'] = {k: float((B[k]['g']['E0'] - B[k]['a']['E0']) * 1000)
                      for k in ('win0.04', 'win0.06', 'win0.08')}
B['dV0_by_window'] = {k: float(B[k]['a']['V0'] - B[k]['g']['V0'])
                      for k in ('win0.04', 'win0.06', 'win0.08')}
_g = [B[k]['g'] for k in ('win0.04', 'win0.06', 'win0.08')]
print(f'  判定：γ 侧 rms {min(q["rms_meV"] for q in _g):.1f}–{max(q["rms_meV"] for q in _g):.1f} meV/at、'
      f'B0 {min(q["B0_GPa"] for q in _g):.0f}–{max(q["B0_GPa"] for q in _g):.0f} GPa、'
      f'Bp {min(q["Bp"] for q in _g):.0f}–{max(q["Bp"] for q in _g):.0f}、'
      f'ΔE0 {min(B["dE0_by_window"].values()):.1f}–{max(B["dE0_by_window"].values()):.1f} meV/at ')
print('  ⇒ 三个窗宽下结果不收敛（α 侧 rms 仅 0.4–0.5 meV/at，问题全在 γ 侧无近平衡帧）'
      ' ⇒ 3 次尝试后判定：口径2 在本数据上不适用')
R['partB'] = B

# ---------------------------------------------------------------- Part C
print('\n' + '=' * 78)
print('Part C — 口径3：联合外推 E/N = a_ph + c·F_rms² + d1·x + d2·x²  (x = V/N − V_ref)')
def solve(da, dg, ca, cg, win=0.03):
    sa = [r for r in da if r['fmax'] < ca and abs(r['v'] / VREF['a'] - 1) < win]
    sg = [r for r in dg if r['fmax'] < cg and abs(r['v'] / VREF['g'] - 1) < win]
    E = np.array([r['e'] for r in sa + sg]); F2 = np.array([r['f2'] for r in sa + sg])
    X = np.array([r['v'] - VREF[r_ph] for r, r_ph in
                  [(r, 'a') for r in sa] + [(r, 'g') for r in sg]])
    ia = np.array([1.0] * len(sa) + [0.0] * len(sg)); ig = 1 - ia
    def res(p):
        aa, ag, c, d1a, d1g, d2 = p
        return aa * ia + ag * ig + c * F2 + d1a * ia * X + d1g * ig * X + d2 * X * X - E
    sol = least_squares(res, [min(z['e'] for z in sa), min(z['e'] for z in sg), 0.05, 0, 0, 2.5],
                        max_nfev=40000)
    return sol.x, float(np.sqrt((res(sol.x) ** 2).mean()) * 1000), len(sa), len(sg)

p, rms, na, ng = solve(rec['a'], rec['g'], 1.5, 3.0)
dE3 = (p[1] - p[0]) * 1000
ea = min(rec['a'], key=lambda z: z['e']); eg = min(rec['g'], key=lambda z: z['e'])
print(f'  样本 α(Na={na}, fmax<1.5) + γ(Ng={ng}, fmax<3.0)；rms = {rms:.2f} meV/at')
print(f'  a(α)={p[0]:.6f}  a(γ)={p[1]:.6f}  c={p[2]:.4f} eV/(eV/A)²  '
      f'd1=({p[3]:.4f},{p[4]:.4f})  d2={p[5]:.3f}')
print(f'  零力极限：α {ea["e"]:.6f} → {p[0]:.6f}（降 {1000*(ea["e"]-p[0]):.2f} meV/at）；'
      f'γ {eg["e"]:.6f} → {p[1]:.6f}（降 {1000*(eg["e"]-p[1]):.2f} meV/at）')
print(f'  口径3 ΔE = {dE3:+.3f} meV/at   P_t = {dE3*KC:.3f} GPa')

rng = np.random.default_rng(11)
bs = []
for _ in range(400):
    da = [rec['a'][i] for i in rng.integers(0, len(rec['a']), len(rec['a']))]
    dg = [rec['g'][i] for i in rng.integers(0, len(rec['g']), len(rec['g']))]
    try:
        q, *_ = solve(da, dg, 1.5, 3.0)
        bs.append((q[1] - q[0]) * 1000)
    except Exception:
        pass
bs = np.array(bs)
print(f'  bootstrap(400): {bs.mean():+.3f} ± {bs.std():.3f} meV/at；16–84% = '
      f'{np.percentile(bs,16):+.2f} … {np.percentile(bs,84):+.2f}')
sens = {}
for ca, cg in ((1.2, 2.5), (1.5, 3.0), (1.5, 3.5), (2.0, 4.0)):
    q, rm, na2, ng2 = solve(rec['a'], rec['g'], ca, cg)
    sens[f'{ca}/{cg}'] = dict(dE_meV_at=float((q[1] - q[0]) * 1000), n=[na2, ng2], rms=rm)
    print(f'  灵敏度 fmax<{ca}/{cg}: n={na2}/{ng2} rms={rm:.2f} → ΔE={(q[1]-q[0])*1000:+.3f} meV/at')
R['partC'] = dict(a_alpha=float(p[0]), a_gamma=float(p[1]), c=float(p[2]),
                  d1=[float(p[3]), float(p[4])], d2=float(p[5]), rms_meV=rms, n=[na, ng],
                  dE_meV_at=float(dE3), P_GPa=float(dE3 * KC),
                  relax_alpha_meV=float((ea['e'] - p[0]) * 1000),
                  relax_gamma_meV=float((eg['e'] - p[1]) * 1000),
                  bootstrap=dict(mean=float(bs.mean()), sd=float(bs.std()),
                                 p16=float(np.percentile(bs, 16)),
                                 p84=float(np.percentile(bs, 84)), n=len(bs)),
                  sensitivity=sens)
R['summary'] = dict(
    dE_frame=dE1, P_frame=dE1 / 1000 / dV1 * GPA,
    dE_zero_force=float(dE3), P_zero_force=float(dE3 * KC),
    shift_meV=dE1 - float(dE3), shift_GPa=dE1 * KC - float(dE3 * KC),
    delta_DFT_manuscript=6.257,
    protocol_component=f'{(dE1-float(dE3)):.2f} (bootstrap 16-84%: {dE1-np.percentile(bs,84):.2f}-{dE1-np.percentile(bs,16):.2f})',
    functional_component=f'{6.257-(dE1-float(dE3)):.2f}')
json.dump(R, open(OUT, 'w'), indent=1, default=float)
print('\nwrote', OUT)
