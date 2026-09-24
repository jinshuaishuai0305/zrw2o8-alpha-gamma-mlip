#!/usr/bin/env python3
"""阶段 06（本地）：G_α(T,P)=G_γ(T,P) 的相变线 —— **不用 EOS 拟合形式**。

为什么不用 Vinet/BM：
  α 的体积网格跨 ±5 %、E 跨度只有 25 meV/atom，Vinet 在这种宽范围上拟合退化
  （实测 B0=2.4 GPa、Bp=13.6，明显非物理），据此求出的 G(T,P) 交点全是 0（假）。
  本脚本改为：对原始 (V, F) 点做**三次样条**，
      P(V,T)      = −dF/dV                （样条解析导数）
      B(V,T)      =  V d²F/dV²            （顺便给出，用于自检）
      G(T,P)      =  F(V*) + P·V*，  V* 由 P(V*)=P 二分求得
  完全由数据决定，不引入任何函数形式假设。

输入：qha_f32_out/{qha_f32,qha_gamma_f32}/vol_*/thermal_properties.yaml（集群 float32 力）
      E(V)：--e-from float64 → 归档弛豫能；float32 → 集群重算能
输出：diag_precision/json/pt_line_final.json + 屏幕表
"""
import os, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, yaml
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'qha_f32_out/'
NAT = {'alpha': 44, 'gamma': 132}
DIRS = {'alpha': OUT + 'qha_f32', 'gamma': OUT + 'qha_gamma_f32'}
SKIP_IMAG = {'alpha': {'vol_0975'}, 'gamma': {'vol_0955'}}   # 有虚频，不进 QHA
GPA = 160.21766208
E_DFT = {'alpha': -78808.8861118710, 'gamma': -236425.6776365414}
DE_DFT = (E_DFT['gamma'] / NAT['gamma'] - E_DFT['alpha'] / NAT['alpha']) * 1000

ap = argparse.ArgumentParser()
ap.add_argument('--e-from', default='float64', choices=['float32', 'float64'])
args = ap.parse_args()


def load_e():
    E = {'alpha': {}, 'gamma': {}}
    if args.e_from == 'float64':
        for r in json.load(open(ROOT + 'qha/relax.json')):
            E['alpha'][r['dir']] = r['E']
        for f in (ROOT + 'qha_gamma/relax_v6_7point_backup.json',
                  ROOT + 'qha_gamma_v6f64/relax.json'):
            if os.path.exists(f):
                for r in json.load(open(f)):
                    E['gamma'][r['dir']] = r['E']
    else:
        for ph, f in (('alpha', OUT + 'relax_alpha.json'), ('gamma', OUT + 'relax_gamma.json')):
            if os.path.exists(f):
                for r in json.load(open(f)):
                    E[ph][r['dir']] = r['E']
    return E


def load_phase(phase, E):
    vols = json.load(open(DIRS[phase] + '/volumes.json'))
    V, Est, Fv, T = [], [], [], None
    for v in sorted(vols, key=lambda x: x['V']):
        if v['dir'] in SKIP_IMAG[phase]:
            continue
        if v['dir'] not in E[phase]:
            continue
        d = DIRS[phase] + '/' + v['dir']
        y = yaml.safe_load(open(d + '/thermal_properties.yaml'))
        tp = y['thermal_properties']
        T = np.array([e['temperature'] for e in tp])
        Fv.append(np.array([e['free_energy'] for e in tp]) / 96.485 / NAT[phase])   # eV/atom
        V.append(v['V'] / NAT[phase]); Est.append(E[phase][v['dir']] / NAT[phase])
    return np.array(V), np.array(Est), np.array(Fv), T


if __name__ == '__main__':
    E = load_e()
    print('E(V) 来源: %s   α %d 点  γ %d 点' % (args.e_from, len(E['alpha']), len(E['gamma'])))
    dat = {}
    for ph in ('alpha', 'gamma'):
        V, Est, Fv, T = load_phase(ph, E)
        dat[ph] = dict(V=V, Est=Est, Fv=Fv, T=T)
        # 静态 E(V) 样条
        se = CubicSpline(V, Est)
        Vm = V[np.argmin(Est)]
        B = Vm * se(Vm, 2) * GPA
        print('[%s] %d 个稳定体积点  V=%.3f..%.3f Å³/at' % (ph, len(V), V.min(), V.max()))
        print('     静态: V0=%.4f Å³/at  E0=%.6f eV/at  B(V0)=%.1f GPa   dE/dV(V0)=%.4f GPa'
              % (Vm, float(se(Vm)), B, -se(Vm, 1) * GPA))

    # 0 K 静态：两相 E(V) 的公切线（等压共存）
    sa = CubicSpline(dat['alpha']['V'], dat['alpha']['Est'])
    sg = CubicSpline(dat['gamma']['V'], dat['gamma']['Est'])

    def dP(V, s):
        return -s(V, 1) * GPA

    def eqs(x):
        Va, Vg = x
        return [dP(Va, sa) - dP(Vg, sg),
                (sa(Va) - Va * sa(Va, 1)) - (sg(Vg) - Vg * sg(Vg, 1))]

    from scipy.optimize import fsolve
    sol, info, ier, msg = fsolve(eqs, [dat['alpha']['V'][np.argmin(dat['alpha']['Est'])],
                                       dat['gamma']['V'][np.argmin(dat['gamma']['Est'])]],
                                 full_output=True)
    P0 = -sa(sol[0], 1) * GPA
    print('\n=== 0 K 静态公切线 ===')
    print('  V_α=%.4f  V_γ=%.4f Å³/at   共存压力 P_eq(0 K) = %+.3f GPa   (ier=%d)'
          % (sol[0], sol[1], P0, ier))
    E0a = float(sa(sol[0]) - sol[0] * sa(sol[0], 1))
    E0g = float(sg(sol[1]) - sol[1] * sg(sol[1], 1))
    print('  静态能差（共同切线截距）: ΔE(γ−α) = %+.3f meV/atom' % ((E0g - E0a) * 1000))
    print('  DFT/PBE 静态能差:          ΔE(γ−α) = %+.3f meV/atom' % DE_DFT)

    # ---- 有限温：F(V,T) 样条 -> P(V,T) -> G(T,P) ----
    def make_G(ph):
        V, Fv, T = dat[ph]['V'], dat[ph]['Fv'], dat[ph]['T']
        def G(T_, P_):
            it = int(np.argmin(abs(T - T_)))
            s = CubicSpline(V, Fv[:, it])
            # 解 P(V) = P_
            f = lambda Vv: -s(Vv, 1) * GPA - P_
            lo, hi = V.min() * 0.97, V.max() * 1.03
            Vst = brentq(f, lo, hi, xtol=1e-12)
            return float(s(Vst) + P_ / GPA * Vst), float(Vst)
        return G

    Ga, Gg = make_G('alpha'), make_G('gamma')
    print('\n=== 相变线（F(V,T) 三次样条，无 EOS 假设）===')
    print('%6s | %13s | %11s %11s | %10s' % ('T(K)', 'P_t(GPa)', 'V_α(Å³/at)', 'V_γ(Å³/at)', 'ΔG(0GPa)'))
    rows = []
    for T_ in [0, 100, 200, 300, 350, 393, 400, 500, 600, 700]:
        f = lambda P: Ga(T_, P)[0] - Gg(T_, P)[0]
        P = np.nan
        try:
            if f(-0.5) * f(8.0) < 0:
                P = brentq(f, -0.5, 8.0, xtol=1e-7)
        except Exception:
            pass
        _, Va = Ga(T_, P if np.isfinite(P) else 0.0)
        _, Vg = Gg(T_, P if np.isfinite(P) else 0.0)
        dg0 = Ga(T_, 0.0)[0] - Gg(T_, 0.0)[0]
        rows.append(dict(T=T_, Pt=None if not np.isfinite(P) else P, Va=Va, Vg=Vg, dG0=dg0))
        print('%6d | %13s | %11.4f %11.4f | %+10.5f'
              % (T_, '%.3f' % P if np.isfinite(P) else '无解', Va, Vg, dg0))

    print('\n=== 常压 (P=0) 下的相稳定性 ===')
    Ts = np.arange(0, 800, 5)
    dg = [Ga(t, 0.0)[0] - Gg(t, 0.0)[0] for t in Ts]
    sgn = np.sign(dg)
    cross = [i for i in range(len(Ts) - 1) if sgn[i] * sgn[i + 1] < 0]
    if cross:
        Tt = brentq(lambda t: Ga(t, 0.0)[0] - Gg(t, 0.0)[0], Ts[cross[0]], Ts[cross[0] + 1], xtol=1e-4)
        print('  γ→α 转变温度 T_t = %.1f K   (实验 393 K)' % Tt)
    else:
        print('  0–800 K 无交点;  ΔG(0 K)=%+.5f, ΔG(800 K)=%+.5f eV/at'
              % (dg[0], dg[-1]))
        print('  ΔG<0 表示 α 更稳定' if dg[0] < 0 else '  ΔG>0 表示 γ 更稳定')

    print('\n=== 零点能 ===')
    for ph in ('alpha', 'gamma'):
        V, Fv = dat[ph]['V'], dat[ph]['Fv']
        i0 = int(np.argmin(np.abs(V - np.array([18.04, 17.14])[ph == 'gamma'])))
        print('  %-6s ZPE(V0) = %.3f meV/atom  (V=%.3f Å³/at)' % (ph, Fv[i0, 0] * 1000, V[i0]))
    ia = int(np.argmin(np.abs(dat['alpha']['V'] - 18.04)))
    ig = int(np.argmin(np.abs(dat['gamma']['V'] - 17.14)))
    print('  ZPE 差 (γ−α) = %+.3f meV/atom'
          % ((dat['gamma']['Fv'][ig, 0] - dat['alpha']['Fv'][ia, 0]) * 1000))

    json.dump(dict(rows=rows, P_eq_0K=float(P0), dE_static=float((E0g - E0a) * 1000),
                   dE_DFT=float(DE_DFT), e_from=args.e_from),
              open(ROOT + 'diag_precision/json/pt_line_final.json', 'w'), indent=1)
    print('\n写出 diag_precision/json/pt_line_final.json')
