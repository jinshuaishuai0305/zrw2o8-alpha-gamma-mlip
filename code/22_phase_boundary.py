#!/usr/bin/env python3
"""α/γ-ZrW2O8 相变线：G_α(T,P) = G_γ(T,P)（V6 势，QHA，含零点能）。

两相数据都是 V6 + float64：
  α: qha/vol_*/                      (6 个动力学稳定点)
  γ: qha_gamma_v6f64/vol_*/          (13 个体积点)

F(V,T) = E_static(V) + F_vib(V,T)      （F_vib 含 ZPE，phonopy 给出）
G(T,P) = min_V [ F(V,T) + P·V ]        （Vinet 光滑化 E(V)）
P_t(T): 二分求 G_α(T,P) = G_γ(T,P)
另给 P=0 下的 γ→α 转变温度 T_t。

两套结果：
  (A) 直接用 V6 的 E_static
  (B) 把 γ 的静态能平移到 DFT/PBE 的相对值（用于"只信 PBE"的对照）
"""
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np, yaml
from scipy.optimize import curve_fit, brentq

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
NAT = {'alpha': 44, 'gamma': 132}
SRC = {'alpha': ROOT + 'qha/', 'gamma': ROOT + 'qha_gamma_v6f64/'}
E_DFT = {'alpha': -78808.8861118710, 'gamma': -236425.6776365414}
DE_DFT = (E_DFT['gamma'] / NAT['gamma'] - E_DFT['alpha'] / NAT['alpha']) * 1000.0
GPA = 160.21766208


def vinet(V, E0, B0, Bp, V0):
    x = (V / V0) ** (1 / 3.)
    return E0 + 2 * B0 * V0 / (Bp - 1) ** 2 * (2 - (5 + 3 * Bp * (x - 1) - 3 * x) * np.exp(-1.5 * (Bp - 1) * (x - 1)))


def load(phase):
    root, n = SRC[phase], NAT[phase]
    rec = json.load(open(root + 'relax.json'))
    summ = json.load(open(root + 'phonon_summary.json'))
    ok = {s['dir'] for s in summ if s['n_imag'] == 0}       # 只保留动力学稳定的点
    rec = sorted([r for r in rec if r['dir'] in ok], key=lambda x: x['V'])
    Vs, Es, Fs, T = [], [], [], None
    for r in rec:
        y = yaml.safe_load(open(root + r['dir'] + '/thermal_properties.yaml'))
        tp = y['thermal_properties']
        T = np.array([e['temperature'] for e in tp])
        # free_energy 单位 kJ/mol（每胞），换算成 eV/atom
        Fs.append(np.array([e['free_energy'] for e in tp]) / 96.485 / n)
        Vs.append(r['V']); Es.append(r['E'] / n)
    return np.array(Vs), np.array(Es), np.array(Fs), T, [r['dir'] for r in rec]


def G_of_T(Vs, Es, Fs, Tgrid, T_, P_, shift):
    """返回 G(T,P) [eV/atom] 和 V(T,P) [Å³/cell]。"""
    it = int(np.argmin(abs(Tgrid - T_)))
    Ftot = Es + Fs[:, it] + shift
    Vat = Vs / (Vs.sum() / Vs.sum())        # 占位
    p, _ = curve_fit(vinet, Vs, Ftot * np.array([1.0] * len(Vs)) * 1,
                     p0=[Ftot.min(), 0.5, 4, Vs[Ftot.argmin()]], maxfev=400000)
    Vg = np.linspace(Vs.min() * 0.85, Vs.max() * 1.15, 20000)
    Fv = vinet(Vg, *p) + P_ / GPA * Vg
    i = int(Fv.argmin())
    return float(Fv[i]), float(Vg[i])


if __name__ == '__main__':
    Va, Ea, Fa, T, da = load('alpha')
    Vg, Eg, Fg, Tg, dg = load('gamma')
    n_a, n_g = NAT['alpha'], NAT['gamma']
    print('α: %d 个稳定体积点 %s  V=%.1f..%.1f Å³' % (len(Va), da, Va.min(), Va.max()))
    print('γ: %d 个稳定体积点  V=%.1f..%.1f Å³' % (len(Vg), Vg.min(), Vg.max()))

    pa, _ = curve_fit(vinet, Va, Ea, p0=[Ea.min(), 0.5, 4, Va[Ea.argmin()]], maxfev=400000)
    pg, _ = curve_fit(vinet, Vg, Eg, p0=[Eg.min(), 0.5, 4, Vg[Eg.argmin()]], maxfev=400000)
    dE_v6 = (pg[0] - pa[0]) * 1000
    print('\n0 K 静态（每原子）: V0(α)=%.4f  V0(γ)=%.4f Å³/at   ΔV=%+.4f (%.2f%%)'
          % (pa[3] / n_a, pg[3] / n_g, pg[3] / n_g - pa[3] / n_a,
             100 * (pg[3] / n_g - pa[3] / n_a) / (pa[3] / n_a)))
    print('  dE(γ−α) V6  = %+.3f meV/atom' % dE_v6)
    print('  dE(γ−α) DFT = %+.3f meV/atom' % DE_DFT)
    i_a0 = int(np.argmin(abs(Va - 793.79))); i_g0 = int(np.argmin(abs(Vg - 2263.29)))
    print('  ZPE(α)=%.2f  ZPE(γ)=%.2f  ZPE 差=%+.2f meV/atom'
          % (Fa[i_a0, 0] * 1000, Fg[i_g0, 0] * 1000, (Fg[i_g0, 0] - Fa[i_a0, 0]) * 1000))

    shift_g_B = (DE_DFT - dE_v6) / 1000.0      # eV/atom
    print('\n对齐 B: γ 静态能平移 %+.3f meV/atom' % (shift_g_B * 1000))

    print('\n=== 相变线（G_α = G_γ）===')
    print('%6s | %13s %13s | %10s %10s' % ('T(K)', 'P_t(A) V6', 'P_t(B) DFT对齐', 'V_α(Å³)', 'V_γ(Å³)'))
    rows = []
    def solveP(T_, sg):
        f = lambda P: (G_of_T(Va, Ea, Fa, T, T_, P, 0.0)[0] - G_of_T(Vg, Eg, Fg, T, T_, P, sg)[0])
        try:
            if f(-1.0) * f(6.0) > 0:
                return np.nan
            return brentq(f, -1.0, 6.0, xtol=1e-5)
        except Exception:
            return np.nan
    for T_ in [0, 100, 200, 300, 400, 500, 600, 700]:
        PA = solveP(T_, 0.0); PB = solveP(T_, shift_g_B)
        _, Vv_a = G_of_T(Va, Ea, Fa, T, T_, PA if np.isfinite(PA) else 0.0, 0.0)
        _, Vv_g = G_of_T(Vg, Eg, Fg, T, T_, PA if np.isfinite(PA) else 0.0, 0.0)
        rows.append(dict(T=T_, Pt_A=None if not np.isfinite(PA) else PA,
                         Pt_B=None if not np.isfinite(PB) else PB, Va=Vv_a, Vg=Vv_g))
        print('%6d | %13s %13s | %10.2f %10.2f'
              % (T_, '%.3f GPa' % PA if np.isfinite(PA) else '  无解',
                 '%.3f GPa' % PB if np.isfinite(PB) else '  无解', Vv_a, Vv_g))

    # P=0 的转变温度
    print('\n=== 常压 (P=0) 下的 γ→α 转变温度 ===')
    for key, sg in (('A (V6 原样)', 0.0), ('B (DFT 对齐)', shift_g_B)):
        def g_(T_):
            return G_of_T(Va, Ea, Fa, T, T_, 0.0, 0.0)[0] - G_of_T(Vg, Eg, Fg, T, T_, 0.0, sg)[0]
        Ts = np.arange(0, 700, 25)
        gs = [g_(t) for t in Ts]
        sign = np.sign(gs)
        cross = [i for i in range(len(Ts) - 1) if sign[i] * sign[i + 1] < 0]
        if cross:
            i = cross[0]
            Tt = brentq(g_, Ts[i], Ts[i + 1], xtol=1e-3)
            print('  %s: T_t = %.1f K   (g<0 侧 = α 更稳定)' % (key, Tt))
        else:
            print('  %s: 0–700 K 内没有交点（g(0)=%+.4f, g(700)=%+.4f eV/atom）'
                  % (key, gs[0], gs[-1]))

    json.dump(dict(rows=rows, dE_v6=dE_v6, dE_DFT=DE_DFT, shift_g_B=shift_g_B),
              open(ROOT + 'diag_precision/json/pt_line_v6.json', 'w'), indent=1)
    print('\n写出 diag_precision/json/pt_line_v6.json')
