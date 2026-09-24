#!/usr/bin/env python3
"""α/γ-ZrW2O8 相变线：G_α(T,P) = G_γ(T,P)，含零点能 —— 只用原始数据点，不假设 EOS 形式。

数据（力的部分为集群 MACE V6 **float32**，20 个体积点）：
  α 44 原子 : qha_f32/vol_*        (7 点，vol_0975 有虚频 → 剔除)
  γ 132 原子: qha_gamma_f32/vol_*  (10 点，vol_0955 有虚频 → 剔除)
  F_vib(V,T) 来自各点的 thermal_properties.yaml（phonopy，mesh α 12³ / γ 8³）
  E_static(V) 来自各点的固定胞弛豫（见 E_SRC）

方法：
  F(V,T) = E_static(V)/N + F_vib(V,T)
  G(T,P) = min_V [ F(V,T) + P·V/N ]        ← 在各自的体积网格上直接取极小，不拟合 EOS
  （先前用 Vinet 拟合宽网格会退化：实测 B0=2.4 GPa / Bp=13.6，得到假的 P_t≈0）
  P_t(T) 由 G_α(T,P) = G_γ(T,P) 二分求得；另给 P=0 下 γ→α 转变温度。

参考态对齐：V6 自己给出的静态能差 ΔE(γ−α) 与 DFT/PBE 的 +7.43 meV/atom 不同，
两条都算，便于区分"静态能差"与"振动贡献"各自的作用。
"""
import os, json, glob, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, yaml
from scipy.optimize import brentq

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
NAT = {'alpha': 44, 'gamma': 132}
DIRS = {'alpha': ROOT + 'qha_f32_out/qha_f32', 'gamma': ROOT + 'qha_f32_out/qha_gamma_f32'}
SKIP = {'alpha': {'vol_0975'}, 'gamma': {'vol_0955'}}
GPA = 160.21766208
E_DFT = {'alpha': -78808.8861118710, 'gamma': -236425.6776365414}
DE_DFT = (E_DFT['gamma'] / NAT['gamma'] - E_DFT['alpha'] / NAT['alpha']) * 1000.0

# 静态能来源（float64 归档弛豫能）
E_SRC = {
    'alpha': [(ROOT + 'qha/relax.json', None)],
    'gamma': [(ROOT + 'qha_gamma/relax_v6_7point_backup.json', None),
              (None, {'vol_0955': -236425.456686, 'vol_0965': -236425.855603,
                      'vol_0975': -236426.150539})],
}


def load_static_E(phase):
    E = {}
    for path, inline in E_SRC[phase]:
        if path and os.path.exists(path):
            for r in json.load(open(path)):
                E[r['dir']] = r['E']
        if inline:
            E.update(inline)
    return E


def load_phase(phase):
    vols = json.load(open(DIRS[phase] + '/volumes.json'))
    Est = load_static_E(phase)
    V, E, Fv, T = [], [], [], None
    for v in sorted(vols, key=lambda x: x['V']):
        if v['dir'] in SKIP[phase]:
            continue
        d = DIRS[phase] + '/' + v['dir']
        y = yaml.safe_load(open(d + '/thermal_properties.yaml'))
        tp = y['thermal_properties']
        T = np.array([e['temperature'] for e in tp])
        Ev = Est.get(v['dir'])
        if Ev is None:
            print('  !! %s 缺静态能，跳过' % v['dir']); continue
        V.append(v['V'] / NAT[phase])
        E.append(Ev / NAT[phase])
        Fv.append(np.array([e['free_energy'] for e in tp]) / 96.485 / NAT[phase])
    return np.array(V), np.array(E), np.array(Fv), T


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shift-dft', action='store_true',
                    help='把 γ 的静态能整体平移，使静态能差等于 DFT/PBE 的 +7.43 meV/atom')
    a = ap.parse_args()

    dat = {}
    for ph in ('alpha', 'gamma'):
        V, E, Fv, T = load_phase(ph)
        dat[ph] = dict(V=V, E=E, Fv=Fv, T=T)
        print('[%s] %d 个稳定体积点  V=%.4f..%.4f Å³/at  E=%.6f..%.6f eV/at'
              % (ph, len(V), V.min(), V.max(), E.min(), E.max()))

    ia = int(np.argmin(dat['alpha']['E'])); ig = int(np.argmin(dat['gamma']['E']))
    dE_raw = (dat['gamma']['E'][ig] - dat['alpha']['E'][ia]) * 1000
    print('\n实测最低点: α V=%.4f E=%.6f | γ V=%.4f E=%.6f'
          % (dat['alpha']['V'][ia], dat['alpha']['E'][ia],
             dat['gamma']['V'][ig], dat['gamma']['E'][ig]))
    print('  ΔE(γ−α) 原始 = %+.3f meV/atom   （DFT/PBE 参考 %+.3f）' % (dE_raw, DE_DFT))

    shift = 0.0
    if a.shift_dft:
        shift = (DE_DFT - dE_raw) / 1000.0
        print('  已把 γ 静态能平移 %+.3f meV/atom 对齐到 DFT/PBE' % (shift * 1000))
    dat['gamma']['E'] = dat['gamma']['E'] + shift

    # 0 K 静态体积模量（样条，仅作自检）
    from scipy.interpolate import CubicSpline
    for ph in ('alpha', 'gamma'):
        s = CubicSpline(dat[ph]['V'], dat[ph]['E'])
        Vm = dat[ph]['V'][np.argmin(dat[ph]['E'])]
        print('  [%s] 静态样条: V0=%.4f Å³/at  B(V0)=%.1f GPa' % (ph, Vm, Vm * s(Vm, 2) * GPA))

    def make_G(ph):
        V, E, Fv, T = dat[ph]['V'], dat[ph]['E'], dat[ph]['Fv'], dat[ph]['T']
        Vg = np.linspace(V.min(), V.max(), 6001)
        Eg = np.interp(Vg, V, E)
        cache = {}
        def G(T_, P_):
            it = int(np.argmin(abs(T - T_)))
            key = it
            if key not in cache:
                cache[key] = np.interp(Vg, V, Fv[:, it])
            F = Eg + cache[key] + P_ / GPA * Vg
            i = int(F.argmin())
            return float(F[i]), float(Vg[i])
        return G

    Ga, Gg = make_G('alpha'), make_G('gamma')
    print('\n=== 相变线 G_α(T,P)=G_γ(T,P) ===')
    print('%6s | %13s | %10s %10s | %12s' % ('T(K)', 'P_t (GPa)', 'V_α(Å³/at)', 'V_γ(Å³/at)', 'ΔG(P=0) eV/at'))
    rows = []
    for T_ in [0, 100, 200, 250, 300, 350, 393, 400, 500, 600, 700]:
        f = lambda P: Ga(T_, P)[0] - Gg(T_, P)[0]
        P = np.nan
        if f(-0.5) * f(10.0) < 0:
            P = brentq(f, -0.5, 10.0, xtol=1e-8)
        _, Va = Ga(T_, P if np.isfinite(P) else 0.0)
        _, Vg = Gg(T_, P if np.isfinite(P) else 0.0)
        dg0 = f(0.0)
        rows.append(dict(T=T_, Pt=None if not np.isfinite(P) else P, Va=Va, Vg=Vg, dG0=dg0))
        print('%6d | %13s | %10.4f %10.4f | %+12.5f'
              % (T_, '%.3f' % P if np.isfinite(P) else '无解', Va, Vg, dg0))

    print('\n=== 常压 (P=0) 相稳定性 ===')
    Ts = np.arange(0, 800, 5)
    dg = [Ga(t, 0.0)[0] - Gg(t, 0.0)[0] for t in Ts]
    sg = np.sign(dg)
    cr = [i for i in range(len(Ts) - 1) if sg[i] * sg[i + 1] < 0]
    Tt = None
    if cr:
        Tt = brentq(lambda t: Ga(t, 0.0)[0] - Gg(t, 0.0)[0], Ts[cr[0]], Ts[cr[0] + 1], xtol=1e-4)
        print('  γ→α 转变温度 T_t = %.1f K   (实验 393 K)' % Tt)
    else:
        which = 'α' if dg[0] < 0 else 'γ'
        print('  0–800 K 无交点：%s 在整个温区更稳定  (ΔG=%+.5f → %+.5f eV/at)'
              % (which, dg[0], dg[-1]))

    print('\n=== 零点能 ===')
    for ph, v0 in (('alpha', 18.04), ('gamma', 17.14)):
        i0 = int(np.argmin(np.abs(dat[ph]['V'] - v0)))
        print('  %-6s ZPE(V0)=%.3f meV/atom  (V=%.4f Å³/at)' % (ph, dat[ph]['Fv'][i0, 0] * 1000, dat[ph]['V'][i0]))
    print('  ZPE 差 (γ−α) = %+.3f meV/atom'
          % ((dat['gamma']['Fv'][ig, 0] - dat['alpha']['Fv'][ia, 0]) * 1000))

    tag = 'dft_aligned' if a.shift_dft else 'v6_raw'
    json.dump(dict(rows=rows, Tt=Tt, dE_raw=dE_raw, dE_DFT=DE_DFT, shift=shift,
                   alpha_V=dat['alpha']['V'].tolist(), alpha_E=dat['alpha']['E'].tolist(),
                   gamma_V=dat['gamma']['V'].tolist(), gamma_E=dat['gamma']['E'].tolist()),
              open(ROOT + 'diag_precision/json/pt_line_%s.json' % tag, 'w'), indent=1)
    print('\n写出 diag_precision/json/pt_line_%s.json' % tag)
