#!/usr/bin/env python3
"""阶段 05+06（本地 phonopy 环境）：从 FORCE_SETS 建 FC -> 热学量 -> QHA -> 相变线。

严格照 skill references/qha-workflow.md：
  · **契约①**：原胞必须从 phonopy_disp.yaml 载入，绝不用 POSCAR 重读
    （否则 0.03 Å 失配会造出假虚频，实测 −5.20 THz / 526 个虚频）
  · 只保留无虚频的体积点进入 QHA（β 相在压缩端不稳定是真实物理）
  · mesh：α 12×12×12、γ 8×8×8（与归档一致）

两相数据：qha_f32_out/{qha_f32,qha_gamma_f32}/vol_*/FORCE_SETS（集群 MACE V6 float32）
E(V)：qha_f32_out/static_E.json（由 E0 来源决定，见 --e-from）

用法: python 33_thermal_and_qha.py [--e-from float32|float64]
"""
import os, sys, json, glob, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, yaml, phonopy

ROOT = '/home/jss/share/AlZrW2O8_MACE/'
OUT = ROOT + 'qha_f32_out/'
MESH = {'alpha': [12, 12, 12], 'gamma': [8, 8, 8]}
NAT = {'alpha': 44, 'gamma': 132}
DIRS = {'alpha': OUT + 'qha_f32', 'gamma': OUT + 'qha_gamma_f32'}
GPA = 160.21766208

ap = argparse.ArgumentParser()
ap.add_argument('--e-from', default='float64', choices=['float32', 'float64'])
args = ap.parse_args()


def load_phase(phase):
    root = DIRS[phase]
    vols = json.load(open(root + '/volumes.json'))
    recs = []
    for v in sorted(vols, key=lambda x: x['V']):
        d = os.path.join(root, v['dir'])
        if not os.path.exists(d + '/FORCE_SETS'):
            print('  跳过（无 FORCE_SETS）%s' % v['dir']); continue
        ph = phonopy.load(phonopy_yaml=d + '/phonopy_disp.yaml',       # 契约①
                          force_sets_filename=d + '/FORCE_SETS',
                          produce_fc=True, log_level=0)
        ph.run_mesh(MESH[phase], with_eigenvectors=False)
        fr = np.array(ph.get_mesh_dict()['frequencies'])
        n_imag = int((fr < -1e-3).sum())
        ph.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
        ph.write_yaml_thermal_properties(filename=d + '/thermal_properties.yaml')
        tp = ph.get_thermal_properties_dict()
        T = np.array(tp['temperatures'])
        F = np.array(tp['free_energy']) / 96.485 / NAT[phase]          # eV/atom
        # ΔV=0 的零点能（phonopy 的 free_energy(0 K) 即 ZPE）
        zpe = float(F[0])
        recs.append(dict(dir=v['dir'], V=v['V'], n_imag=n_imag, fmin=float(fr.min()),
                         zpe=zpe, T=T, F=F))
        print('  %-9s V=%8.3f  虚频 %4d  最低模 %+.3f THz  ZPE=%.2f meV/at'
              % (v['dir'], v['V'], n_imag, fr.min(), zpe * 1000), flush=True)
    return recs


def vinet(V, E0, B0, Bp, V0):
    x = (V / V0) ** (1 / 3.)
    return E0 + 2 * B0 * V0 / (Bp - 1) ** 2 * (2 - (5 + 3 * Bp * (x - 1) - 3 * x) * np.exp(-1.5 * (Bp - 1) * (x - 1)))


if __name__ == '__main__':
    from scipy.optimize import curve_fit, brentq
    print('=== E(V) 来源: %s ===' % args.e_from)
    data = {}
    for phase in ('alpha', 'gamma'):
        print('\n[%s] mesh %s' % (phase, MESH[phase]))
        recs = load_phase(phase)
        stable = [r for r in recs if r['n_imag'] == 0]
        print('  稳定体积点 %d / %d' % (len(stable), len(recs)))
        data[phase] = dict(all=recs, stable=stable)

    # 静态能 E(V)
    E_of_V = {'alpha': {}, 'gamma': {}}
    if args.e_from == 'float64':
        for r in json.load(open(ROOT + 'qha/relax.json')):
            E_of_V['alpha'][r['dir']] = r['E']
        for f in (ROOT + 'qha_gamma/relax_v6_7point_backup.json',
                  ROOT + 'qha_gamma_v6f64/relax.json'):
            if os.path.exists(f):
                for r in json.load(open(f)):
                    E_of_V['gamma'][r['dir']] = r['E']
    else:
        for ph, f in (('alpha', OUT + 'relax_alpha.json'), ('gamma', OUT + 'relax_gamma.json')):
            if os.path.exists(f):
                for r in json.load(open(f)):
                    E_of_V[ph][r['dir']] = r['E']
    print('\nE 来源(%s): α %d 点, γ %d 点'
          % (args.e_from, len(E_of_V['alpha']), len(E_of_V['gamma'])))

    # ---- 汇总每相的 V, E, F(V,T) ----
    pack = {}
    for phase in ('alpha', 'gamma'):
        st = [r for r in data[phase]['stable'] if r['dir'] in E_of_V[phase]]
        Vs = np.array([r['V'] for r in st])
        Es = np.array([E_of_V[phase][r['dir']] / NAT[phase] for r in st])
        Fs = np.array([r['F'] for r in st])
        T = st[0]['T']
        pack[phase] = dict(Vs=Vs, Es=Es, Fs=Fs, T=T,
                           dirs=[r['dir'] for r in st], zpe=np.array([r['zpe'] for r in st]))
        p, _ = curve_fit(vinet, Vs, Es, p0=[Es.min(), 0.5, 4, Vs[Es.argmin()]], maxfev=400000)
        print('\n[%s] 静态 Vinet: V0=%.4f Å³/at  E0=%.6f eV/at  B0=%.1f GPa  Bp=%.2f'
              % (phase, p[3] / NAT[phase], p[0], p[1] * GPA, p[2]))
        pack[phase]['static_bm'] = p

    pa, pg = pack['alpha']['static_bm'], pack['gamma']['static_bm']
    dE = (pg[0] - pa[0]) * 1000
    dV = pg[3] / NAT['gamma'] - pa[3] / NAT['alpha']
    print('\n=== 0 K 静态对照 ===')
    print('  V0(α)=%.4f  V0(γ)=%.4f Å³/at   ΔV=%+.4f (%.2f%%)' %
          (pa[3] / NAT['alpha'], pg[3] / NAT['gamma'], dV, 100 * dV / (pa[3] / NAT['alpha'])))
    print('  dE(γ−α) = %+.3f meV/atom   → 一阶 P_t = %+.3f GPa'
          % (dE, dE * 1e-3 / dV * GPA))

    # ---- G(T,P) 与相变线 ----
    # 每个温度只拟合一次 Vinet，并缓存（否则相变线求解要重拟合上千次）
    _fitcache = {}
    def fit_T(phase, T_):
        key = (phase, int(T_))
        if key not in _fitcache:
            pk = pack[phase]
            it = int(np.argmin(abs(pk['T'] - T_)))
            Ftot = pk['Es'] + pk['Fs'][:, it]
            p, _ = curve_fit(vinet, pk['Vs'], Ftot,
                             p0=[Ftot.min(), 0.5, 4, pk['Vs'][Ftot.argmin()]], maxfev=400000)
            Vg = np.linspace(pk['Vs'].min() * 0.80, pk['Vs'].max() * 1.20, 4000)
            _fitcache[key] = (p, Vg)
        return _fitcache[key]

    def G(phase, T_, P_):
        p, Vg = fit_T(phase, T_)
        Fv = vinet(Vg, *p) + P_ / GPA * Vg
        i = int(Fv.argmin())
        return float(Fv[i]), float(Vg[i])

    print('\n=== 相变线 P_t(T): G_α(T,P) = G_γ(T,P) ===')
    print('%6s | %14s | %10s %10s' % ('T(K)', 'P_t (GPa)', 'V_α(Å³)', 'V_γ(Å³)'))
    rows = []
    for T_ in [0, 100, 200, 250, 300, 350, 400, 500, 600, 700]:
        f = lambda P: G('alpha', T_, P)[0] - G('gamma', T_, P)[0]
        P = np.nan
        try:
            if f(-1.0) * f(8.0) < 0:
                P = brentq(f, -1.0, 8.0, xtol=1e-6)
        except Exception as e:
            print('   T=%d 求解失败 %s' % (T_, e))
        if np.isfinite(P):
            _, Va = G('alpha', T_, P); _, Vg = G('gamma', T_, P)
        else:
            _, Va = G('alpha', T_, 0.0); _, Vg = G('gamma', T_, 0.0)
        rows.append(dict(T=T_, Pt=None if not np.isfinite(P) else P, Va=Va, Vg=Vg))
        print('%6d | %14s | %10.2f %10.2f'
              % (T_, '%.3f' % P if np.isfinite(P) else '无解', Va, Vg))

    print('\n=== 常压(P=0) γ→α 转变温度 ===')
    Ts = np.arange(0, 800, 25)
    g0 = [G('alpha', t, 0.0)[0] - G('gamma', t, 0.0)[0] for t in Ts]
    s = np.sign(g0)
    cross = [i for i in range(len(Ts) - 1) if s[i] * s[i + 1] < 0]
    Tt = None
    if cross:
        i = cross[0]
        Tt = brentq(lambda t: G('alpha', t, 0.0)[0] - G('gamma', t, 0.0)[0], Ts[i], Ts[i + 1], xtol=1e-4)
        print('  T_t = %.1f K   (g(0)=%+.5f, g(800)=%+.5f eV/at)' % (Tt, g0[0], g0[-1]))
    else:
        print('  0–800 K 内无交点：g(0)=%+.5f, g(800)=%+.5f eV/at' % (g0[0], g0[-1]))

    print('\n=== ZPE ===')
    for ph in ('alpha', 'gamma'):
        pk = pack[ph]
        i0 = int(np.argmin(abs(pk['Vs'] / NAT[ph] - ([18.04, 17.14][ph == 'gamma']))))
        print('  %s: ZPE(V0)=%.2f meV/atom   (V=%.2f)' % (ph, pk['zpe'][i0] * 1000, pk['Vs'][i0]))
    za = pack['alpha']['zpe'].min(); zg = pack['gamma']['zpe'].min()
    print('  ZPE 差 (γ−α) = %+.2f meV/atom' % ((zg - za) * 1000))

    json.dump(dict(rows=rows, Tt=Tt, dE=dE, dV=dV,
                   alpha=dict(Vs=pack['alpha']['Vs'].tolist(), Es=pack['alpha']['Es'].tolist(),
                              zpe=pack['alpha']['zpe'].tolist(), dirs=pack['alpha']['dirs']),
                   gamma=dict(Vs=pack['gamma']['Vs'].tolist(), Es=pack['gamma']['Es'].tolist(),
                              zpe=pack['gamma']['zpe'].tolist(), dirs=pack['gamma']['dirs']),
                   e_from=args.e_from),
              open(ROOT + 'diag_precision/json/qha_f32_result.json', 'w'), indent=1)
    print('\n写出 diag_precision/json/qha_f32_result.json')
