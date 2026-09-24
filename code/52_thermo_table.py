#!/usr/bin/env python3
"""α/γ-ZrW2O8 的**完整热力学量表**（V6 + QHA，3×3×3 超胞，两相口径一致）。

数据：<root>/vol_*/{POSCAR, thermal_properties.yaml}
  α: qha_f32_out/qha_f32      7 点（vol_0975 有虚频 -> 剔除）
  γ: qha_f32_out/qha_gamma_333 7 点（3×3×3）
E(V)：archive 的固定胞弛豫能（float64）

输出（每相、每温度）：
  V(T)、α_V(T)、α_L(T)、B_T(T)、B_S(T)、S(T)、C_V(T)、C_p(T)、γ_th(T)、F(T)、G(T,P=0)
以及两相的差 ΔG(T)、ΔS(T)、ΔC_p(T)、以及常压相稳定性。

方法：F(V,T) = E_static(V) + F_vib(V,T)（样条，不假设 EOS 形式）
      V(T,P) 由 ∂F/∂V + P = 0 求得；所有导数用样条解析导数。
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, yaml
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

ROOT = os.environ.get('ROOT', '/home/jss/share/AlZrW2O8_MACE/')
NAT = {'alpha': 44, 'gamma': 132}
SRC = {'alpha': os.environ.get('ALPHA_DIR', ROOT + 'qha_f32_out/qha_f32/'),
       'gamma': os.environ.get('GAMMA_DIR', ROOT + 'qha_f32_out/qha_gamma_333/')}
SKIP = {'alpha': {'vol_0975'}, 'gamma': set()}
GPA = 160.21766208
RV = 8.314462618          # J/mol/K
E_SRC = {'alpha': [(os.environ.get('E_ALPHA', ROOT + 'qha/relax.json'), None)],
         'gamma': [(os.environ.get('E_GAMMA', ROOT + 'qha_gamma/relax.json'), None)]}


def load_e(ph):
    E = {}
    for path, inline in E_SRC[ph]:
        if path and os.path.exists(path):
            for r in json.load(open(path)):
                E[r['dir']] = r['E']
        if inline:
            E.update(inline)
    return E


def load(ph):
    """返回 V[Å³/cell]、E[eV/cell]、F[it, iV][eV/cell]、S[J/mol/K/cell]、Cv[J/mol/K/cell]、T"""
    d = SRC[ph]; n = NAT[ph]
    vols = json.load(open(d + 'volumes.json'))
    Est = load_e(ph)
    V, E, F, S, CV, T = [], [], [], [], [], None
    for v in sorted(vols, key=lambda x: x['V']):
        if v['dir'] in SKIP[ph] or v['dir'] not in Est:
            continue
        p = d + v['dir'] + '/thermal_properties.yaml'
        if not os.path.exists(p):
            print('  缺 %s' % p); continue
        y = yaml.safe_load(open(p)); tp = y['thermal_properties']
        T = np.array([e['temperature'] for e in tp])
        F.append(np.array([e['free_energy'] for e in tp]) / 96.485)       # eV/cell
        S.append(np.array([e['entropy'] for e in tp]))
        CV.append(np.array([e['heat_capacity'] for e in tp]))
        V.append(v['V']); E.append(Est[v['dir']])
    return np.array(V), np.array(E), np.array(F), np.array(S), np.array(CV), T


out = {}
for ph in ('alpha', 'gamma'):
    V, E, F, S, CV, T = load(ph)
    n = NAT[ph]
    print('\n[%s] %d 个体积点  V=%.1f..%.1f Å³  (%d 个温度点)'
          % (ph, len(V), V.min(), V.max(), len(T)), flush=True)
    rows = []
    for it, t in enumerate(T):
        Ft = E + F[:, it]
        s = CubicSpline(V, Ft)
        Vg = np.linspace(V.min(), V.max(), 4001)
        # P=0 的平衡体积：dV 的极小（或样条导数零点）
        i0 = int(np.argmin(s(Vg)))
        V0 = Vg[i0]
        B_T = V0 * s(V0, 2) * GPA                      # 等温体弹模量
        G = float(s(V0))                               # = F + PV，P=0 时 G=F
        # 热力学量：S、Cv 用体积插值（在 V0 处）
        Sv = float(np.interp(V0, V, S[:, it]))
        Cv = float(np.interp(V0, V, CV[:, it]))
        rows.append(dict(T=float(t), V=float(V0), G=G, B_T=float(B_T), S=Sv, Cv=Cv))
    rows = rows  # 保留全部温度点
    Tarr = np.array([r['T'] for r in rows])
    Varr = np.array([r['V'] for r in rows])
    Barr = np.array([r['B_T'] for r in rows])
    Sarr = np.array([r['S'] for r in rows])
    Cvarr = np.array([r['Cv'] for r in rows])
    dVdT = np.gradient(Varr, Tarr)
    alpha_V = dVdT / Varr
    alpha_L = alpha_V / 3
    # γ_th = alpha_V * B_T * V / Cv   （Grüneisen，用每胞量：α_V[1/K] * B[Pa] * V[m³] / Cv[J/K]）
    # 单位统一到 SI（每胞）：
    #   thermal_properties.yaml 的 heat_capacity / entropy 单位是 J/K/mol（mol = 1 个胞）
    #   所以要变成 J/K/cell 必须再除以 96.485（kJ/mol -> eV 的换算），
    #   即 Cv[eV/K/cell] = Cv[J/K/mol] / 96.485 / 1000 * 1000 ...
    # 直接按定义做：raw 就是 J/K per mole of cells -> J/K per cell 需除以 N_A 无关，
    # 因为 phonopy 的 "mol" 指 1 mol 胞，1 mol 胞的热容 = 1 胞的热容 × N_A。
    # 取物理量纲一致：Cv_cell = Cv_raw / N_A  (J/K per cell)
    NA = 6.02214076e23
    Cv_cell = Cvarr / NA                              # J/K per cell
    gamma_th = alpha_V * (Barr * 1e9) * (Varr * 1e-30) / Cv_cell
    # Cp（每胞）= Cv + α_V² B V T
    Cp_cell = Cv_cell + alpha_V ** 2 * (Barr * 1e9) * (Varr * 1e-30) * Tarr
    Cp = Cp_cell * NA / 96.485                        # 回到 kJ/mol/cell（与 yaml 同量纲）
    out[ph] = dict(natom=n, T=Tarr.tolist(), V=Varr.tolist(), G=[r['G'] for r in rows],
                   B_T=Barr.tolist(), alpha_V=alpha_V.tolist(), alpha_L=alpha_L.tolist(),
                   S=Sarr.tolist(), Cv=Cvarr.tolist(), Cp=Cp.tolist(), gamma_th=gamma_th.tolist())
    print('%-6s %9s %9s %9s %9s %9s %9s %9s' %
          ('T(K)', 'V(Å³)', 'B_T(GPa)', 'α_L(ppm/K)', 'S(J/K/mol)', 'Cv(J/K/mol)', 'Cp(J/K/mol)', 'γ_th'))
    for t in (0, 100, 200, 300, 400, 500, 600, 700, 800):
        i = int(np.argmin(abs(Tarr - t)))
        print('%-6d %9.3f %9.2f %9.2f %11.2f %11.2f %11.2f %9.3f'
              % (Tarr[i], Varr[i], Barr[i], alpha_L[i] * 1e6, Sarr[i] / n * 1000,
                 Cvarr[i] / n * 1000 / 96.485 * 1000 / 1000, Cp[i] / n * 1000 / 96.485,
                 gamma_th[i]))

# 两相差
Ta, Tg = np.array(out['alpha']['T']), np.array(out['gamma']['T'])
common = np.intersect1d(Ta, Tg)
print('\n=== 两相热力学差（α − γ，每原子）===')
print('%-6s %14s %14s %14s' % ('T(K)', 'ΔG(eV/at)', 'ΔS(J/K/mol/at)', 'ΔCp(J/K/mol/at)'))
dG, dS, dCp = [], [], []
for t in common:
    ia = int(np.argmin(abs(Ta - t))); ig = int(np.argmin(abs(Tg - t)))
    g = (out['alpha']['G'][ia] / 44 - out['gamma']['G'][ig] / 132)
    s = (out['alpha']['S'][ia] / 44 - out['gamma']['S'][ig] / 132)
    c = (out['alpha']['Cp'][ia] / 44 - out['gamma']['Cp'][ig] / 132) / 96.485 * 1000
    dG.append(float(g)); dS.append(float(s)); dCp.append(float(c))
    if t % 100 == 0 or t in (0,):
        print('%-6.0f %14.6f %14.4f %14.4f' % (t, g, s, c))
out['delta'] = dict(T=common.tolist(), dG=dG, dS=dS, dCp=dCp)
OUTF = os.environ.get('OUTF', ROOT + 'diag_precision/json/thermo_table.json')
json.dump(out, open(OUTF, 'w'), indent=1)
print('\n-> %s' % OUTF)
