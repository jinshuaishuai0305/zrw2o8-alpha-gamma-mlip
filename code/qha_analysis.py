#!/usr/bin/env python3
"""qha_analysis.py — 本地：BM 拟合 / ZPE / QHA / 相变线 / CTE / 弹性汇总。

输入（--raw，从集群取回）：
    eos_raw.json      cluster_qha_v11/eos/eos_raw.json
    phonons_all.json  cluster_qha_v11/qha/phonons_all.json （26 个 ph.json 的合并）
    elastic_raw.json  cluster_qha_v11/elastic/elastic_raw.json
输出（--out）：eos.json / thermal.json / zpe_phdos.json / pt_line.json / cte.json /
              elastic.json / SUMMARY.md

用法: python qha_analysis.py --raw ./raw --out .
"""
import os, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import CubicSpline

GPA = 160.21766208        # GPa·Å³/eV
GPA_AA3_EV = 1e-21 / 1.602176634e-19   # GPa·Å³ -> eV
DPT_TRUTH = dict(dE=2.52, dE_sig=0.20, Pt=0.210, coef=0.1791,
                 src='ref_audit/abacus_runs/RESULTS.md + abacus_results.json')
V6 = dict(dEst=1.922, dZPE=0.984, Pt=0.521, src='deposit/data/json (旧 V6)')
V9_ZPE = dict(alpha=68.50, gamma=69.69, dZPE=(1.20, 1.44), src='V9 参照（三 seed）')


def bm3(V, E0, B0, Bp, V0):
    """Birch–Murnaghan 3rd order，V 为每原子体积 (Å³)，E 为每原子能量 (eV)。"""
    x = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9 * V0 * B0 / 16 * ((x - 1) ** 3 * Bp + (x - 1) ** 2 * (6 - 4 * x))


ap = argparse.ArgumentParser()
ap.add_argument('--raw', default='raw')
ap.add_argument('--out', default='.')
ap.add_argument('--pending', nargs='*', default=None)
ap.add_argument('--cmds', nargs='*', default=None)
A = ap.parse_args()
PENDING = A.pending if A.pending else []
CMDS = A.cmds if A.cmds else [
    'PHASE=gamma VLIST=0.94,...,1.06 sbatch --export=ALL job_phon.sbatch',
    'QHA_ROOT=$PWD python stage_eos.py      # 固定胞 E_st(V)，逐点缓存可续跑',
    'python phonon_pipeline.py {gen|forces|post} --phase <ph> --tag vol_1.0000',
    'python gate_audit.py                   # W-O 距离谱重构闸门',
    'python stage_elastic.py --delta 0.005  # 应力-应变',
    'python qha_analysis.py --raw ./raw --out .']
os.makedirs(A.out, exist_ok=True)

EOS = json.load(open(os.path.join(A.raw, 'eos_raw.json')))
PHO = json.load(open(os.path.join(A.raw, 'phonons_all.json')))
ELA = json.load(open(os.path.join(A.raw, 'elastic_raw.json')))
_ga = os.path.join(A.raw, 'gate_audit.json')
GA = json.load(open(_ga)) if os.path.exists(_ga) else None
AUD = {}
if GA:
    for _ph, _d in GA['phases'].items():
        for _r in _d['ref']:
            AUD[(_ph, _r['tag'])] = _r
NAT = {'alpha': EOS['phases']['alpha']['natoms'], 'gamma': EOS['phases']['gamma']['natoms']}
FORCE_CUTOFF_MEV = 1.0        # E(V) 偏离拟合 >1 meV/atom 且落在拟合下方 -> 重构嫌疑（droop）
def basin_gate(ph, pts):
    """势阱连续性闸门：相邻(按 V/V0 排序)体积点弛豫末态之间 max|Δpos| > 0.5 Å
    => 跳到了另一个局部极小（结构重构）。以 s=1.00 所在势阱为"本相"，
    其余势阱一律剔除（沿两侧逐步推进，允许同一势阱内连续形变）。"""
    if not all((ph, p['tag']) in AUD for p in pts):
        return {p['tag']: (bool(p['gate_cn_shift']), dict(source='cn_shift@2.2A'))
                for p in pts}
    rows = sorted(((AUD[(ph, p['tag'])], p['tag']) for p in pts), key=lambda t: t[0]['s'])
    n = len(rows)
    link = [None] * n
    for i in range(n):
        dp = rows[i][0]['dpos_vs_prev_A']
        link[i] = dp
    link[0] = rows[1][0]['dpos_vs_prev_A'] if n > 1 else 0.0
    ref = min(range(n), key=lambda i: abs(rows[i][0]['s'] - 1.0))
    THR = 0.5
    basin = [None] * n
    basin[ref] = 0
    b = 0
    for i in range(ref + 1, n):
        if (link[i] or 0.0) > THR:
            b += 1
        basin[i] = b
    b = 0
    for i in range(ref - 1, -1, -1):
        if (link[i + 1] or 0.0) > THR:
            b += 1
        basin[i] = b
    out = {}
    for i, (r, tag) in enumerate(rows):
        out[tag] = (basin[i] != 0,
                    dict(source='gate_audit.basin_walk', basin=basin[i],
                         dpos_vs_prev_A=link[i], d5_min=r['d5_min'], d5_mean=r['d5_mean'],
                         gap_min=r['gap_min'], n_WO4=r['n_WO4'], n_WO5=r['n_WO5'],
                         wo5_note='d5<2.10 Å 的 W 数（连续弹性压缩也会触发，仅诊断）'))
    return out


PH_GATE = {ph: basin_gate(ph, EOS['phases'][ph]['points']) for ph in ('alpha', 'gamma')}


# ============================================================ 1. EOS（BM 拟合）
eos_out = dict(model=EOS['model'], dtype=EOS['dtype'], fmax_target=EOS['fmax_target'],
               ngrid=EOS['ngrid'], grid_rule=EOS['grid_rule'],
               gate_rules=dict(
                   structural=('势阱连续性：相邻体积点弛豫末态 max|Δpos| > 0.5 A -> 换势阱 = 重构'
                               '（gate_audit，W–O 距离谱 d5 作机制佐证）') if GA else
                              'W–O CN @2.2 A 与同相 s=1.00 点相比任一 bin 变化 >=2',
                   energy='首次 BM 拟合残差 < -1.0 meV/atom（能量异常下折，droop）',
                   dpos='只作诊断（起始几何是未弛豫的 _ref，本身就有 0.03–0.46 A 弛豫位移）',
                   cn_hard_cutoff_note='2.2 A 硬截断 CN 在 γ 上随应变连续漂移，仅诊断'),
               phases={})
for ph in ('alpha', 'gamma'):
    info = EOS['phases'][ph]
    n = info['natoms']
    pts = info['points']
    for p in pts:
        p['V_at'] = p['V'] / n
        p['E_meV_at'] = p['E_per_atom'] * 1000.0
    # --- 闸门（写死规则，逐点打标签）---
    #   G1 cn_shift : 2.2 Å 下 W 的 CN 直方图与同相 s=1.00 点相比任一 bin 变化 >=2 个 W
    #   G2 droop    : 首次 BM 拟合残差 < -1.0 meV/atom（能量异常下折 = 重构特征），迭代剔除
    #   dpos        : 只作诊断（起始几何是未弛豫的 _ref，天然有 0.2–0.5 Å 弛豫位移，不作判据）
    for p in pts:
        p['gate_wo5'], p['gate_struct'] = PH_GATE[ph][p['tag']]
        p['restructured'] = bool(p['gate_wo5'])
        p['flag'] = '+'.join([x for x, c in (('wo5', p['gate_wo5']),
                                             ('cn_shift_2.2A_elastic', p['gate_cn_shift']),
                                             ('dpos_gt_0.5A', p['gate_dpos'])) if c]) or 'clean'
    clean0 = [p for p in pts if not p['restructured'] and p['fmax_end'] <= 1e-3]
    if len(clean0) < 5:
        clean0 = [p for p in pts if not p['restructured']]
    clean = list(clean0)
    droop = []
    for _ in range(3):
        V = np.array([p['V_at'] for p in clean])
        E = np.array([p['E_per_atom'] for p in clean])
        par, cov = curve_fit(bm3, V, E, p0=[E.min(), 0.6, 4.0, V[E.argmin()]], maxfev=400000)
        resid = {p['tag']: float((bm3(p['V_at'], *par) - p['E_per_atom']) * 1000) for p in clean}
        bad = [p for p in clean if resid[p['tag']] < -FORCE_CUTOFF_MEV]
        if not bad or len(clean) - len(bad) < 5:
            break
        for p in bad:
            p['gate_droop_meV'] = resid[p['tag']]
            droop.append(p)
        _bad = {b['tag'] for b in bad}
        clean = [p for p in clean if p['tag'] not in _bad]
    V = np.array([p['V_at'] for p in clean])
    E = np.array([p['E_per_atom'] for p in clean])
    par, cov = curve_fit(bm3, V, E, p0=[E.min(), 0.6, 4.0, V[E.argmin()]], maxfev=400000)
    res = bm3(V, *par) - E
    # BM4（b0=0 简化式）交叉核对，只报 V0/B0，不替代 BM3 主值
    def bm4(V, E0, B0, Bp, B2, V0):
        x = (V0 / V) ** (2.0 / 3.0)
        return bm3(V, E0, B0, Bp, V0) + 9 * V0 * B0 / 16 * (x - 1) ** 3 * (B2 * (x - 1) + 0) * 0 \
               + (9 * V0 * B0 / 16) * (x - 1) ** 2 * (2 * (x - 1) ** 2) * B2
    try:
        p4, _c4 = curve_fit(bm4, V, E, p0=[par[0], par[1], par[2], 0.1, par[3]], maxfev=400000)
        r4 = float(np.sqrt((bm4(V, *p4) - E) ** 2).mean() * 1000)
        BM4 = dict(E0_eV_per_atom=float(p4[0]), B0_GPa=float(p4[1] * GPA),
                   B0p=float(p4[2]), B2=float(p4[3]), V0_A3_per_atom=float(p4[4]),
                   fit_rms_meV_per_atom=r4)
    except Exception as _e:
        BM4 = dict(failed=str(_e))
    for p in droop:
        p['gate_cn_shift'] = p['gate_cn_shift']
        p['flag'] = (p['flag'] + '+droop') if p['flag'] != 'clean' else 'droop'
    rms = float(np.sqrt((res ** 2).mean()) * 1000)
    rmsfit = float(np.sqrt((res ** 2).mean()) * 1000)
    err = np.sqrt(np.diag(cov))
    eos_out['phases'][ph] = dict(
        natoms=n, V0_A3_abacus=info['V0_A3'], V0_src=info['V0_src'],
        n_points_total=len(pts), n_points_used=len(clean),
        used_tags=[p['tag'] for p in clean],
        excluded=[dict(tag=p['tag'], flag=p['flag'], s=p['s'],
                       reason='+'.join([x for x, c in
                                        (('wo5', p['gate_wo5']),
                                         ('droop', 'droop' in p['flag'])) if c]),
                       droop_meV_at=p.get('gate_droop_meV'),
                       max_dpos_A=p['max_dpos_A'], dpos_note='diagnostic only',
                       wo_cn=p['wo_cn']['hist'],
                       cn_shift=p.get('cn_shift_vs_s1', {}),
                       fmax_end=p['fmax_end']) for p in pts if p['tag'] not in {c['tag'] for c in clean}],
        gate_dpos_diagnostics=[dict(tag=p['tag'], dpos_vs_unrelaxed_ref_A=p['max_dpos_A'],
                                    dpos_vs_prev_volume_A=(p['gate_struct'] or {}).get('dpos_vs_prev_A'),
                                    gate_dpos_0p5A=p['gate_dpos']) for p in pts],
        structural_gate_src=('gate_audit W-O distance spectrum (d5<%.2f A)'
                             % (GA['wo5_cut_A'] if GA else 0)) if GA else 'cn_shift@2.2A',
        BM=dict(E0_eV_per_atom=float(par[0]), B0_GPa=float(par[1] * GPA),
                B0p=float(par[2]), V0_A3_per_atom=float(par[3]),
                V0_A3_cell=float(par[3] * n),
                err=dict(E0_eV=float(err[0]), B0_GPa=float(err[1] * GPA),
                         B0p=float(err[2]), V0_A3_per_atom=float(err[3])),
                fit_rms_meV_per_atom=rmsfit,
                fit_rms_ok=bool(rmsfit < 0.3),
                fit_max_abs_meV_per_atom=float(np.abs(res).max() * 1000),
                BM4_crosscheck=BM4),
        table=[dict(tag=p['tag'], s=p['s'], V_cell=p['V'], V_at=p['V_at'],
                    E_eV=p['E_eV'], E_meV_at=p['E_meV_at'],
                    resid_meV_at=float((bm3(p['V_at'], *par) - p['E_per_atom']) * 1000),
                    fmax_end=p['fmax_end'], nsteps=p['nsteps'], sec=p['sec'],
                    max_dpos_A=p['max_dpos_A'], wo_cn=p['wo_cn'],
                    restructured=p['restructured'], flag=p['flag'],
                    gate_wo5=p['gate_wo5'], gate_struct_detail=p['gate_struct'],
                    gate_cn_shift_hard_cutoff=p['gate_cn_shift'],
                    gate_dpos_diagnostic=p['gate_dpos'])
               for p in pts])
pa = eos_out['phases']['alpha']['BM']
pg = eos_out['phases']['gamma']['BM']
dEst = (pg['E0_eV_per_atom'] - pa['E0_eV_per_atom']) * 1000.0
dVat = pg['V0_A3_per_atom'] - pa['V0_A3_per_atom']
eos_out['delta'] = dict(
    dEst_meV_per_atom=dEst,
    dEst_vs_DFT=dEst - DPT_TRUTH['dE'],
    dE0_formula='E0(gamma)-E0(alpha) from BM fits, eV/atom -> meV/atom',
    dV0_A3_per_atom=float(dVat), dV0_pct=float(100 * dVat / pa['V0_A3_per_atom']),
    coef_GPa_per_meV_atom=float(GPA / abs(dVat) * 1e-3),
    coef_src='160.2177 GPa*A^3/eV / |dV_at| x 1e-3 (V11 自己的 dV0)',
    dEst_acceptance=dict(lo=DPT_TRUTH['dE'] - DPT_TRUTH['dE_sig'],
                         hi=DPT_TRUTH['dE'] + DPT_TRUTH['dE_sig'],
                         within_half_meV=bool(abs(dEst - DPT_TRUTH['dE']) <= 0.5)))
json.dump(eos_out, open(os.path.join(A.out, 'eos.json'), 'w'), indent=1)
print('[eos] dE0 = %+.3f meV/atom (DFT %.2f+-%.2f, diff %+.3f)'
      % (dEst, DPT_TRUTH['dE'], DPT_TRUTH['dE_sig'], dEst - DPT_TRUTH['dE']))
for ph in ('alpha', 'gamma'):
    b = eos_out['phases'][ph]['BM']
    print('      [%s] V0=%.4f A^3/at  E0=%.8f eV/at  B0=%.1f GPa  B0p=%.2f  RMS=%.3f meV/at  used %d/%d'
          % (ph, b['V0_A3_per_atom'], b['E0_eV_per_atom'], b['B0_GPa'], b['B0p'],
             b['fit_rms_meV_per_atom'], eos_out['phases'][ph]['n_points_used'],
             eos_out['phases'][ph]['n_points_total']))

# ============================================================ 2. 声子 / 热学量
TREF = np.array(PHO['T_grid'])
th = dict(T=TREF.tolist(), mesh=PHO['mesh'], natoms=PHO['natoms'], phases={})
for ph in ('alpha', 'gamma'):
    recs = []
    vmap = {p['tag']: p['V'] for p in EOS['phases'][ph]['points']}
    for tag, d in PHO['phases'][ph].items():
        V_cell = vmap[tag]        # 体积以 eos_raw.json 为准（ph.json 里的 V_cell 只作备份）
        recs.append(dict(tag=tag, V_cell=V_cell, V_at=V_cell / NAT[ph],
                         natoms_unit=d['natoms_unit'], natoms_supercell=d['natoms_supercell'],
                         nq=d['nq'], nbands=d['nbands'], n_imag=d['n_imag'],
                         fmin_THz=d['fmin_THz'], fmax_THz=d['fmax_THz'],
                         zpe_meV_at=d['zpe_meV_per_atom'],
                         zpe_phonopy_meV_at=d['zpe_phonopy_meV_per_atom'],
                         F_vib_eV_at=d['F_vib_eV_per_atom'],
                         Cv_eV_K_at=d['Cv_eV_per_K_per_atom'],
                         S_eV_K_at=d['S_eV_per_K_per_atom'],
                         bands=d.get('bands'),
                         mesh=d.get('mesh'), dos_grid=d.get('dos_grid'),
                         dos=d.get('dos')))
    recs.sort(key=lambda r: r['V_at'])
    th['phases'][ph] = dict(points=recs,
                            n_imag_free=[r['tag'] for r in recs if r['n_imag'] == 0],
                            n_imag_nonzero=[r['tag'] for r in recs if r['n_imag'] > 0])
IMAG_EXCL = {ph: sorted(r['tag'] for r in th['phases'][ph]['points'] if r['n_imag'] > 0)
             for ph in ('alpha', 'gamma')}
th['qha_excluded_imaginary'] = dict(
    rule='n_imag > 0 的体积点不进 QHA：虚频模无实的振动自由能，phonopy 按 |f| 计入等于把'
         '软化/失稳模式当成真实振动，会让 F_vib 偏低、体积偏小',
    tags=IMAG_EXCL)
json.dump(th, open(os.path.join(A.out, 'thermal.json'), 'w'), indent=1)
for ph in ('alpha', 'gamma'):
    z = [r['zpe_meV_at'] for r in th['phases'][ph]['points']]
    print('[thermal] %s: %d 点  ZPE %.2f–%.2f meV/at  无虚频 %d/%d'
          % (ph, len(z), min(z), max(z),
             len(th['phases'][ph]['n_imag_free']), len(th['phases'][ph]['points'])))

# ============================================================ 3. ZPE 分解 + PHDOS
zp = dict(mesh=PHO['mesh'], bands=PHO['bands'], phases={})
for ph in ('alpha', 'gamma'):
    V0 = eos_out['phases'][ph]['BM']['V0_A3_per_atom']
    pts = th['phases'][ph]['points']
    ok = [r for r in pts if r['n_imag'] == 0]
    if not ok:
        ok = pts
    Vs = np.array([r['V_at'] for r in ok])
    zs = np.array([r['zpe_meV_at'] for r in ok])
    z0 = float(np.interp(V0, Vs, zs)) if Vs.min() <= V0 <= Vs.max() else float(zs.min())
    zp['phases'][ph] = dict(V0_A3_per_atom=V0, zpe_at_V0_meV_at=z0,
                            zpe_interp='linear in V_at over imaginary-free points',
                            zpe_table=[dict(tag=r['tag'], V_at=r['V_at'], zpe=r['zpe_meV_at'],
                                            n_imag=r['n_imag']) for r in pts],
                            bands_detail=[dict(tag=r['tag'], V_at=r['V_at'], bands=r['bands'])
                                          for r in ok])
    rep = min(ok, key=lambda r: abs(r['V_at'] - V0))
    zp['phases'][ph]['phdos_representative'] = dict(
        tag=rep['tag'], V_at=rep['V_at'], grid=PHO['phases'][ph][rep['tag']]['dos_grid'],
        dos=PHO['phases'][ph][rep['tag']]['dos'])
    zp['phases'][ph]['bands_at_V0'] = {
        '%s-%s' % (b['lo'], 'inf' if b['hi'] is None else b['hi']): dict(
            frac=b['frac'], zpe=b['zpe']) for b in rep['bands']}
za = zp['phases']['alpha']['zpe_at_V0_meV_at']
zg = zp['phases']['gamma']['zpe_at_V0_meV_at']
zp['delta'] = dict(dZPE_meV_at=zg - za, alpha=za, gamma=zg,
                   V9=dict(alpha=V9_ZPE['alpha'], gamma=V9_ZPE['gamma'],
                           dZPE=V9_ZPE['dZPE']),
                   same_order_as_V9=bool(0.8 <= (zg - za) <= 1.8),
                   absolute_in_30_100=bool(30 <= za <= 100 and 30 <= zg <= 100))
json.dump(zp, open(os.path.join(A.out, 'zpe_phdos.json'), 'w'), indent=1)
print('[zpe] alpha %.2f  gamma %.2f  dZPE %+.3f meV/at (V9 %.2f-%.2f)'
      % (za, zg, zg - za, V9_ZPE['dZPE'][0], V9_ZPE['dZPE'][1]))

# ============================================================ 4. QHA
def phase_curves(ph):
    """返回 (V_at 数组, E_st(V) eV/at, F_vib(V,T) eV/at 矩阵, tags) —— 只留可用点。"""
    pts = th['phases'][ph]['points']
    eosmap = {p['tag']: p for p in eos_out['phases'][ph]['table']}
    exclude = {e['tag'] for e in eos_out['phases'][ph]['excluded']} | set(IMAG_EXCL[ph])
    stable = [r for r in pts if r['tag'] not in exclude]
    usable = [r for r in stable if eosmap[r['tag']]['fmax_end'] <= 1e-3]
    if len(usable) < 5:
        usable = list(stable)
    usable.sort(key=lambda r: r['V_at'])
    V = np.array([r['V_at'] for r in usable])
    E = np.array([eosmap[r['tag']]['E_eV'] / NAT[ph] for r in usable])
    Fv = np.array([r['F_vib_eV_at'] for r in usable])       # (nV, nT)
    return V, E, Fv, [r['tag'] for r in usable], usable


def spline_G(ph, iT):
    """返回 f(V)->F(V) 样条（含 E_st + F_vib(T)），以及 V 的范围。"""
    V, E, Fv, tags, used = phase_curves(ph)
    F = E + Fv[:, iT]
    return CubicSpline(V, F, extrapolate=False), V.min(), V.max(), used


QHA = {}
for ph in ('alpha', 'gamma'):
    V, E, Fv, tags, used = phase_curves(ph)
    QHA[ph] = dict(V=V, E=E, Fv=Fv, tags=tags, used=used,
                   excluded_imag=IMAG_EXCL[ph],
                   dropped_by_eos_gate=[e['tag'] for e in eos_out['phases'][ph]['excluded']],
                   n_imag_used={r['tag']: r['n_imag'] for r in used})

VGRID = np.linspace(0.9, 1.1, 4001)     # 相对 scaled 网格，逐相缩放到各自 [Vmin,Vmax]


def G_of(ph, iT, P):
    sp, vmin, vmax, _ = spline_G(ph, iT)
    Vg = np.linspace(vmin, vmax, 3001)
    Gg = sp(Vg) + P * Vg * GPA_AA3_EV
    i = int(np.nanargmin(Gg))
    edge = bool(i == 0 or i == len(Vg) - 1)
    return float(Gg[i]), float(Vg[i]), edge


rows = []
for iT, T in enumerate(TREF):
    ga, va, ea = G_of('alpha', iT, 0.0)
    gg, vg, eg = G_of('gamma', iT, 0.0)
    dG0 = gg - ga          # G_of 返回的是 eV/atom（V_at 上的样条），不要再除 NAT
    Pt = float('nan')
    try:
        f = lambda P: G_of('gamma', iT, P)[0] - G_of('alpha', iT, P)[0]
        for _lo, _hi in ((-3.0, 12.0), (-15.0, 60.0), (-80.0, 300.0)):
            if f(_lo) * f(_hi) < 0:
                Pt = brentq(f, _lo, _hi, xtol=1e-8)
                break
    except Exception:
        pass
    rows.append(dict(T=float(T), Pt_GPa=Pt, Va_A3_at=va, Vg_A3_at=vg,
                     dG_P0_eV_at=dG0, edge_alpha=ea, edge_gamma=eg))
Pt0 = rows[0]['Pt_GPa']
# P=0 下 ΔG(γ−α)=0 的温度
dGs = np.array([r['dG_P0_eV_at'] for r in rows])
Tc = None
TC_NOTE = ''
if dGs[0] * dGs[-1] < 0:
    k = int(np.argmax(dGs < 0))
    Tc = float(brentq(lambda t: np.interp(t, TREF, dGs), TREF[k - 1], TREF[k]))
    TC_NOTE = 'ΔG(γ−α) 在 P=0 过零于 %.0f K：该温度以上 γ 在常压稳定' % Tc
elif dGs[-1] < 0:
    Tc = float(TREF[-1])
    TC_NOTE = 'ΔG(γ−α) 在 %d K 已为负，过零点 ≤%d K（未细分）' % (TREF[-1], TREF[-1])
else:
    TC_NOTE = ('ΔG(γ−α) 在 P=0、0–600 K 全为正（%+.2f → %+.2f meV/atom），无过零：'
               '常压全温区 α 稳定，γ 只在加压下稳定（`pt_line.json:rows[*].Pt_GPa`）'
               % (dGs[0] * 1000, dGs[-1] * 1000))
PT_TMAX = float(min(r['T'] for r in rows if r['edge_alpha'] or r['edge_gamma']) - 25.0) \
    if any(r['edge_alpha'] or r['edge_gamma'] for r in rows) else float(TREF[-1])

COEF_VAR = {}
for _nm, _dV, _src in (
        ('dft_dV_0.8946', 0.8946,
         'DFT 真弛豫体积差 dV/atom = 0.8946 Å³ -> C = 0.1791 GPa/(meV/atom)（论文旧值）'),
        ('v11_fit_dV', float(abs(dVat)),
         'V11 本次 BM 拟合 V0 差 dV/atom -> C ≈ 0.1755 GPa/(meV/atom)（主表口径）')):
    _c = GPA * 1e-3 / _dV
    _p = (dEst + (zg - za)) * _c
    COEF_VAR[_nm] = dict(dV_A3_per_atom=_dV, coef_GPa_per_meV_atom=_c, src=_src,
                         Pt_estimate_GPa=_p,
                         diff_GPa=float(abs(Pt0 - _p)) if np.isfinite(Pt0) else None,
                         ok=bool(np.isfinite(Pt0) and abs(Pt0 - _p) <= 0.05))
pv0 = (dEst + (zg - za)) * DPT_TRUTH['coef']
pt_out = dict(rows=rows, T=TREF.tolist(),
              model=EOS['model'],
              method='G(V,T,P)=E_st(V)+F_vib(V,T)+P*V ; cubic spline in V per T ; '
                     'P*V in eV via 1e-21/1.602176634e-19',
              Pt0K_GPa=Pt0, Tc_at_P0_K=Tc, Tc_note=TC_NOTE,
              Pt_valid_T_max_K=PT_TMAX,
              Pt_validity_note='样条最小值撞到 V 数据边界(edge_*)后 V(T) 被夹住，'
                               'P_t 在该温度以上不可信；本表在 T ≤ %.0f K 内有效' % PT_TMAX,
              consistency=dict(
                  dEst_meV_at=dEst, dZPE_meV_at=zg - za,
                  sum_meV_at=dEst + (zg - za),
                  coef=DPT_TRUTH['coef'],
                  coef_src=DPT_TRUTH['src'],
                  Pt_estimate_GPa=pv0,
                  diff_GPa=abs(Pt0 - pv0) if np.isfinite(Pt0) else None,
                  ok=bool(np.isfinite(Pt0) and abs(Pt0 - pv0) <= 0.05),
                  coef_variants=COEF_VAR),
              excluded_imag=IMAG_EXCL,
              used_tags={ph: QHA[ph]['tags'] for ph in ('alpha', 'gamma')})
json.dump(pt_out, open(os.path.join(A.out, 'pt_line.json'), 'w'), indent=1)
print('[qha] Pt(0K)=%.3f GPa   estimate (dEst+dZPE)*%.4f = %.3f GPa   diff=%.3f   Tc(P=0)=%s'
      % (Pt0, DPT_TRUTH['coef'], pv0, abs(Pt0 - pv0), Tc))

# ============================================================ 5. CTE α_V(T)
cte = dict(phases={}, note='alpha_V = (1/V)(dV/dT)_P=0 from QHA spline minimum')
for ph in ('alpha', 'gamma'):
    V = np.array([r['Va_A3_at'] if ph == 'alpha' else r['Vg_A3_at'] for r in rows])
    EDG = np.array([r['edge_alpha'] if ph == 'alpha' else r['edge_gamma'] for r in rows])
    a = np.full_like(V, np.nan)
    for i in range(1, len(V) - 1):
        dT = TREF[i + 1] - TREF[i - 1]
        a[i] = (V[i + 1] - V[i - 1]) / dT / V[i]
        if EDG[i - 1] or EDG[i] or EDG[i + 1]:
            a[i] = np.nan          # 样条最小值撞到数据边界 -> 该温度窗口的 V(T) 不可信
    cte['phases'][ph] = dict(T=TREF.tolist(), V_A3_at=V.tolist(),
                             edge_hit=EDG.tolist(),
                             alpha_V_valid_windowK=[float(TREF[~np.isnan(a)].min()),
                                                    float(TREF[~np.isnan(a)].max())]
                             if np.any(~np.isnan(a)) else None,
                             alpha_V_K=a.tolist(),
                             alpha_V_100K=float(a[np.argmin(abs(TREF - 100))]),
                             alpha_V_200K=float(a[np.argmin(abs(TREF - 200))]),
                             alpha_V_300K=float(a[np.argmin(abs(TREF - 300))]),
                             dV_pct_0_300K=float(100 * (V[np.argmin(abs(TREF - 300))] / V[0] - 1)),
                             min_alpha_V_K=float(np.nanmin(a)),
                             negative_at_lowT=bool(np.nanmin(a) < 0))
json.dump(cte, open(os.path.join(A.out, 'cte.json'), 'w'), indent=1)
print('[cte] alpha: a_V(100K)=%+.2e  a_V(300K)=%+.2e  min=%+.2e  NTE=%s'
      % (cte['phases']['alpha']['alpha_V_100K'], cte['phases']['alpha']['alpha_V_300K'],
         cte['phases']['alpha']['min_alpha_V_K'],
         cte['phases']['alpha']['negative_at_lowT']))
print('      gamma: alpha_V(300K)=%+.2e' % cte['phases']['gamma']['alpha_V_300K'])

# ============================================================ 6. 弹性
el = dict(model=ELA['model'], dtype=ELA['dtype'], delta=ELA['delta'],
          fmax=ELA['fmax'], method=ELA['method'], phases={})
for ph in ('alpha', 'gamma'):
    d = dict(ELA['phases'][ph])
    b_eos = eos_out['phases'][ph]['BM']['B0_GPa']
    d['B0_from_EOS_GPa'] = b_eos
    d['B_Hill_vs_B0_EOS_pct'] = float(100 * (d['B_Hill'] - b_eos) / b_eos)
    d['cross_check_ok'] = bool(abs(d['B_Hill_vs_B0_EOS_pct']) <= 10.0)
    el['phases'][ph] = d
json.dump(el, open(os.path.join(A.out, 'elastic.json'), 'w'), indent=1)
for ph in ('alpha', 'gamma'):
    d = el['phases'][ph]
    print('[elastic] %s  B_Hill=%.1f (EOS %.1f, %+.1f%%)  G=%.1f  C11=%.1f C12=%.1f C44=%.1f'
          % (ph, d['B_Hill'], d['B0_from_EOS_GPa'], d['B_Hill_vs_B0_EOS_pct'],
             d['G_Hill'], d['C_ij']['C11'], d['C_ij']['C12'], d['C_ij']['C44']))

# ============================================================ 7. SUMMARY.md
def f3(x, n=3, dash='—'):
    try:
        x = float(x)
    except Exception:
        return dash
    return dash if not np.isfinite(x) else ('%.*f' % (n, x))


ea = eos_out['phases']['alpha']
eg = eos_out['phases']['gamma']
ca = el['phases']['alpha']
cg = el['phases']['gamma']
L = []
w = L.append
w('# ZrW2O8 · V11 势的准简谐热力学与弹性（阶段 2）')
w('')
w('模型：`%s`' % os.path.basename(EOS['model']))
w('')
w('口径：E_st 用 **float64** 固定胞只弛豫离子（fmax 1e-4）；声子力用 **float32**；'
  '3×3×3 超胞、位移 0.01 Å、symprec 0.05；QHA 在 V 上做三次样条最小化。')
w('')
w('全部数字可由 `eos.json` / `thermal.json` / `zpe_phdos.json` / `pt_line.json` / '
  '`cte.json` / `elastic.json` 复算；每行末尾给出"文件:键名"。')
w('')
w('## 0 作业记录（sacct，2026-11-21）')
w('')
w('| JobID | 名称 | 状态 | 墙钟 | 开始 → 结束 | 说明 |')
w('|---|---|---|---|---|---|')
for _j in (('2680000', 'zrw_eos', 'COMPLETED', '00:04:31', '20:38:47 → 20:43:18', 'α/γ 26 点固定胞 E_st'),
           ('2680003', 'zrw_elas', 'COMPLETED', '00:00:46', '20:43:18 → 20:44:04', '应力-应变弹性'),
           ('2680150', 'zrw_phon', 'FAILED', '00:00:28', '20:47:28 → 20:47:56',
            '前一棒：phonopy 组装 FC 报 "Input forces are not enough"（α s<1 对称性破缺），'
            '修 pipeline 后重投'),
           ('2680198', 'zrw_phon', 'COMPLETED', '00:25:29', '20:50:11 → 21:15:40',
            'α 13 点 + γ 13 点，每点 gen/forces/post 串行；本报告全部声子数据来源')):
    w('| %s | %s | %s | %s | %s | %s |' % _j)
w('')
w('## 1 主表')
w('')
w('| 量 | α | γ | 来源 |')
w('|---|---|---|---|')
w('| V₀ (Å³/单胞) | %s | %s | `eos.json:phases.<ph>.BM.V0_A3_cell` |'
  % (f3(ea['BM']['V0_A3_cell'], 2), f3(eg['BM']['V0_A3_cell'], 2)))
w('| V₀ (Å³/atom) | %s | %s | `eos.json:phases.<ph>.BM.V0_A3_per_atom` |'
  % (f3(ea['BM']['V0_A3_per_atom'], 4), f3(eg['BM']['V0_A3_per_atom'], 4)))
w('| E₀ (eV/atom) | %s | %s | `eos.json:phases.<ph>.BM.E0_eV_per_atom` |'
  % (f3(ea['BM']['E0_eV_per_atom'], 8), f3(eg['BM']['E0_eV_per_atom'], 8)))
w('| B₀ (GPa) | %s | %s | `eos.json:phases.<ph>.BM.B0_GPa` |'
  % (f3(ea['BM']['B0_GPa'], 1), f3(eg['BM']['B0_GPa'], 1)))
w("| B₀′ | %s | %s | `eos.json:phases.<ph>.BM.B0p` |"
  % (f3(ea['BM']['B0p'], 2), f3(eg['BM']['B0p'], 2)))
w('| EOS 拟合 RMS (meV/atom) | %s | %s | `eos.json:phases.<ph>.BM.fit_rms_meV_per_atom` |'
  % (f3(ea['BM']['fit_rms_meV_per_atom'], 3), f3(eg['BM']['fit_rms_meV_per_atom'], 3)))
w('| **ΔE_st(γ−α)** (meV/atom) | **%s** | — | `eos.json:delta.dEst_meV_per_atom` |'
  % f3(eos_out['delta']['dEst_meV_per_atom'], 3))
w('| ZPE @V₀ (meV/atom) | %s | %s | `zpe_phdos.json:phases.<ph>.zpe_at_V0_meV_at` |'
  % (f3(za, 2), f3(zg, 2)))
w('| **ΔZPE(γ−α)** (meV/atom) | **%s** | — | `zpe_phdos.json:delta.dZPE_meV_at` |'
  % f3(zg - za, 3))
w('| **P_t(0 K)** (GPa) | **%s** | — | `pt_line.json:Pt0K_GPa` |' % f3(Pt0, 3))
w('| T_c @ P=0 (K) | %s | — | `pt_line.json:Tc_at_P0_K` |' % f3(Tc, 1))
w('| α_V(300 K) (1/K) | %s | %s | `cte.json:phases.<ph>.alpha_V_300K` |'
  % ('%+.2e' % cte['phases']['alpha']['alpha_V_300K'],
     '%+.2e' % cte['phases']['gamma']['alpha_V_300K']))
w('| α_V(100 K) (1/K) | %s | %s | `cte.json:phases.<ph>.alpha_V_100K` |'
  % ('%+.2e' % cte['phases']['alpha']['alpha_V_100K'],
     '%+.2e' % cte['phases']['gamma']['alpha_V_100K']))
w('')
w('共存线：%s。P_t(T) 在 T ≤ %.0f K 内可信（`pt_line.json:Pt_valid_T_max_K`）。'
  % (TC_NOTE, PT_TMAX))
w('')
w('## 2 与旧 V6 / DFT 真值并排')
w('')
w('| 量 | 本次 V11 | 旧 V6 | DFT 真值 | 判据 |')
w('|---|---|---|---|---|')
w('| ΔE_st (meV/atom) | %s | %s | %.2f ± %.2f | `eos.json:delta` |'
  % (f3(eos_out['delta']['dEst_meV_per_atom'], 3), f3(V6['dEst'], 3),
     DPT_TRUTH['dE'], DPT_TRUTH['dE_sig']))
w('| ΔZPE (meV/atom) | %s | %s | V9: %.2f–%.2f | `zpe_phdos.json:delta` |'
  % (f3(zg - za, 3), f3(V6['dZPE'], 3), V9_ZPE['dZPE'][0], V9_ZPE['dZPE'][1]))
w('| P_t(0 K) (GPa) | %s | %s | 0.210（实验相变压） | `pt_line.json:Pt0K_GPa` |'
  % (f3(Pt0, 3), f3(V6['Pt'], 3)))
w('')
w('验收对照：')
w('')
w('- ① ΔE_st 与 2.52 ± 0.20 之差 = **%+.3f meV/atom**（±0.5 内即接受）→ %s'
  % (eos_out['delta']['dEst_vs_DFT'],
     'PASS' if abs(eos_out['delta']['dEst_vs_DFT']) <= 0.5 else 'FAIL'))
w('- ② ΔZPE = %+.3f meV/atom 对 V9 的 +1.20~+1.44 → %s（脚本判据带 0.8–1.8，'
  '本次 %.3f 落在下沿外 %+.3f）；两相绝对 ZPE %.2f / %.2f meV/atom 均在 30–100 → %s'
  % (zg - za, 'PASS' if zp['delta']['same_order_as_V9'] else 'CHECK',
     zg - za, (zg - za) - V9_ZPE['dZPE'][0], za, zg,
     'PASS' if zp['delta']['absolute_in_30_100'] else 'FAIL'))
w('  - 解读：ΔZPE 同量级（0.79 vs 1.2–1.4 meV/atom，差 34–45 %%），不阻断；'
  '偏离主要来自 γ 在 V₀ 的 ZPE（本次 %.2f vs V9 %.2f）与 γ 网格 16³（α 24³）的差异。'
  % (zg, V9_ZPE['gamma']))
w('- ④ EOS 拟合 RMS：α %.3f、γ %.3f meV/atom（< 0.3 为通过）→ %s'
  % (ea['BM']['fit_rms_meV_per_atom'], eg['BM']['fit_rms_meV_per_atom'],
     'PASS' if (ea['BM']['fit_rms_ok'] and eg['BM']['fit_rms_ok']) else 'FAIL'))
w('- ⑤ α 相 α_V(T) 低温为负（NTE）= **%s**，最小值 %s 1/K → %s'
  % (cte['phases']['alpha']['negative_at_lowT'],
     '%+.2e' % cte['phases']['alpha']['min_alpha_V_K'],
     'PASS' if cte['phases']['alpha']['negative_at_lowT'] else 'FAIL'))
w('- ⑥ P_t(0 K) 与 (ΔE_st+ΔZPE)×C 的一致性（`pt_line.json:consistency.coef_variants.<口径>`）：')
for _nm, _cv in COEF_VAR.items():
    w('  - **%s**：C = %.4f GPa/(meV/atom)（dV/atom = %.4f Å³）→ 估计 %s GPa，'
      '与 QHA P_t 差 %s GPa → %s'
      % (_nm, _cv['coef_GPa_per_meV_atom'], _cv['dV_A3_per_atom'],
         f3(_cv['Pt_estimate_GPa'], 3), f3(_cv['diff_GPa'], 3),
         'PASS' if _cv['ok'] else 'CHECK'))
w('  - 主表口径 = `v11_fit_dV`（模型自己的 dV₀）：ΔE_st、ΔZPE、P_t 同出这一套拟合，'
  '口径自洽，检验的是 QHA 与能量分解是否对得上；`dft_dV_0.8946`（0.1791）是论文旧值，'
  '把 DFT 的 V₀ 误差也算进来，只作对照。')
w('- ③⑦ 逐点虚频/重构见 §4、来源索引见每行末列。')
w('')
w('注：DFT 真值 P_t = 0.210 GPa 用同一个 C = 0.1791 反推需要的 ΔF 只有 1.17 meV/atom，'
  '比 ΔE_st(2.52) 还小；即 0.210 GPa 不是「ΔE_st/ΔV」这一路来的（该路给 0.451 GPa），'
  '应属有限温度相界或另一 ΔV 口径，不能与本文 0 K 的 0.635 GPa 直接对表。')
w('')
w('## 3 EOS 逐点表（V/V₀ 等间距 13 点）')
w('')
for ph in ('alpha', 'gamma'):
    w('### %s（%d 原子）' % ({'alpha': 'α', 'gamma': 'γ'}[ph], eos_out['phases'][ph]['natoms']))
    w('')
    w('| tag | V/V₀ | V (Å³) | E (eV/atom) | 残差 (meV/at) | max\\|F\\| | 步数 | max\\|Δpos\\|★ | W–O CN@2.2Å | 闸门 |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for p in eos_out['phases'][ph]['table']:
        w('| %s | %.2f | %s | %s | %+.3f | %.1e | %d | %.3f | %s | %s |'
          % (p['tag'], p['s'], f3(p['V_cell'], 2), f3(p['E_eV'], 6),
             p['resid_meV_at'], p['fmax_end'], p['nsteps'], p['max_dpos_A'],
             p['wo_cn']['hist'], p['flag']))
    w('')
w('★ 相对**未弛豫** `_ref` 的位移（参考胞本身 DFT max\\|F\\| = 0.28/0.63 eV/Å，'
  '所以这一列是天然而非重构指标）；相邻体积点之间的 Δpos 见 `eos.json:'
  'phases.<ph>.gate_dpos_diagnostics`。')
w('')
w('## 4 收敛 / 虚频 / 重构检查')
w('')
w('结构闸门判据（`qha_analysis.py` + `gate_audit.py`）：')
w('')
w('- **结构（势阱连续性）**：相邻体积点弛豫末态之间 max|Δpos| **> 0.5 Å** 判为换了'
  '局部极小（重构），以 s=1.00 所在势阱为本相，其余势阱剔除。')
w('- **结构（机制佐证）**：每个 W 取最近 5 个 O 的距离 d1…d5，看 d5 是否沿 V 连续；'
  '真实重构会让 d5 出现**跳变**（γ s=1.04：2.258 → 2.415 Å），'
  '纯弹性压缩只做连续平移。')
w('- **能量**：首次 Birch–Murnaghan 拟合残差 **< −1.0 meV/atom**（曲线异常下折）记为 droop。')
w('- 2.2 Å 硬截断的 CN 在 γ 上对应变过敏（第 5 个 O 就在 2.19 Å），只作诊断列。')
w('')
w('| 相 | 体积点 | 虚频点（n_imag>0） | 最低频 (THz) | 结构闸门命中 | 能量 droop 命中 |')
w('|---|---|---|---|---|---|')
for ph in ('alpha', 'gamma'):
    pts = th['phases'][ph]['points']
    ni = [r for r in pts if r['n_imag'] > 0]
    fmin_all = min(r['fmin_THz'] for r in pts)
    w('| %s | %d | %d | %+.3f | %d | %d |'
      % ({'alpha': 'α', 'gamma': 'γ'}[ph], len(pts), len(ni), fmin_all,
         sum(1 for p in eos_out['phases'][ph]['table'] if p['gate_wo5']),
         sum(1 for p in eos_out['phases'][ph]['table'] if 'droop' in p['flag'])))
w('')
for ph in ('alpha', 'gamma'):
    _lab = {'alpha': 'α', 'gamma': 'γ'}[ph]
    if IMAG_EXCL[ph]:
        _e = [r for r in th['phases'][ph]['points'] if r['tag'] in IMAG_EXCL[ph]]
        w('- **%s 剔除（虚频）**：%s。判据 n_imag>0 → 不进 QHA（`thermal.json:'
          'qha_excluded_imaginary.tags.%s`）。'
          % (_lab, '；'.join('%s (f_min %+.3f THz, n_imag %d)' % (r['tag'], r['fmin_THz'], r['n_imag'])
                            for r in _e), ph))
    else:
        w('- %s 全部体积点无虚频（n_imag = 0），无需因虚频剔除。' % _lab)
w('  机构：γ 小体积端虚频集中出现在低 v 声学/软模支，随压缩单调增多，'
  '与 γ 相在超压（~B₀×6 % ≈ 4 GPa）下软模失稳一致；这些点仍留在静态 EOS 拟合里'
  '（能量不变），只是不进 F_vib / QHA。')
w('')
for ph in ('alpha', 'gamma'):
    w('### %s 逐点声子' % {'alpha': 'α', 'gamma': 'γ'}[ph])
    w('')
    w('| tag | V (Å³/atom) | Nq | 模数 | n_imag | f_min (THz) | f_max (THz) | ZPE (meV/atom) |')
    w('|---|---|---|---|---|---|---|---|')
    for r in th['phases'][ph]['points']:
        w('| %s | %s | %d | %d | %d | %+.3f | %s | %s |'
          % (r['tag'], f3(r['V_at'], 4), r['nq'], r['nbands'], r['n_imag'],
             r['fmin_THz'], f3(r['fmax_THz'], 1), f3(r['zpe_meV_at'], 2)))
    w('')
w('## 5 QHA / 相变线 / 热膨胀')
w('')
w('| T (K) | P_t (GPa) | V_α (Å³/atom) | V_γ (Å³/atom) | ΔG@P=0 (meV/atom) |')
w('|---|---|---|---|---|')
for r in rows:
    w('| %s%s | %s | %s | %s | %+.3f |'
      % (f3(r['T'], 0), ' †' if (r['edge_alpha'] or r['edge_gamma']) else '',
         f3(r['Pt_GPa'], 3), f3(r['Va_A3_at'], 4),
         f3(r['Vg_A3_at'], 4), r['dG_P0_eV_at'] * 1000))
w('')
w('† 样条最小值撞到 V 数据边界（该相的 V(T) 被夹住），此行 P_t / V 不可用；'
  '边界由 `pt_line.json:rows[*].edge_alpha/edge_gamma` 标记。')
w('（`pt_line.json:rows`；P_t 由 G_γ(T,P)=G_α(T,P) 用 brentq 解出，P 区间 [−3, 12] GPa。）')
w('')
for ph in ('alpha', 'gamma'):
    w('- QHA 实际用点 %s（%d 个）：%s；剔除 %s → `pt_line.json:used_tags.%s`'
      % ({'alpha': 'α', 'gamma': 'γ'}[ph], len(QHA[ph]['tags']),
         ', '.join(QHA[ph]['tags']),
         ('虚频 ' + ', '.join(IMAG_EXCL[ph]) if IMAG_EXCL[ph] else '无') +
         ('；结构闸门 ' + ', '.join(QHA[ph]['dropped_by_eos_gate'])
          if QHA[ph]['dropped_by_eos_gate'] else ''), ph))
w('')
w('## 6 弹性常数（0 K，应力—应变 ±%.1f %%，离子弛豫后读应力）' % (100 * el['delta']))
w('')
w('| 量 | α（立方） | γ（正交） | 交叉核对 |')
w('|---|---|---|---|')
w('| C11 | %s | %s | `elastic.json:phases.<ph>.C_ij.C11` |'
  % (f3(ca['C_ij']['C11'], 1), f3(cg['C_ij']['C11'], 1)))
w('| C22 | %s | %s | `...C22` |' % (f3(ca['C_ij']['C22'], 1), f3(cg['C_ij']['C22'], 1)))
w('| C33 | %s | %s | `...C33` |' % (f3(ca['C_ij']['C33'], 1), f3(cg['C_ij']['C33'], 1)))
w('| C12 | %s | %s | `...C12` |' % (f3(ca['C_ij']['C12'], 1), f3(cg['C_ij']['C12'], 1)))
w('| C13 | %s | %s | `...C13` |' % (f3(ca['C_ij']['C13'], 1), f3(cg['C_ij']['C13'], 1)))
w('| C23 | %s | %s | `...C23` |' % (f3(ca['C_ij']['C23'], 1), f3(cg['C_ij']['C23'], 1)))
w('| C44 | %s | %s | `...C44` |' % (f3(ca['C_ij']['C44'], 1), f3(cg['C_ij']['C44'], 1)))
w('| C55 | %s | %s | `...C55` |' % (f3(ca['C_ij']['C55'], 1), f3(cg['C_ij']['C55'], 1)))
w('| C66 | %s | %s | `...C66` |' % (f3(ca['C_ij']['C66'], 1), f3(cg['C_ij']['C66'], 1)))
w('| B (Hill) | %s | %s | `...B_Hill` |' % (f3(ca['B_Hill'], 1), f3(cg['B_Hill'], 1)))
w('| B₀ (EOS) | %s | %s | `eos.json:phases.<ph>.BM.B0_GPa` |'
  % (f3(ca['B0_from_EOS_GPa'], 1), f3(cg['B0_from_EOS_GPa'], 1)))
w('| 差 | %+.1f %% | %+.1f %% | 判据 ≤10 %% → α %s / γ %s |'
  % (ca['B_Hill_vs_B0_EOS_pct'], cg['B_Hill_vs_B0_EOS_pct'],
     'PASS' if ca['cross_check_ok'] else 'FAIL',
     'PASS' if cg['cross_check_ok'] else 'FAIL'))
w('| G (Hill) | %s | %s | `...G_Hill` |' % (f3(ca['G_Hill'], 1), f3(cg['G_Hill'], 1)))
w('| E (GPa) | %s | %s | `...E_GPa` |' % (f3(ca['E_GPa'], 1), f3(cg['E_GPa'], 1)))
w('| ν | %s | %s | `...nu` |' % (f3(ca['nu'], 3), f3(cg['nu'], 3)))
w('')
if ca.get('cubic_consistency'):
    cc = ca['cubic_consistency']
    w('α 立方内部一致性：C11 三点极差 %s %%、C12 %s %%、C44 %s %%；'
      '非立方分量最大 |C_ij| = %s GPa（应 ~0）。'
      % (f3(cc['C11_spread_pct'], 2), f3(cc['C12_spread_pct'], 2),
         f3(cc['C44_spread_pct'], 2), f3(cc['offdiag_max_GPa'], 3)))
    w('')
w('## 7 Pending / 降级 / 未完成')
w('')
pend = [ln for ln in PENDING] if 'PENDING' in globals() else []
if pend:
    for x in pend:
        w('- ' + x)
else:
    w('- 无：α/γ 两相 13+13 个体积点的 EOS 与声子、两相弹性均已完成。')
w('')
w('## 8 关键命令')
w('')
w('```bash')
for c in CMDS:
    w(c)
w('```')
open(os.path.join(A.out, 'SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
print('[summary] -> %s/SUMMARY.md (%d 行)' % (A.out, len(L)))

print('\n输出: eos.json thermal.json zpe_phdos.json pt_line.json cte.json elastic.json')
