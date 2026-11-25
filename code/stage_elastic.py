#!/usr/bin/env python3
"""stage_elastic.py — 应力—应变法弹性常数（0 K，float64）。

基态 = eos/<ph>/vol_1.0000 的固定胞弛豫结构（V = ABACUS 真弛豫 V0）。
每个独立应变 j（Voigt 1..6）取 ±delta（默认 0.005 = ±0.5 %），
施加均匀应变后只弛豫离子（cell 冻结），读 ASE 应力，
C[:,j] = (sigma(+d) - sigma(-d)) / (2 d) / GPa。

用法: QHA_ROOT=<.../cluster_qha_v11> python stage_elastic.py [--delta 0.005]
"""
import os, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from ase.units import GPa
from mace.calculators import MACECalculator

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.environ.get('QHA_ROOT', '.'))
ap.add_argument('--delta', type=float, default=0.005)
ap.add_argument('--fmax', type=float, default=float(os.environ.get('FMAX_E', 1e-3)))
ap.add_argument('--phases', default=os.environ.get('PHASES', 'alpha,gamma'))
ap.add_argument('--model', default=os.environ.get(
    'MACE_MODEL', '/home/jinshuaishuai/workdir_tyut-youzhiyong/jinshuaishuai/'
                  'cluster_v11/ZrW_v11_seed3_stagetwo.model'))
ap.add_argument('--dtype', default=os.environ.get('DTYPE', 'float64'))
a = ap.parse_args()

VOIGT_IJ = {3: (1, 2), 4: (0, 2), 5: (0, 1)}
NAMES = ['C11', 'C22', 'C33', 'C12', 'C13', 'C23', 'C44', 'C55', 'C66']
IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2), (3, 3), (4, 4), (5, 5)]

calc = MACECalculator(model_paths=[a.model], device=os.environ.get('DEV', 'cuda'),
                      default_dtype=a.dtype)
print('model=%s dtype=%s delta=%g fmax=%g' % (a.model, a.dtype, a.delta, a.fmax), flush=True)


def strain_tensor(j, s):
    e = np.zeros((3, 3))
    if j < 3:
        e[j, j] = s
    else:
        i, k = VOIGT_IJ[j]
        e[i, k] = e[k, i] = s / 2.0
    return e


out = {'model': a.model, 'dtype': a.dtype, 'delta': a.delta, 'fmax': a.fmax,
       'method': 'stress-strain, +/-delta, ion-only relaxation at frozen cell',
       'phases': {}}

for ph in a.phases.split(','):
    st = json.load(open(os.path.join(a.root, 'eos', ph, 'vol_1.0000', 'poscar.json')))
    from ase import Atoms
    at0 = Atoms(symbols=st['symbols'], cell=np.array(st['cell']),
                scaled_positions=np.array(st['scaled_positions']), pbc=True)
    n = len(at0)
    base_cell = at0.get_cell()[:]
    t0 = time.time()
    atb = at0.copy()
    atb.calc = calc
    opt = FIRE(atb, logfile=None, loginterval=100, maxstep=0.1)
    opt.run(fmax=a.fmax, steps=20000)
    sig0 = atb.get_stress(voigt=True) / GPa
    print('[%s] base V=%.3f natoms=%d E/at=%.8f sigma0=%s GPa (%.0fs)'
          % (ph, atb.get_volume(), n, atb.get_potential_energy() / n,
             np.round(sig0, 3), time.time() - t0), flush=True)

    C = np.zeros((6, 6))
    rows = []
    for j in range(6):
        sig = {}
        for sign in (+1, -1):
            at = atb.copy()
            at.calc = calc
            at.set_cell(base_cell @ (np.eye(3) + strain_tensor(j, sign * a.delta)),
                        scale_atoms=True)
            o = FIRE(at, logfile=None, loginterval=100, maxstep=0.1)
            o.run(fmax=a.fmax, steps=20000)
            at.calc = calc
            sig[sign] = at.get_stress(voigt=True) / GPa
            rows.append(dict(j=j, sign=sign, nsteps=int(o.nsteps),
                             fmax_end=float(np.abs(at.get_forces()).max()),
                             sigma_voigt_GPa=sig[sign].tolist()))
        C[:, j] = (sig[+1] - sig[-1]) / (2.0 * a.delta)
        print('  [%s] strain %d done (%s GPa) (%.0fs)'
              % (ph, j, np.round(C[:, j], 1), time.time() - t0), flush=True)
    C = 0.5 * (C + C.T)

    c11, c22, c33 = C[0, 0], C[1, 1], C[2, 2]
    c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]
    c44, c55, c66 = C[3, 3], C[4, 4], C[5, 5]
    Bv = ((c11 + c22 + c33) + 2 * (c12 + c13 + c23)) / 9.0
    Gv = ((c11 + c22 + c33) - (c12 + c13 + c23) + 3 * (c44 + c55 + c66)) / 15.0
    s = np.linalg.inv(C[:3, :3])
    Br = 1.0 / np.sum(s)
    Gr = 15.0 / (4 * (s[0, 0] + s[1, 1] + s[2, 2])
                 - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
                 + 3 * (1 / c44 + 1 / c55 + 1 / c66))
    Bh, Gh = (Bv + Br) / 2, (Gv + Gr) / 2
    E = 9 * Bh * Gh / (3 * Bh + Gh)
    nu = (3 * Bh - 2 * Gh) / (2 * (3 * Bh + Gh))
    rho = float(atb.get_masses().sum() / atb.get_volume() * 1.66053906660)

    # 立方相内部一致性：C11≈C22≈C33, C12≈C13≈C23, C44≈C55≈C66
    cub = None
    if ph == 'alpha':
        v111 = [c11, c22, c33]
        v112 = [c12, c13, c23]
        v144 = [c44, c55, c66]
        cub = dict(C11=[float(x) for x in v111], C12=[float(x) for x in v112],
                   C44=[float(x) for x in v144],
                   C11_spread_pct=float(100 * (max(v111) - min(v111)) / np.mean(v111)),
                   C12_spread_pct=float(100 * (max(v112) - min(v112)) / np.mean(v112)),
                   C44_spread_pct=float(100 * (max(v144) - min(v144)) / np.mean(v144)),
                   offdiag_max_GPa=float(max(abs(x) for x in
                                             (C[0, 3], C[0, 4], C[0, 5], C[1, 3], C[3, 4],
                                              C[3, 5], C[4, 5]))))

    out['phases'][ph] = dict(natoms=n, V=float(atb.get_volume()),
                             E_per_atom=float(atb.get_potential_energy() / n),
                             sigma0_GPa=sig0.tolist(),
                             C=C.tolist(),
                             C_ij={nm: float(C[i, j]) for nm, (i, j) in zip(NAMES, IDX)},
                             B_Voigt=Bv, B_Reuss=Br, B_Hill=Bh,
                             G_Voigt=Gv, G_Reuss=Gr, G_Hill=Gh,
                             E_GPa=E, nu=nu, rho_g_cm3=rho,
                             cubic_consistency=cub, strain_rows=rows)
    print('[%s] C=%s\n     B_Hill=%.1f GPa  G_Hill=%.1f  E=%.1f  nu=%.3f'
          % (ph, np.round(C, 1).tolist(), Bh, Gh, E, nu), flush=True)

os.makedirs(os.path.join(a.root, 'elastic'), exist_ok=True)
json.dump(out, open(os.path.join(a.root, 'elastic', 'elastic_raw.json'), 'w'), indent=1)
print('-> %s/elastic/elastic_raw.json' % a.root)
