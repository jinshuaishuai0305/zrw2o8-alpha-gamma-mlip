#!/usr/bin/env python3
"""phonon_pipeline.py — 有限位移声子（3x3x3，0.01 Å，symprec 0.05）三阶段。

stage gen    (phonopy env) : 从 eos/<ph>/<tag>/poscar.json 建 phonopy 对象，
                             生成对称约化位移，落 phonopy_disp.yaml / base.json / disps.json
stage forces (mace env)   : 逐位移点做一次 MACE 单点（float32），增量写 forces.jsonl
                             （已算过的索引跳过 -> 作业被 requeue 后可续跑）
stage post   (phonopy env) : 组装 FORCE_SETS -> FC -> mesh -> 热学量 -> ph.json

用法: python phonon_pipeline.py {gen|forces|post} --phase alpha --tag vol_1.0000 \
          --root <cluster_qha_v11> [--mesh 24] [--smat 3]
"""
import os, sys, json, time, hashlib, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np

THZ_TO_MEV = 4.135667696
KJMOL_TO_EV = 96.48533212331002
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 1e9)]

ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['gen', 'forces', 'post'])
ap.add_argument('--phase', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--root', default=os.environ.get('QHA_ROOT', '.'))
ap.add_argument('--smat', type=int, default=3)
ap.add_argument('--mesh', type=int, default=0)
ap.add_argument('--dist', type=float, default=0.01)
ap.add_argument('--symprec', type=float, default=0.05)
ap.add_argument('--dtype', default=os.environ.get('DTYPE', 'float32'))
ap.add_argument('--model', default=os.environ.get(
    'MACE_MODEL', '/home/jinshuaishuai/workdir_tyut-youzhiyong/jinshuaishuai/'
                  'cluster_v11/ZrW_v11_seed3_stagetwo.model'))
a = ap.parse_args()

MESH = a.mesh or {'alpha': 24, 'gamma': 16}[a.phase]
VDIR = os.path.join(a.root, 'phonons', a.phase, a.tag)
os.makedirs(VDIR, exist_ok=True)
DISP_YAML = os.path.join(VDIR, 'phonopy_disp.yaml')
FORCE_JSONL = os.path.join(VDIR, 'forces.jsonl')

# --------------------------------------------------------------------------- gen
if a.stage == 'gen':
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    src = os.path.join(a.root, 'eos', a.phase, a.tag, 'poscar.json')
    src_md5 = hashlib.md5(open(src, 'rb').read()).hexdigest()
    dj_f = os.path.join(VDIR, 'disps.json')
    if (os.path.exists(dj_f) and os.path.exists(DISP_YAML)
            and os.path.exists(os.path.join(VDIR, 'base.json'))
            and json.load(open(dj_f)).get('src_md5') == src_md5):
        print('[gen] %s/%s CACHED ndisp=%d' % (a.phase, a.tag,
                                               len(json.load(open(dj_f))['disps'])), flush=True)
        sys.exit(0)
    if os.path.exists(FORCE_JSONL):
        os.remove(FORCE_JSONL)
    st = json.load(open(src))
    uc = PhonopyAtoms(symbols=st['symbols'], cell=np.array(st['cell']),
                      scaled_positions=np.array(st['scaled_positions']))
    smat = np.eye(3, dtype=int) * a.smat
    ph = Phonopy(uc, supercell_matrix=smat, symprec=a.symprec)
    try:
        ph.generate_displacements(distance=a.dist, is_plusminus='auto', symprec=a.symprec)
    except TypeError:      # phonopy >=4 去掉了 generate_displacements 的 symprec（已由构造器接管）
        ph.generate_displacements(distance=a.dist, is_plusminus='auto')
    ph.save(DISP_YAML)
    sc = ph.supercell
    json.dump(dict(cell=np.array(sc.cell).tolist(), symbols=list(sc.symbols),
                   scaled_positions=np.array(sc.scaled_positions).tolist()),
              open(os.path.join(VDIR, 'base.json'), 'w'))
    disps = np.array(ph.displacements)
    json.dump(dict(natoms=len(sc), smat=smat.tolist(), distance=a.dist,
                   symprec=a.symprec, src_md5=src_md5,
                   disps=[dict(i=int(r[0]), d=[float(x) for x in r[1:4]]) for r in disps]),
              open(os.path.join(VDIR, 'disps.json'), 'w'))
    print('[gen] %s/%s natom_sc=%d ndisp=%d  (symprec=%.2f, dist=%.3f)'
          % (a.phase, a.tag, len(sc), len(disps), a.symprec, a.dist), flush=True)

# ------------------------------------------------------------------------ forces
elif a.stage == 'forces':
    from ase import Atoms
    from mace.calculators import MACECalculator

    base = json.load(open(os.path.join(VDIR, 'base.json')))
    dj = json.load(open(os.path.join(VDIR, 'disps.json')))
    disps = dj['disps']
    done = set()
    if os.path.exists(FORCE_JSONL):
        for ln in open(FORCE_JSONL):
            ln = ln.strip()
            if ln:
                try:
                    done.add(json.loads(ln)['k'])
                except Exception:
                    pass
    todo = [k for k in range(len(disps)) if k not in done]
    print('[forces] %s/%s natom=%d ndisp=%d done=%d todo=%d dtype=%s'
          % (a.phase, a.tag, len(base['symbols']), len(disps),
             len(done), len(todo), a.dtype), flush=True)
    if todo:
        calc = MACECalculator(model_paths=[a.model], device=os.environ.get('DEV', 'cuda'),
                              default_dtype=a.dtype)
        ref = Atoms(symbols=base['symbols'], cell=np.array(base['cell']),
                    scaled_positions=np.array(base['scaled_positions']), pbc=True)
        with open(FORCE_JSONL, 'a') as fh:
            for n, k in enumerate(todo):
                at = ref.copy()
                p = at.get_positions()
                p[disps[k]['i']] += np.array(disps[k]['d'])
                at.set_positions(p)
                at.calc = calc
                fr = at.get_forces()
                fh.write(json.dumps(dict(k=k, forces=np.asarray(fr).tolist(),
                                         e=float(at.get_potential_energy()))) + '\n')
                fh.flush()
                if n == 0 or (n + 1) % 25 == 0:
                    print('   disp %d/%d (%.1fs)' % (n + 1, len(todo), time.time() % 1e5),
                          flush=True)
    print('[forces] done -> %s' % FORCE_JSONL, flush=True)

# -------------------------------------------------------------------------- post
elif a.stage == 'post':
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import write_FORCE_SETS

    dj = json.load(open(os.path.join(VDIR, 'disps.json')))
    nd = len(dj['disps'])
    F = {}
    for ln in open(FORCE_JSONL):
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            F[r['k']] = r
    if len(F) != nd:
        raise SystemExit('forces 不完整: %d/%d' % (len(F), nd))

    # 关键：必须用与 gen 完全相同的 (cell, smat, primitive_matrix, symprec) 重建对象。
    # phonopy.load(phonopy_disp.yaml) 会用默认 symprec=1e-5 重算对称性，弛豫后的胞在该
    # 精度下退化成 P1，导致 "Input forces are not enough to calculate force constants"。
    st = json.load(open(os.path.join(a.root, 'eos', a.phase, a.tag, 'poscar.json')))
    uc = PhonopyAtoms(symbols=st['symbols'], cell=np.array(st['cell']),
                      scaled_positions=np.array(st['scaled_positions']))
    smat = np.array(dj['smat'])
    ph = Phonopy(uc, supercell_matrix=smat, primitive_matrix=np.eye(3), symprec=a.symprec)
    try:
        ph.generate_displacements(distance=dj['distance'], is_plusminus='auto',
                                  symprec=a.symprec)
    except TypeError:
        ph.generate_displacements(distance=dj['distance'], is_plusminus='auto')
    nat_sc = len(ph.supercell)
    dvec = np.array(ph.displacements)
    if len(dvec) != nd:
        raise SystemExit('位移数不匹配: rebuild %d vs forces %d' % (len(dvec), nd))
    for k in range(nd):
        assert abs(dvec[k][0] - dj['disps'][k]['i']) < 1e-9
        assert np.allclose(dvec[k][1:4], dj['disps'][k]['d'], atol=1e-9)
    nprim = len(ph.primitive)
    ds = ph.dataset
    k = 0
    for key in ('first_atoms', 'second_atoms'):
        for ent in ds.get(key, []):
            ent['forces'] = np.array(F[k]['forces'], dtype=float)
            k += 1
    if k != nd:
        raise SystemExit('dataset 位移数 %d != forces %d' % (k, nd))
    write_FORCE_SETS(ds, filename=os.path.join(VDIR, 'FORCE_SETS'))
    ph.produce_force_constants()
    ph.run_mesh([MESH] * 3, with_eigenvectors=False, is_mesh_symmetry=True)
    fr = np.array(ph.get_mesh_dict()['frequencies'])
    nq, nb = fr.shape
    n_imag = int((fr < -1e-3).sum())
    fmin = float(fr.min())
    fu = fr[fr > 1e-3]
    zpe_mesh = float((fu * THZ_TO_MEV / 2.0).sum() / nq / nprim)
    half = fu * THZ_TO_MEV / 2.0
    bands = []
    tot = fu.size
    for lo, hi in BANDS:
        m = (fu >= lo) & (fu < hi)
        bands.append(dict(lo=lo, hi=None if hi > 1e8 else hi, n_modes=int(m.sum()),
                          frac=float(m.sum() / tot),
                          zpe=float(half[m].sum() / nq / nprim)))
    grid = np.arange(0, 40, 0.1)
    dos, _ = np.histogram(fu, bins=np.append(grid, grid[-1] + 0.1))
    dos = dos / nq / nprim / 0.1

    try:
        ph.run_thermal_properties(tstep=25, tmax=600, tmin=0)
    except TypeError:
        ph.run_thermal_properties(t_step=25, t_max=600, t_min=0)
    tp = ph.get_thermal_properties_dict()
    T = np.array(tp.get('temperatures'))
    Fk = np.array(tp.get('free_energy', tp.get('free_energies')))
    S = np.array(tp.get('entropy', tp.get('entropies')))
    Cv = np.array(tp.get('heat_capacity', tp.get('heat_capacities')))
    zpe_ph = float(Fk[0] / KJMOL_TO_EV / nprim * 1000.0)   # meV/atom

    out = dict(phase=a.phase, tag=a.tag, natoms_unit=nprim, natoms_supercell=nat_sc,
               mesh=[MESH] * 3,
               nq=nq, nbands=nb, n_imag=n_imag, fmin_THz=fmin, fmax_THz=float(fr.max()),
               zpe_meV_per_atom=zpe_mesh, zpe_phonopy_meV_per_atom=zpe_ph,
               bands=bands, dos_grid=grid.tolist(), dos=dos.tolist(),
               T=T.tolist(),
               F_vib_eV_per_atom=(Fk / KJMOL_TO_EV / nprim).tolist(),
               S_eV_per_K_per_atom=(S / KJMOL_TO_EV / 1000.0 / nprim).tolist(),
               Cv_eV_per_K_per_atom=(Cv / KJMOL_TO_EV / 1000.0 / nprim).tolist())
    json.dump(out, open(os.path.join(VDIR, 'ph.json'), 'w'), indent=1)
    print('[post] %s/%s ZPE=%.3f meV/at  n_imag=%d  fmin=%+.3f THz  fmax=%.1f THz'
          % (a.phase, a.tag, zpe_mesh, n_imag, fmin, float(fr.max())), flush=True)
