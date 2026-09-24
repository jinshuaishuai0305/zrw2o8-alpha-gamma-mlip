import warnings, glob, os, json
warnings.filterwarnings('ignore')
from ase.io import read
from collections import Counter
files = sorted(glob.glob('/home/jss/share/AlZrW2O8_MACE/*.xyz')) + \
        sorted(glob.glob('/home/jss/share/AlZrW2O8_MACE/mace_train/*.xyz'))
for f in files:
    try:
        idx = read(f, index=':')
    except Exception as e:
        print(f"{os.path.basename(f):35s} READ FAIL {e}"); continue
    c = Counter(len(a) for a in idx)
    e44 = [a.get_potential_energy()/44 for a in idx if len(a)==44 and a.calc is not None and a.calc.results.get('energy') is not None]
    e132= [a.get_potential_energy()/132 for a in idx if len(a)==132 and a.calc is not None and a.calc.results.get('energy') is not None]
    print(f"{os.path.basename(f):35s} n={len(idx):5d}  sizes={dict(sorted(c.items()))}")
    if e44:  print(f"      44-atom E/at: min={min(e44):.8f} max={max(e44):.8f} eV  ({len(e44)} frames)")
    if e132: print(f"     132-atom E/at: min={min(e132):.8f} max={max(e132):.8f} eV  ({len(e132)} frames)")
