#!/usr/bin/env python3
"""
Micro-benchmark: is `scatter_add_` on CUDA float32 run-to-run reproducible when
it reduces the numbers MACE actually reduces?

models.py:314 does
    e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=1)
i.e. for a single structure a scatter_add_ over all atoms of the atomic
reference energies, each of magnitude |E0| ~ 1791 eV  (v6: -1791.041659268 eV).
The sum is ~-2.36e5 eV for gamma, whose float32 ulp is 0.03125 eV = 31.25 meV.

We reproduce exactly that reduction, and the identical one in float64, plus the
same reduction with randomised element order.
"""
import numpy as np, torch
E0 = -1791.041659268
print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0))
print(f"E0 per atom = {E0} eV\n")
rng = np.random.default_rng(0)
for n in (44, 132, 4096):
    src = torch.full((n, 1), E0, dtype=torch.float32, device="cuda")
    idx = torch.zeros(n, 1, dtype=torch.long, device="cuda")
    vals = [torch.zeros(1, 1, device="cuda").scatter_add_(0, idx, src).item()
            for _ in range(20)]
    v = np.array(vals)
    exact = E0 * n
    print(f"n={n:5d}  float32 scatter_add_ over identical E0:")
    print(f"        exact      = {exact:.9f} eV   (float32 stored: "
          f"{np.float32(exact)!r})")
    print(f"        min        = {v.min():.9f}")
    print(f"        max        = {v.max():.9f}")
    print(f"        range      = {(v.max()-v.min())*1000:.4f} meV total"
          f" = {(v.max()-v.min())*1000/n:.6f} meV/atom")
    print(f"        distinct   = {len(set(vals))}/{len(vals)}")
    print(f"        ulp(|sum|) = {np.spacing(np.float32(exact))*1000:.4f} meV")
    # float64 control
    src64 = src.double(); o64 = torch.zeros(1, 1, dtype=torch.float64, device="cuda")
    vals64 = [o64.clone().scatter_add_(0, idx, src64).item() for _ in range(20)]
    print(f"        float64 scatter_add_ range = "
          f"{(max(vals64)-min(vals64))*1000:.9f} meV  "
          f"distinct={len(set(vals64))}/20")
    # index order permuted -> different summation order
    perm = torch.randperm(n, device="cuda")
    vals_perm = []
    for _ in range(5):
        vals_perm.append(torch.zeros(1, 1, device="cuda")
                         .scatter_add_(0, idx, src[perm]).item())
    print(f"        permuted-order float32 range = "
          f"{(max(vals_perm)-min(vals_perm))*1000:.4f} meV total"
          f" = {(max(vals_perm)-min(vals_perm))*1000/n:.6f} meV/atom")
    print()

# also: the pairwise/atomic reduction used by the readout sum, models.py:392
print("sanity: explicit sequential float32 accumulation of n=132 E0 values")
acc = np.float32(0.0)
for _ in range(132):
    acc = np.float32(acc + np.float32(E0))
print(f"  sequential f32 = {float(acc):.6f} eV   vs exact {E0*132:.6f} eV  "
      f"-> error {(float(acc)-E0*132)*1000:.4f} meV total = "
      f"{(float(acc)-E0*132)*1000/132:.6f} meV/atom")
