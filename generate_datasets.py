"""
Instance generator for UPMSP-SDFS, following the data-generation procedure of the
paper (SWEVO-D-26-00938, Section 5.1), which in turn follows Liaw et al. [52] and
Vallada & Ruiz [53]:

  processing time  p_{i,f}     ~ U[10, 100]   (integer; per family x machine; unrelated machines)
  job weight       w_f         ~ U[1, 10]     (integer; per family)
  setup time       sigma_{f,g} ~ U[1, 49]     (integer; f != g; sigma_{f,f}=0; sequence-dependent, asymmetric)
  due date         d_f         ~ U[ MP(1-r-R/2), MP(1-r+R/2) ]
                   MP = mean processing time (makespan estimate
                        = sum_f demand_f * mean_i p_{i,f} / N_M),
                   r  = average tardiness factor,
                   R  = relative range of due dates.

The output Excel files have the SAME format as datasets/D*/Example_*.xlsx:
  Jobs            : Job Family | Demand | Weight | Due Date            (one row per family)
  Machines        : Machine    | Initial Setup                        (one row per machine)
  ProcessingTimes : Job Family / Machine | M1 ... M{N_M}              (one row per family)
  SetupTimes      : From \\ To | <families...>                        (one row per family)

By default it writes to datasets_generated/ so the released datasets/ are NOT
overwritten. Use --out datasets to regenerate in place (NOT recommended: it would
change the instances all experiments are run on).

Usage:
  python generate_datasets.py                 # all datasets -> datasets_generated/
  python generate_datasets.py --out mydata    # custom output folder
  python generate_datasets.py --only D2 D3    # subset
  python generate_datasets.py --seed 12345
"""
import os
import string
import argparse
import itertools

import numpy as np
import pandas as pd

# ----------------------------- generation parameters --------------------------
PROC_LOW, PROC_HIGH = 10, 100        # U[10, 100]
WEIGHT_LOW, WEIGHT_HIGH = 1, 10      # U[1, 10]
SETUP_LOW, SETUP_HIGH = 1, 49        # U[1, 49] (off-diagonal)

# (name, n_families, n_machines, total_jobs, n_instances) -- matches the paper:
# D1 = 20 instances (1 train + 19 validation); D2-D4 = 5 instances each.
DATASETS = [
    ('D1', 10, 10, 500, 20),
    ('D2', 10, 20, 1000, 5),
    ('D3', 15, 20, 1000, 5),
    ('D4', 20, 20, 1000, 5),
]
OUTPUT_DIR = 'datasets'
BASE_SEED = 3


# --------------------------------- helpers ------------------------------------
def family_labels(n):
    """A, B, ..., Z, then AA, AB, ... for n > 26 (single letters cover up to 26)."""
    if n <= 26:
        return list(string.ascii_uppercase[:n])
    labels = list(string.ascii_uppercase)
    for combo in itertools.product(string.ascii_uppercase, repeat=2):
        labels.append(''.join(combo))
        if len(labels) >= n:
            break
    return labels[:n]


def gen_demand(total, n_families, rng, spread=1.0 / 3.0):
    """Split `total` jobs among the families: each family ~ U[(1-s)mu, (1+s)mu]
    (mu = total/n_families), then normalized and integer-adjusted to sum EXACTLY
    to `total` (every family gets >= 1)."""
    raw = rng.uniform(1.0 - spread, 1.0 + spread, size=n_families)
    d = np.maximum(1, np.round(total * raw / raw.sum())).astype(int)
    diff = total - int(d.sum())
    order = rng.permutation(n_families)
    i = 0
    while diff != 0:
        k = order[i % n_families]
        if diff > 0:
            d[k] += 1; diff -= 1
        elif d[k] > 1:
            d[k] -= 1; diff += 1
        i += 1
    return d


def gen_instance(n_families, n_machines, total_jobs, rng, r=0.1, R=0.15):
    """Return the four sheets (as DataFrames) of one instance."""
    fams = family_labels(n_families)
    demand = gen_demand(total_jobs, n_families, rng)                       # per family
    proc = {f: rng.integers(PROC_LOW, PROC_HIGH + 1, size=n_machines) for f in fams}   # U[10,100]
    weight = {f: int(rng.integers(WEIGHT_LOW, WEIGHT_HIGH + 1)) for f in fams}         # U[1,10]
    setup = {f: {g: (0 if f == g else int(rng.integers(SETUP_LOW, SETUP_HIGH + 1)))
                 for g in fams} for f in fams}                            # U[1,49], diag 0
    init_setup = [str(rng.choice(fams)) for _ in range(n_machines)]       # random family per machine

    # due dates ~ U[MP(1-r-R/2), MP(1-r+R/2)], MP = makespan estimate
    MP = sum(int(demand[i]) * float(np.mean(proc[f])) for i, f in enumerate(fams)) / n_machines
    low, high = MP * (1.0 - r - R / 2.0), MP * (1.0 - r + R / 2.0)
    due = {f: max(1, int(round(rng.uniform(low, high)))) for f in fams}

    jobs_df = pd.DataFrame({
        'Job Family': fams,
        'Demand': [int(demand[i]) for i in range(n_families)],
        'Weight': [weight[f] for f in fams],
        'Due Date': [due[f] for f in fams],
    })
    machines_df = pd.DataFrame({
        'Machine': [f'M{i + 1}' for i in range(n_machines)],
        'Initial Setup': init_setup,
    })
    proc_df = pd.DataFrame({'Job Family / Machine': fams})
    for i in range(n_machines):
        proc_df[f'M{i + 1}'] = [int(proc[f][i]) for f in fams]
    setup_df = pd.DataFrame({'From \\ To': fams})
    for g in fams:
        setup_df[g] = [setup[f][g] for f in fams]

    return {'Jobs': jobs_df, 'Machines': machines_df,
            'ProcessingTimes': proc_df, 'SetupTimes': setup_df}


def write_instance(path, sheets):
    with pd.ExcelWriter(path) as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)


def generate(output_dir=OUTPUT_DIR, datasets=DATASETS, base_seed=BASE_SEED, only=None):
    for di, (name, n_fam, n_mach, total, n_inst) in enumerate(datasets):
        if only and name not in only:
            continue
        out = os.path.join(output_dir, name)
        os.makedirs(out, exist_ok=True)
        for k in range(1, n_inst + 1):
            rng = np.random.default_rng(base_seed + di * 100000 + k)
            sheets = gen_instance(n_fam, n_mach, total, rng)
            write_instance(os.path.join(out, f'Example_{k}.xlsx'), sheets)
        # dataset_info.xlsx (summary), matching the folder layout
        pd.DataFrame([{'Dataset': name, 'Job Families': n_fam,
                       'Machines': n_mach, 'Total Demand': total}]).to_excel(
            os.path.join(out, 'dataset_info.xlsx'), sheet_name='Sheet1', index=False)
        print(f"[{name}] {n_inst} instances  ({n_fam} families, {n_mach} machines, "
              f"{total} jobs)  -> {out}")


def main():
    ap = argparse.ArgumentParser(description='Generate UPMSP-SDFS instances (paper Section 5.1 format).')
    ap.add_argument('--out', default=OUTPUT_DIR, help='output folder (default: datasets_generated)')
    ap.add_argument('--seed', type=int, default=BASE_SEED, help='base random seed')
    ap.add_argument('--only', nargs='*', default=None, help='subset of dataset names, e.g. --only D2 D3')
    args = ap.parse_args()
    print(f"Generating datasets into '{args.out}' (seed {args.seed})")
    print(f"  p~U[{PROC_LOW},{PROC_HIGH}], w~U[{WEIGHT_LOW},{WEIGHT_HIGH}], "
          f"sigma~U[{SETUP_LOW},{SETUP_HIGH}]")
    generate(output_dir=args.out, base_seed=args.seed, only=args.only)
    print("Done.")


if __name__ == '__main__':
    main()
