import os
import re
import glob

import pandas as pd

# =============================================================================
# Instance loader for the .xlsx datasets (datasets/D*/Example_*.xlsx).
#
# Each workbook has four sheets:
#   Jobs            : Job Family | Demand | Weight | Due Date   (one row per family)
#   Machines        : Machine | Initial Setup                   (one row per machine)
#   ProcessingTimes : Job Family / Machine | M1 ... Mn          (p_{i,f})
#   SetupTimes      : From \ To | <families...>                 (sigma_{f,g})
#
# The returned object exposes the same attribute names that PPO.train / PPO.test
# expect from a "dataset" object, so it is a drop-in replacement for the legacy
# `dataset.Dataset` class.
# =============================================================================


class InstanceData:
    def __init__(self, machines, job_types, demand, initial_setup, weighted_tardiness,
                 processing_times, due_dates, setup_times, name=''):
        self.MACHINES = machines
        self.JOB_TYPES = job_types
        self.DEMAND = demand
        self.INITIAL_SETUP = initial_setup
        self.WEIGHTED_TARDINESS = weighted_tardiness
        self.PROCESSING_TIMES = processing_times
        self.DUE_DATES = due_dates
        self.SETUP_TIMES = setup_times
        self.name = name

    @property
    def total_jobs(self):
        return sum(self.DEMAND.values())


def load_instance(path):
    """Load one .xlsx scheduling instance into an InstanceData object."""
    sheets = pd.read_excel(path, sheet_name=None)
    jobs_df = sheets['Jobs']
    machines_df = sheets['Machines']
    proc_df = sheets['ProcessingTimes'].set_index(sheets['ProcessingTimes'].columns[0])
    setup_df = sheets['SetupTimes'].set_index(sheets['SetupTimes'].columns[0])

    job_types = [str(f) for f in jobs_df['Job Family'].tolist()]

    return InstanceData(
        machines=len(machines_df),
        job_types=job_types,
        demand={str(f): int(d) for f, d in zip(jobs_df['Job Family'], jobs_df['Demand'])},
        initial_setup=[str(s) for s in machines_df['Initial Setup'].tolist()],
        weighted_tardiness={str(f): int(w) for f, w in zip(jobs_df['Job Family'], jobs_df['Weight'])},
        processing_times={str(f): [int(t) for t in proc_df.loc[f].tolist()] for f in job_types},
        due_dates={str(f): int(d) for f, d in zip(jobs_df['Job Family'], jobs_df['Due Date'])},
        setup_times={str(a): {str(b): int(setup_df.loc[a, b]) for b in job_types} for a in job_types},
        name=os.path.splitext(os.path.basename(path))[0],
    )


def list_instances(path):
    """Resolve a path into an ordered list of instance files.

    - file path  -> [path]
    - directory  -> sorted Example_*.xlsx inside it (natural numeric order)
    """
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, 'Example_*.xlsx'))
        if not files:
            files = glob.glob(os.path.join(path, '*.xlsx'))

        def _num(f):
            m = re.search(r'(\d+)', os.path.basename(f))
            return int(m.group(1)) if m else 0

        return sorted(files, key=_num)
    return [path]


def instance_from_dataset(test=True):
    """Build an InstanceData from the legacy `dataset.Dataset` class (fallback
    when no data path is configured). `test=True` uses the TEST_* fields,
    otherwise the training fields."""
    from dataset import Dataset
    if test:
        machines = Dataset.TEST_MACHINES
        job_types = [str(f) for f in Dataset.TEST_JOB_TYPES]
        init = list(getattr(Dataset, 'TEST_INITIAL_SETUP', [job_types[0]] * machines))
        return InstanceData(
            machines=machines, job_types=job_types,
            demand=dict(Dataset.TEST_DEMAND), initial_setup=init,
            weighted_tardiness=dict(Dataset.TEST_WEIGHTED_TARDINESS),
            processing_times=Dataset.TEST_PROCESSING_TIMES, due_dates=Dataset.TEST_DUE_DATES,
            setup_times=Dataset.TEST_SETUP_TIMES, name='dataset_TEST')
    machines = Dataset.MACHINES
    job_types = [str(f) for f in Dataset.JOB_TYPES]
    init = list(getattr(Dataset, 'TRAIN_INITIAL_SETUP', [job_types[0]] * machines))
    return InstanceData(
        machines=machines, job_types=job_types,
        demand=dict(Dataset.DEMAND), initial_setup=init,
        weighted_tardiness=dict(Dataset.WEIGHTED_TARDINESS),
        processing_times=Dataset.PROCESSING_TIMES, due_dates=Dataset.DUE_DATES,
        setup_times=Dataset.SETUP_TIMES, name='dataset_TRAIN')


def resolve_train_instance(path):
    """Return the training InstanceData: load from `path` if it exists, else fall
    back to the training fields of dataset.Dataset."""
    if path and os.path.exists(path):
        return load_instance(path)
    print(f"[data] TRAIN_DATA_PATH '{path}' not set/found -> falling back to dataset.py")
    return instance_from_dataset(test=False)


def resolve_test_instances(path):
    """Return a list of test InstanceData: every instance at `path` (file or
    folder) if it exists, else fall back to the TEST_* fields of dataset.Dataset."""
    if path and os.path.isdir(path):
        files = list_instances(path)
        if files:
            return [load_instance(p) for p in files]
    if path and os.path.isfile(path):
        return [load_instance(path)]
    print(f"[data] TEST_DATA_PATH '{path}' not set/found -> falling back to dataset.py")
    return [instance_from_dataset(test=True)]
