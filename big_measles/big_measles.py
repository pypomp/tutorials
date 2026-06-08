"""
Model 001b fit using all 1422 units. All parameters are unit-specific. 10k particles for 200 iterations of IF2. Intended to be run as an array job. This job requires the A100 GPU partitions due to them having more RAM.
"""

import os
import importlib.util
import jax
import pickle
import time
import pypomp as pp
import numpy as np
from datetime import datetime
from importlib.metadata import version

now = datetime.now()
print("DATE: ", now.date())
print("TIME: ", now.time())
print("pypomp version:", version("pypomp"))
print("jax version:", version("jax"))

SLURM_ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

units_path = "units.py"
spec = importlib.util.spec_from_file_location("units", units_path)
if spec is None:
    raise ImportError(f"Could not load module spec from {units_path}")
units = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise ImportError(f"No loader found for module spec from {units_path}")
spec.loader.exec_module(units)
UNITS = units.UNITS

print(jax.devices())

MAIN_SEED = 631409 + SLURM_ARRAY_TASK_ID
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_FITR = (2, 500, 1000, 10000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 200)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 6, 12)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
N_UNITS = (5, len(UNITS), len(UNITS), len(UNITS))[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

UNITS = UNITS[:N_UNITS]

DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 12
RW_SD = pp.RWSigma(
    sigmas={
        "R0": DEFAULT_SD * 0.25,
        "sigma": DEFAULT_SD * 0.25,
        "gamma": DEFAULT_SD * 0.5,
        "iota": DEFAULT_SD,
        "rho": DEFAULT_SD * 0.5,
        "sigmaSE": DEFAULT_SD,
        "psi": DEFAULT_SD * 0.25,
        "cohort": DEFAULT_SD * 0.5,
        "amplitude": DEFAULT_SD * 0.5,
        "S_0": DEFAULT_IVP_SD,
        "E_0": DEFAULT_IVP_SD,
        "I_0": DEFAULT_IVP_SD,
        "R_0": DEFAULT_IVP_SD,
    },
    init_names=["S_0", "E_0", "I_0", "R_0"],
)
COOLING_RATE = 0.5


# --MAKE INITIAL PARAMETERS------------------------------------------------------

measles_box = {
    "R0": (10.0, 60.0),
    "sigma": (25.0, 100.0),
    "gamma": (25.0, 320.0),
    "iota": (0.004, 3.0),
    "rho": (0.1, 0.9),
    "sigmaSE": (0.04, 0.1),
    "psi": (0.05, 3.0),
    "cohort": (0.1, 0.7),
    "amplitude": (0.1, 0.6),
    "S_0": (0.01, 0.07),
    "E_0": (0.000004, 0.0001),
    "I_0": (0.000003, 0.001),
    "R_0": (0.9, 0.99),
}

key, subkey = jax.random.split(key)
dummy_initial_params_list = pp.Pomp.sample_params(measles_box, NREPS_FITR, key=subkey)

initial_params = pp.PanelPomp.sample_params(
    measles_box,
    n=NREPS_FITR,
    units=UNITS,
    key=subkey,
    shared_names=[],
)

# --MAKE POMPS------------------------------------------------------
print("Starting pomp creation")

pomp_dict = {
    unit: pp.models.UKMeasles.Pomp(
        unit=[unit],
        theta=dummy_initial_params_list,
        model="001b",
        clean=True,
    )
    for unit in UNITS
}

panel_measles_obj = pp.PanelPomp(
    Pomp_dict=pomp_dict,
    theta=initial_params,
)

# --MIF 1------------------------------------------------------
print("Starting MIF 1")

key, subkey = jax.random.split(key)
panel_measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(panel_measles_obj.results(ignore_nan=False))

# --PFILTER 1------------------------------------------------------

panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel_measles_obj.results(ignore_nan=False))

# --MIX AND MATCH------------------------------------------------------

panel_measles_obj.mix_and_match()
panel_measles_obj.prune(n=1, refill=False)

# --PFILTER 2------------------------------------------------------

panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel_measles_obj.results(ignore_nan=False))

print(panel_measles_obj.time())

# --SAVE RESULTS------------------------------------------------------
print("Starting save")
time0 = time.time()
with open(
    f"results/results_level_{RUN_LEVEL}_task_{SLURM_ARRAY_TASK_ID}.pkl", "wb"
) as f:
    pickle.dump(panel_measles_obj, f)
time1 = time.time()
print(f"Time taken to save results: {time1 - time0} seconds")

panel_measles_obj.print_summary()
