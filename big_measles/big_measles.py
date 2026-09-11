"""
Model 001b fit using all 1422 units. All parameters are unit-specific. 10k particles for 200 iterations of IF2. Intended to be run as an array job.
"""

import os
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

UNITS = pp.models.UKMeasles.units()

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

COOLING_RATE = 0.5
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
).geometric_cooling(a=COOLING_RATE)


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
initial_params = pp.PanelPomp.sample_params(
    measles_box,
    n=NREPS_FITR,
    units=UNITS,
    key=subkey,
    shared_names=[],
)

# --MAKE POMPS------------------------------------------------------
print("Starting pomp creation")

panel_measles_obj = pp.models.UKMeasles.panel_pomp(
    units=UNITS,
    theta=initial_params,
    model="001b",
    clean=True,
)

# --MIF 1------------------------------------------------------
print("Starting MIF 1")

key, subkey = jax.random.split(key)
panel_measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
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
