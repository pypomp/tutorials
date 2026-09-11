"""Model 001b panel fit across UK measles units."""

import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
import pypomp as pp

from pymeasles.provenance import parse_fit_args, save_experiment_artifacts
from pymeasles.units import UNITS

args = parse_fit_args()
RUN_LEVEL = args.level
SLURM_ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

now = datetime.now()
print(f"DATE: {now.date()}  TIME: {now.time()}")
print(f"pypomp version: {version('pypomp')}")
print(f"jax version: {version('jax')}")
print(f"Devices: {jax.devices()}")
print(f"Running at level {RUN_LEVEL} (array task: {SLURM_ARRAY_TASK_ID})")

MAIN_SEED = 631409 + max(0, SLURM_ARRAY_TASK_ID)
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

# Hyperparameter schedules by level
NP_FITR = (2, 500, 1000, 10000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 200)[RUN_LEVEL - 1]
NREPS_FITR = args.reps if args.reps is not None else (2, 3, 6, 12)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

if args.units is not None:
    N_UNITS = args.units
else:
    N_UNITS = (5, 200, 200, 200)[RUN_LEVEL - 1]

active_units = UNITS[:N_UNITS]
print(f"Using {len(active_units)} units")

# -- SEARCH HYPERPARAMETERS & PARAMETER BOX -----------------------------------
DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 12
COOLING_RATE = 0.5

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
).geometric_cooling(COOLING_RATE)

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

# -- INITIAL PARAMETER SAMPLING -----------------------------------------------
key, subkey = jax.random.split(key)
initial_params = pp.PanelPomp.sample_params(
    measles_box,
    n=NREPS_FITR,
    units=active_units,
    key=subkey,
    shared_names=[],
)

# -- BUILD PANEL POMP OBJECT --------------------------------------------------
print("Building panel POMP object...")
panel_measles_obj = pp.models.UKMeasles.panel_pomp(
    units=active_units,
    theta=initial_params,
    model="001b",
    clean=True,
)

# -- MIF ESTIMATION -----------------------------------------------------------
print("Starting MIF search...")
key, subkey = jax.random.split(key)
panel_measles_obj.mif(
    J=NP_FITR,
    M=NFITR,
    rw_sd=RW_SD,
    key=subkey,
)

print("MIF completed. First results preview:")
print(panel_measles_obj.results(ignore_nan=False).head())

# -- PARTICLE FILTER EVALUATION -----------------------------------------------
print("Running evaluation particle filter 1...")
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
res_eval1 = panel_measles_obj.results()
eval1_lls = res_eval1.groupby("theta_idx")["shared logLik"].first().tolist()
print(
    f"Evaluation particle filter 1 log-likelihoods ({len(eval1_lls)} reps): {eval1_lls}"
)

# -- MIX AND MATCH & PRUNE ----------------------------------------------------
print("Running mix_and_match and pruning...")
panel_measles_obj.mix_and_match()
panel_measles_obj.prune(n=1, refill=False)

print("Running evaluation particle filter 2 (CLL=True)...")
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, CLL=True)

print(f"Total time elapsed: {panel_measles_obj.time()}")

# -- SAVE RESULTS & PROVENANCE ------------------------------------------------
ts = now.strftime("%Y-%m-%d_%H%M%S")
tag = f"_{args.name}" if args.name else ""
out_dir = (
    Path(__file__).resolve().parent
    / "results"
    / f"{ts}_u{len(active_units)}_L{RUN_LEVEL}{tag}"
)

run_parameters = {
    "model": "001b",
    "run_level": RUN_LEVEL,
    "n_units": len(active_units),
    "np_fitr": NP_FITR,
    "nfitr": NFITR,
    "nreps_fitr": NREPS_FITR,
    "np_eval": NP_EVAL,
    "nreps_eval": NREPS_EVAL,
    "cooling_rate": COOLING_RATE,
    "default_sd": DEFAULT_SD,
    "default_ivp_sd": DEFAULT_IVP_SD,
    "seed": MAIN_SEED,
    "measles_box": {k: list(v) for k, v in measles_box.items()},
    "active_units": active_units,
}

save_experiment_artifacts(
    panel_pomp=panel_measles_obj,
    out_dir=out_dir,
    script_path=__file__,
    metadata=run_parameters,
    save_model=args.save_model,
    task_id=SLURM_ARRAY_TASK_ID,
)


panel_measles_obj.print_summary()
print("Done!")
