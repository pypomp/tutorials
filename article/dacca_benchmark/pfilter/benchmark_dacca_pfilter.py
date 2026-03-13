"""
benchmarking the pfilter method for dacca model;
"""
import os 

# Set JAX platform before importing JAX
if os.environ.get("USE_CPU", "false").lower() == "true":
    os.environ["JAX_PLATFORMS"] = "cpu"

import time, math, pickle
import numpy as np
import jax
import datetime
import jax.numpy as jnp
import pypomp as pp
from pypomp.RWSigma_class import RWSigma

SEED = 631409
J = 1000

# add 3 levels : dubugging, fine on laptop, cluster
RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
# J for eval (pfilter and mop)
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1] 
# nreps for eval (pfilter and mop)
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

def logmeanexp(x, ignore_nan=True):
    x = np.asarray(x, dtype=float)

    if ignore_nan:
        x = x[~np.isnan(x)]
    if x.size == 0:
        return -np.inf
    
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x - m))))


def logmeanexp_se(x, ignore_nan=True):
    """
    Monte Carlo standard error for log-mean-exp of x
    """
    x = np.asarray(x, dtype=float)

    if ignore_nan:
        x = x[~np.isnan(x)]

    n = x.size
    if n <= 1:
        return np.nan

    m = np.max(x)
    L = np.exp(x - m)

    mean_L = np.mean(L)
    var_L  = np.var(L, ddof=1) / n
    se_L   = np.sqrt(var_L)

    return float(se_L / mean_L)

def set_device(device:str):
    """
    device in {"cpu", "gpu"}
    - cpu: force JAX CPU
    - others: let JAX use GPU if available (CUDA/Metal/...)
    Note: we should call this before the first jax.devices or any JAX computation.
    """
    if device.lower() == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        if "JAX_PLATFORMS" in os.environ:
            del os.environ["JAX_PLATFORMS"]  

def method_profiling(tag: str, pomp_obj: pp.Pomp, fn, *args, **kwargs):
    """
    Here we call the jax.profilter.trace() to profile each methods;
    we also record the execution time;
    "tag" is the name of that profile trace;
    returns the output of fn(*args, **kwargs) and elapsed time;
    later we can get peak memory.
    """
    t0 = time.time()
    logdir = f"/tmp/jax_profile/{tag}"
    with jax.profiler.trace(logdir):
        out = fn(*args, **kwargs)
    runtime = time.time() - t0
    return pomp_obj, out, runtime

def memory_profiling(tag: str, pomp_obj: pp.Pomp, fn, *args, **kwargs):
    """
    we don't record the time here, just the memory.
    """
    logdir = f"/tmp/jax_profile/{tag}"
    with jax.profiler.trace(logdir):
        out = fn(*args, **kwargs)
    return pomp_obj, out

def _summarize_ll(ll_array: np.ndarray | jnp.ndarray):
    """
    Given a vector of loglik (not None), return mean/se and n.
    """
    if ll_array is None:
        return dict(
            logmeanexp = float("nan"),
            logmeanexp_se = float("nan"),
            n_reps = 0,
        )
    ll_np = np.asarray(ll_array) # convert the ll_array to numpy array
    n = ll_np.size
    if n == 0:
        return dict(
            logmeanexp = float("nan"),
            logmeanexp_se = float("nan"),
            n_reps = 0,
        )
    mean_ll = float(logmeanexp(ll_np))
    se_ll = float(logmeanexp_se(ll_np)) if n > 1 else 0.0
    return dict(
        logmeanexp = mean_ll,
        logmeanexp_se = se_ll,
        n_reps = n,
    )

def _summarize_time(time_array: np.ndarray | jnp.ndarray):
    """
    Given a vector of times (not None), return mean/se and n.
    """
    if time_array is None:
        return dict(
            mean_time = float("nan"),
            se_time = float("nan"),
            n_reps = 0,
        )
    time_np = np.asarray(time_array) # convert the time_array to numpy array
    n = time_np.size
    if n == 0:
        return dict(
            mean_time = float("nan"),
            se_time = float("nan"),
            n_reps = 0,
        )
    mean_time = float(np.mean(time_np))
    se_time = float(np.std(time_np, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return dict(
        mean_time = mean_time,
        se_time = se_time,
        n_reps = n,
    )

# for pfilter and mop, the functions don't record execution time internally, so we use method_profiling to get the time and peak memory.;

def bench_dacca_pfilter(
    J: int, 
    reps: int, 
    seed: int, 
    mem_profile: bool = False
) -> dict:
    """
    Run time is recorded via pfilter internally;
    Reps is handled inside the pfilter function internally.
    Plug the original pfilter into method_profiling to get peak memory.
    """
    devices = jax.devices() 
    platform = devices[0].platform if devices else jax.default_backend()

    key = jax.random.key(seed)
    dacca = pp.dacca() 
    key, subkey = jax.random.split(key)

    profile_tag = f"dacca_pfilter_{platform}"
    if not mem_profile:
        dacca.pfilter(J=J, reps=reps, key=subkey)
    #if mem_profile:
        #dacca, out = memory_profiling(tag = profile_tag, pomp_obj = dacca, fn = dacca.pfilter, J=J, reps=reps, key=subkey)
    ll = dacca.results_history[-1].logLiks.squeeze().values
    stats = _summarize_ll(ll)
    return dict(
        method="pfilter",
        backend=platform,
        particles=J,
        logmeanexp_ll=stats["logmeanexp"],
        logmeanexp_se=stats["logmeanexp_se"],
        n_reps=stats["n_reps"],
        run_time=dacca.results_history[-1].execution_time/reps if reps>0 else float("nan"),
        #peak_mem_mib=float("nan"), # TODO parse from profiler
        obj=dacca,
    )

devices = jax.devices()
platform = devices[0].platform if devices else jax.default_backend()
print(f"[info] Running Dacca benchmark on backend={devices}")

results = []

res_pfilter = bench_dacca_pfilter(J=NP_EVAL, reps=NREPS_EVAL, seed=SEED)
results.append(res_pfilter)
print("[pfilter] ", {k:v for k,v in res_pfilter.items() if k!="obj"})   

bench_summary_dict = []
for r in results:
    r_2 = {k:v for k,v in r.items() if k != "obj"}
    bench_summary_dict.append(r_2)

final_result = {
    "backend": platform,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
    "settings": {
        "SEED": SEED,
        "NP_EVAL": NP_EVAL,
        "EVAL_REPS": NREPS_EVAL
    },
    "benchmarks": bench_summary_dict
}

def make_filename(prefix="dacca_benchmark", run_level=1, backend="cpu"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_L{run_level}_{backend}_{ts}.pkl"

filename = make_filename(
    prefix="pfilter_dacca_benchmark",
    run_level=RUN_LEVEL,
    backend=platform,
)

with open(filename, "wb") as f:
    pickle.dump(final_result, f)

print(f"[info] Saved benchmark result to {filename}")
