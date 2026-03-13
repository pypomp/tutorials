"""
gridsearch_dacca_mif.py for obtaining the appropriate parameters for model fitting.

grid searching the hyperparameters (random walk and cooling rate) for mif2 on dacca model. 
"""
import os 
import itertools
import copy
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
# J for mif2 and train
NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1] 
# M for mif2 
NFITR = (2, 20, 100, 100)[RUN_LEVEL - 1]
# nreps for mif2 and train
NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1] 
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1] 
# nreps for eval (pfilter and mop)
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

RW_SD = jnp.array([0.02] * 2 + [0.0] + [0.02] * 18)
RW_SD_INIT = jnp.array([0.0] * 21)
COOLING_RATE = 0.5

BASE_SIGMAS = {
    "gamma": 0.02,
    "m": 0.02,
    "rho": 0.02,
    "epsilon": 0.02,
    "omega": 0.02,
    "c": 0.02,
    "beta_trend": 0.02,
    "sigma": 0.02,
    "tau": 0.02,
    **{f"bs{i}": 0.02 for i in range(1, 7)},
    **{f"omegas{i}": 0.02 for i in range(1, 7)},
}

def make_rw_sd_base(scale: float = 1.0, init_names=None):
    if init_names is None:
        init_names = []

    sigmas = {k: float(v) * float(scale) for k, v in BASE_SIGMAS.items()}

    return pp.RWSigma(
        sigmas=sigmas,
        init_names=init_names,
    )

A_list = [0.5, 0.7, 0.85, 0.9, 0.95]
RW_SD_list = [0.3, 0.5, 1.0, 1.5, 2.0]

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

def bench_dacca_mif(
    J: int, 
    np_eval: int,
    M: int, 
    rw_sd: RWSigma, 
    a: float, 
    reps: int, 
    reps_eval: int,
    seed: int,
    mem_profile: bool = False
) -> dict:
    """
    Time is recorded via mif function internally;
    Reps is handled by the `_mif_loop` function;
    Plug `_mif_loop` into memory_profiling to get peak memory.
    """
    devices = jax.devices()
    platform = devices[0].platform if devices else jax.default_backend()

    key = jax.random.key(seed)
    dacca = pp.dacca()
    key, subkey = jax.random.split(key)

    profile_tag = f"dacca_mif_{platform}"
    
    params_box = {k: [v * 0.5, v * 1.5] for k, v in dacca.theta[0].items()}
    params_box["rho"] = [0.0, 0.0]
    key, subkey = jax.random.split(key)
    initial_params_list = pp.Pomp.sample_params(params_box, NREPS_FITR, key=subkey)

    def _mif_loop(subkey):
        ll_vec = jnp.zeros((reps,))
        time_vec = jnp.zeros((reps,))
        for i in range(reps):
            subkey, key_mif, key_eval = jax.random.split(subkey, 3)
            dacca.mif(theta=initial_params_list, J=J, M=M, rw_sd = rw_sd, a=a, key=key_mif)
            time = dacca.results_history[-1].execution_time
            time_vec = time_vec.at[i].set(time)
            dacca.pfilter(J=np_eval, reps = reps_eval, key = key_eval)
            ll_eval = float(dacca.results_history[-1].to_dataframe()["logLik"].max())
            ll_vec = ll_vec.at[i].set(ll_eval)
        return ll_vec, time_vec, subkey
    
    if not mem_profile:
        ll_vec, time_vec, subkey = _mif_loop(subkey)
    #if mem_profile:
        #dacca, out = memory_profiling(tag = profile_tag, pomp_obj = dacca, fn = _mif_loop, subkey = subkey)
        #ll_vec = out[0]
        #time_vec = out[1]

    ll_stats = _summarize_ll(ll_vec)
    t_stats = _summarize_time(time_vec)
    return dict(
        method="mif",
        backend=platform,
        particles=J,
        logmeanexp_ll=ll_stats["logmeanexp"],
        logmeanexp_se=ll_stats["logmeanexp_se"],
        n_reps=ll_stats["n_reps"],
        n_iter=M,
        run_time=t_stats["mean_time"],
        se_run_time=t_stats["se_time"],
        # peak_mem_mib=float("nan"), 
        obj=dacca,
    )


devices = jax.devices()
platform = devices[0].platform if devices else jax.default_backend()
print(f"[info] Running Dacca benchmark on backend={devices}")

results = []

grid = list(itertools.product(A_list, RW_SD_list))
for idx, (a_i, rw_sd_i) in enumerate(grid):
    seed_i = SEED + idx
    RW_SD_i = make_rw_sd_base(scale=rw_sd_i)
    res_mif = bench_dacca_mif(J=NP_FITR, np_eval=NP_EVAL, M=NFITR, rw_sd=RW_SD_i, a = a_i, reps=NREPS_FITR, reps_eval=NREPS_EVAL, seed=seed_i)
    
    res_mif["config"] = {
        "a": float(a_i),
        "rw_sd": float(rw_sd_i),
        "seed": int(seed_i),
        "RUN_LEVEL": int(RUN_LEVEL),
        "NP_FITR": int(NP_FITR),
        "NFITR": int(NFITR),
        "NREPS_FITR": int(NREPS_FITR),
        "NP_EVAL": int(NP_EVAL),
        "NREPS_EVAL": int(NREPS_EVAL),
    }

    results.append(res_mif)
    print("[mif] ", {k: v for k, v in res_mif.items() if k != "obj"})


