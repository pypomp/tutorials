"""
dacca_benchmark.py for ms table

benchmarking dacca Pomp model on multiple inference methods (pfilter, MOP, mif2, train, ifad) under both CPU and GPU
Here, we construct model, run methods with multiple reps with fixed PRNG seeds, record timing, loglik summary, etc.
we collect results in a dict, and save the full object for QMD file
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
# J for mif2 and train
NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1] 
# nreps for mif2 and train
NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1]
# number of parallel starting points per algorithm call
N_STARTS = (2, 10, 20, 20)[RUN_LEVEL - 1]
# M for ifad mif2 iterations
NFITR_IFAD = (1, 10, 50, 50)[RUN_LEVEL - 1] 
# M for ifad train iterations
NTRAIN_IFAD = (1, 10, 20, 25)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1] 
# nreps for eval (pfilter and mop)
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

#TRAIN_ITERS = 50
TRAIN_OPTIMIZER = "Adam"
TRAIN_SCALE = False
TRAIN_LS = False
TRAIN_N_MONITORS = NP_EVAL


RW_SD = jnp.array([0.02] * 2 + [0.0] + [0.02] * 18)
RW_SD_INIT = jnp.array([0.0] * 21)
COOLING_RATE = 0.5
MOP_ALPHA = 0.97
ETA = 0.1

RW_SD = pp.RWSigma(
        sigmas={
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
        },
        init_names=[],
    )
# cashing the results 
# default parameters for each method

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

def bench_dacca_ifad(
    J: int, 
    M1: int,
    M2:int, 
    rw_sd: RWSigma, 
    cooling: float, 
    optimizer: str, 
    scale: bool, 
    ls: bool, 
    eta: float,
    np_eval: int,
    reps_eval: int,
    n_monitors: int,
    reps: int,
    seed: int,
    mem_profile: bool = False
) -> dict:
    """
    IFAD = run mif first (M1 iters), then train (M2 iters),
    per replicate.
    We add the two execution times, and extract the final post-train logLik.
    Time is recorded via train and mif functions internally;
    Reps is handled by the `_ifad_loop` function;
    Plug `_ifad_loop` into memory_profiling to get peak memory.
    """
    devices = jax.devices()
    platform = devices[0].platform if devices else jax.default_backend()

    key = jax.random.key(seed)
    dacca = pp.dacca()
    key, subkey = jax.random.split(key)

    profile_tag = f"dacca_ifad_{platform}"

    params_box = {k: [v * 0.5, v * 1.5] for k, v in dacca.theta[0].items()}
    params_box["rho"] = [0.0, 0.0]
    eta_dict = {param: eta for param in dacca.canonical_param_names}

    def ifad_loop(subkey):
        ll_vec   = jnp.zeros((reps,))
        time_vec = jnp.zeros((reps,))
        for i in range(reps):
            subkey, key_starts, subkey_mif, subkey_train, subkey_eval = jax.random.split(subkey, 5)
            initial_params_list = pp.Pomp.sample_params(params_box, N_STARTS, key=key_starts)

            dacca.mif(theta=initial_params_list, J=J, M=M1, rw_sd=rw_sd, a=cooling, key=subkey_mif)
            t_mif = dacca.results_history[-1].execution_time
            # replace the data with top n thetas
            # dacca.prune(n=10, refill=True)
            dacca.train(J=J, M=M2, optimizer=optimizer, eta=eta_dict, scale=scale, ls=ls, n_monitors=n_monitors, key=subkey_train)
            t_train = dacca.results_history[-1].execution_time
            dacca.pfilter(J=np_eval, reps=reps_eval, key=subkey_eval)
            ll_last = dacca.results_history[-1].to_dataframe()["logLik"].max()
            ll_vec   = ll_vec.at[i].set(ll_last)
            time_vec = time_vec.at[i].set(t_mif + t_train)

        return ll_vec, time_vec, subkey
    
    if not mem_profile:
        ll_vec, time_vec, subkey = ifad_loop(subkey)
    #if mem_profile:
        #dacca, out = memory_profiling(tag = profile_tag, pomp_obj = dacca, fn = ifad_loop, subkey = subkey)
        #ll_vec = out[0]
        #time_vec = out[1]

    ll_stats = _summarize_ll(ll_vec)
    t_stats = _summarize_time(time_vec)
    return dict(
        method="ifad",
        backend=platform,
        particles=J,
        logmeanexp_ll=ll_stats["logmeanexp"],
        logmeanexp_se=ll_stats["logmeanexp_se"],
        n_reps=ll_stats["n_reps"],
        n_mif_iter=M1,
        n_train_iter=M2,
        run_time=t_stats["mean_time"],
        se_run_time=t_stats["se_time"],
        # peak_mem_mib=float("nan"),
        obj=dacca,
    )

devices = jax.devices()
platform = devices[0].platform if devices else jax.default_backend()
print(f"[info] Running Dacca benchmark on backend={devices}")

results = []

res_ifad = bench_dacca_ifad(J=NP_FITR, M1=NFITR_IFAD, M2=NTRAIN_IFAD, rw_sd=RW_SD, eta=ETA, cooling=COOLING_RATE, optimizer=TRAIN_OPTIMIZER, scale=TRAIN_SCALE, ls=TRAIN_LS, n_monitors=TRAIN_N_MONITORS, np_eval=NP_EVAL, reps_eval=NREPS_EVAL, reps=NREPS_FITR, seed=SEED)
results.append(res_ifad)
print("[ifad]    ", {k:v for k,v in res_ifad.items() if k!="obj"})

# save results: save all the disctionaries except the 'obj' fields, and save one final object recording all the elements
bench_summary_dict = []
for r in results:
    r_2 = {k:v for k,v in r.items() if k != "obj"}
    bench_summary_dict.append(r_2)

final_result = {
    "backend": platform,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
    "settings": {
        "SEED": SEED,
        "NP_FITR": NP_FITR,
        "FITR_REPS": NREPS_FITR,
        "IFAD_MIF_ITERS": NFITR_IFAD,
        "IFAD_TRAIN_ITERS": NFITR_IFAD,
    },
    "benchmarks": bench_summary_dict
}

def make_filename(prefix="dacca_benchmark", run_level=1, backend="cpu"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_L{run_level}_{backend}_{ts}.pkl"

filename = make_filename(
    prefix="dacca_benchmark",
    run_level=RUN_LEVEL,
    backend=platform,
)

with open(filename, "wb") as f:
    pickle.dump(final_result, f)

print(f"[info] Saved benchmark result to {filename}")


