"""
dacca_benchmark.py for ms table

benchmarking dacca Pomp model on multiple inference methods (pfilter, MOP, mif2, train, ifad) under both CPU and GPU
Here, we construct model, run methods with multiple reps with fixed PRNG seeds, record timing, loglik summary, etc.
we collect results in a dict, and save the full object for QMD file

Need update the functions to match the new functions in corresponding method files. 
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
# M for mif2 
NFITR = (2, 20, 100, 100)[RUN_LEVEL - 1]
# M for train
NTRAIN = (2, 20, 40, 50)[RUN_LEVEL - 1] 
# nreps for mif2 and train
NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1] 
# J for eval (pfilter and mop)
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1] 
# nreps for eval (pfilter and mop)
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
# M for ifad mif2 iterations
NFITR_IFAD = (1, 10, 50, 50)[RUN_LEVEL - 1] 
# M for ifad train iterations
NTRAIN_IFAD = (1, 10, 20, 25)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

#MIF_ITERS = 50
#MIF_RW_SD = 0.05
#MIF_RW_SD_INIT = 1e-20
#MIF_COOLING = 0.9

#TRAIN_ITERS = 50
TRAIN_OPTIMIZER = "SGD"
TRAIN_SCALE = True
TRAIN_LS = True
TRAIN_N_MONITORS = 1

#IFAD_MIF_ITERS = 30
#IFAD_TRAIN_ITERS = 20
#IFAD_SIGMAS = 0.05
#IFAD_SIGMAS_INIT = 1e-20
#IFAD_COOLING = 0.9

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

# jax profiler: out of memory - remove it from the benchmarking for now
def memory_profiling(tag: str, pomp_obj: pp.Pomp, fn, *args, **kwargs):
    """
    we don't record the time here, just the memory.
    """
    logdir = f"/tmp/jax_profile/{tag}"
    with jax.profiler.trace(logdir):
        out = fn(*args, **kwargs)
    return pomp_obj, out

def time_profiling(tag: str, pomp_obj: pp.Pomp, fn, *args, **kwargs):
    """
    we don't record the memory here, just the time.
    """
    t0 = time.time()
    out = fn(*args, **kwargs)
    runtime = time.time() - t0
    return pomp_obj, runtime, out 

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

def bench_dacca_mop(
    J: int, 
    reps: int, 
    seed: int, 
    alpha: float,
    mem_profile: bool = False
) -> dict:
    """
    Run time is recorded via method_profiling function;
    Reps is handled by the `_mop_loop` function.
    Plug `_mop_loop` into method_profiling to get peak memory.
    """
    devices = jax.devices()
    platform = devices[0].platform if devices else jax.default_backend()

    key = jax.random.key(seed)
    dacca = pp.dacca()
    key, subkey = jax.random.split(key)

    profile_tag = f"dacca_mop_{platform}"

    def _mop_loop(subkey):
        ll_vec = jnp.zeros((reps,))
        for i in range(reps):
            subkey, subsubkey = jax.random.split(subkey)
            ll_est = dacca.mop(J=J, key=subsubkey, alpha=alpha)[0]
            ll_vec = ll_vec.at[i].set(ll_est)
        return ll_vec, subkey
    
    if not mem_profile:
        dacca, run_time, out = time_profiling(tag = profile_tag, pomp_obj = dacca, fn = _mop_loop, subkey = subkey)
    #if not mem_profile:
        #dacca, out, run_time = method_profiling(tag = profile_tag, pomp_obj = dacca, fn = _mop_loop, subkey = subkey)
    
    ll_vec = out[0]
    ll_stats = _summarize_ll(ll_vec)
    return dict(
        method="mop",
        backend=platform,
        particles=J,
        logmeanexp_ll=ll_stats["logmeanexp"],
        logmeanexp_se=ll_stats["logmeanexp_se"],
        n_reps=ll_stats["n_reps"],
        run_time=run_time/reps if reps>0 else float("nan"),
        # peak_mem_mib=float("nan"), 
        obj=dacca,
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

    def _mif_loop(subkey):
        ll_vec = jnp.zeros((reps,))
        time_vec = jnp.zeros((reps,))
        for i in range(reps):
            subkey, key_mif, key_eval = jax.random.split(subkey, 3)
            dacca.mif(J=J, M=M, rw_sd = rw_sd, a=a, key=key_mif)
            time = dacca.results_history[-1].execution_time
            time_vec = time_vec.at[i].set(time)
            dacca.pfilter(J=np_eval, reps = reps_eval, key = key_eval)
            ll_eval = float(dacca.results_history[-1].to_dataframe()["logLik"][0])
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

def bench_dacca_train(
    J: int, 
    M: int, 
    optimizer: str, 
    eta: float, 
    scale: bool, 
    ls: bool, 
    n_monitors: int, 
    reps: int, 
    seed: int,
    mem_profile: bool = False
) -> dict:
    """
    time is recorded via train function internally;
    Reps is handled by the `_train_loop` function;
    Plug `_train_loop` into memory_profiling to get peak memory.
    """
    devices = jax.devices()
    platform = devices[0].platform if devices else jax.default_backend()

    key = jax.random.key(seed)
    dacca = pp.dacca()
    key, subkey = jax.random.split(key)

    profile_tag = f"dacca_train_{platform}"

    def _train_loop(subkey):
        ll_vec = jnp.zeros((reps,))
        time_vec = jnp.zeros((reps,))
        for i in range(reps):
            subkey, subsubkey = jax.random.split(subkey)
            dacca.train(J=J, M=M, optimizer=optimizer, eta = eta, scale=scale, ls=ls, n_monitors=n_monitors, key=subsubkey)
            ll_last = dacca.results_history[-1].traces_da[0, : ,0].squeeze().values[-1]
            ll_vec = ll_vec.at[i].set(ll_last)
            time = dacca.results_history[-1].execution_time
            time_vec = time_vec.at[i].set(time)
        return ll_vec, time_vec, subkey
    
    if not mem_profile:
        ll_vec, time_vec, subkey = _train_loop(subkey)
    #if mem_profile:
        #dacca, out = memory_profiling(tag = profile_tag, pomp_obj = dacca, fn = _train_loop, subkey = subkey)
        #ll_vec = out[0]
        #time_vec = out[1]
  
    ll_stats = _summarize_ll(ll_vec)
    t_stats = _summarize_time(time_vec)
    return dict(
        method="train",
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

    def ifad_loop(subkey):
        ll_vec   = jnp.zeros((reps,))
        time_vec = jnp.zeros((reps,))
        for i in range(reps):
            subkey, subsubkey = jax.random.split(subkey)
            subkey_mif, subkey_train = jax.random.split(subsubkey)

            dacca.mif(J=J, M=M1, rw_sd=rw_sd, a=cooling, key=subkey_mif)
            t_mif = dacca.results_history[-1].execution_time

            dacca.train(J=J, M=M2, optimizer=optimizer, eta=eta, scale=scale, ls=ls, n_monitors=n_monitors, key=subkey_train)
            t_train = dacca.results_history[-1].execution_time

            ll_last = dacca.results_history[-1].traces_da[0, : ,0].squeeze().values[-1]
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

res_pfilter = bench_dacca_pfilter(J=NP_EVAL, reps=NREPS_EVAL, seed=SEED)
results.append(res_pfilter)
print("[pfilter] ", {k:v for k,v in res_pfilter.items() if k!="obj"})   

res_mop = bench_dacca_mop(J=NP_EVAL, reps=NREPS_EVAL, seed=SEED, alpha=MOP_ALPHA)
results.append(res_mop)
print("[mop]     ", {k:v for k,v in res_mop.items() if k!="obj"})

res_mif = bench_dacca_mif(J=NP_FITR, M=NFITR, rw_sd = RW_SD, a=COOLING_RATE, reps=NREPS_FITR, seed=SEED)
results.append(res_mif)
print("[mif]     ", {k:v for k,v in res_mif.items() if k!="obj"})

res_train = bench_dacca_train(J=NP_FITR, M=NFITR, optimizer=TRAIN_OPTIMIZER, eta=ETA, scale=TRAIN_SCALE, ls=TRAIN_LS, n_monitors=TRAIN_N_MONITORS, reps=NREPS_FITR, seed=SEED)
results.append(res_train)
print("[train]   ", {k:v for k,v in res_train.items() if k!="obj"})

res_ifad = bench_dacca_ifad(J=NP_FITR, M1=NFITR_IFAD, M2=NFITR_IFAD, rw_sd=RW_SD, eta=ETA, cooling=COOLING_RATE, optimizer=TRAIN_OPTIMIZER, scale=TRAIN_SCALE, ls=TRAIN_LS, n_monitors=TRAIN_N_MONITORS, reps=NREPS_FITR, seed=SEED)
results.append(res_ifad)
print("[ifad]    ", {k:v for k,v in res_ifad.items() if k!="obj"})

# save results: save all the disctionaries except the 'obj' fields, and save one final object recording all the elements
bench_summary_dict = []
for r in results:
    r_2 = {k:v for k,v in r.items() if k != "obj"}
    bench_summary_dict.append(r_2)

final_result = {
    "backend": platform,
    "settings": {
        "SEED": SEED,
        "NP_EVAL": NP_EVAL,
        "NP_FITR": NP_FITR,
        "FITR_REPS": NREPS_FITR,
        "EVAL_REPS": NREPS_EVAL,
        "MIF_ITERS": NFITR,
        "TRAIN_ITERS": NFITR,
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


