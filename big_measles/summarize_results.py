# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow"]
# ///
"""
Write the small CSV summaries of a panel fit that the tutorial reads.

Reads traces.parquet and results.parquet from a run directory and writes the
summaries to <run_dir>/summary/. Run from this directory with:

    uv run summarize_results.py results/2026-09-04_233751_u1422_L4
"""

import sys
from pathlib import Path

import pandas as pd

# The 20 towns analyzed by He et al. (2010).
HE10_TOWNS = [
    "Bedwellty", "Birmingham", "Bradford", "Bristol", "Cardiff", "Consett",
    "Dalton.in.Furness", "Halesworth", "Hastings", "Hull", "Leeds", "Lees",
    "Liverpool", "London", "Manchester", "Mold", "Northwich", "Nottingham",
    "Oswestry", "Sheffield",
]  # fmt: skip


def main(run_dir: Path):
    out = run_dir / "summary"
    out.mkdir(exist_ok=True)
    fmt = "%.8g"

    traces = pd.read_parquet(run_dir / "traces.parquet")
    mif = traces[traces["method"] == "mif"]

    panel_trace = mif.loc[mif["unit"] == "shared", ["theta_idx", "iteration", "logLik"]]
    panel_trace.to_csv(out / "panel_loglik_trace.csv", index=False, float_format=fmt)

    town_traces = mif[mif["unit"].isin(HE10_TOWNS)].drop(columns=["method", "se"])
    town_traces.to_csv(out / "he10_town_traces.csv.gz", index=False, float_format=fmt)

    # Each search's first pfilter row precedes the pfilter of the pruned,
    # mixed-and-matched parameter set, which reuses theta_idx 0.
    pfilter = traces[traces["method"] == "pfilter"]
    chain_evals = pfilter.drop_duplicates(subset=["theta_idx", "unit"], keep="first")
    chain_evals = chain_evals[["theta_idx", "unit", "logLik", "se"]]
    chain_evals.to_csv(out / "chain_evaluations.csv.gz", index=False, float_format=fmt)

    results = pd.read_parquet(run_dir / "results.parquet")
    results.to_csv(out / "final_estimates.csv.gz", index=False, float_format=fmt)

    for path in sorted(out.iterdir()):
        print(f"{path}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
