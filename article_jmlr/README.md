
# Setup for building ms.pdf

Install the project dependencies and build the manuscript from this directory:
```bash
make venv
make -B ms.pdf
```

`make venv` runs `uv sync --project ..`, creating the shared environment at
`../.venv`. The Makefile sets `QUARTO_PYTHON` to that environment, so activating
it is unnecessary. The PDF build also initializes the pinned `quant` submodule
for benchmark data (requiring network access on the first build). To use another
environment, set `QUARTO_PYTHON` explicitly:
```bash
make -B ms.pdf QUARTO_PYTHON=/absolute/path/to/bin/python
```

## Notes from May 21

- Need to differentiate between our package and ``dynamax`` and ``pomp``.
  - For the former, they have no support for highly nonlinear/non-Gaussian SSMS, with unknown process densities, or the particle filter.
    - We have to argue that there are enough of these, and we can appeal to epidemiology and ecology. We address the broadest of SSMs that are tractable for likelihood-based inference. This would (TODO) include SSMs with both discrete and continuous states. Cite the usual string of papers to justify this. 
    - Usage of SMC is mentioned in the README for parameter posterior estimation, but to our knowledge it is not actually included within the package.
    - ``dynamax`` is fundamentally a Bayesian package, though it is possible to estimate the MLE via gradient descent within their package. This, however, does not address frequentist parameter inference. We provide facilities for both frequentist (TODO: likelihood profiling and MCAP) and Bayesian (TODO: BIF, NUTS with particle filter gradient estimates).
    - Can we improve upon Linderman's AOAS paper on a neural system?
  - For the latter, we compete mostly in terms of speed and likelihood optimization, as well as in providing CUDA/GPU and native Python support.
- Need to finish the tutorial article?

