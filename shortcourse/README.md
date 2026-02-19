
# Simulation-based inference for epidemiological dynamics with pypomp

Aaron A. King, Edward L. Ionides and Kunyang He

## Overview

This short course introduces statistical inference techniques and computational methods for dynamic models of epidemiological systems.
The course explores deterministic and stochastic formulations of epidemiological dynamics and develop inference methods appropriate for a range of models.
Special emphasis will be on exact and approximate likelihood as the key elements in parameter estimation, hypothesis testing, and model selection.
Specifically, the course emphasizes sequential Monte Carlo techniques.

This is a Python-`pypomp` version of the previous [R-pomp](https://github.com/kingaa/sbied) short course. The `pypomp` package supports GPU computation and automatic differentiation via `JAX`.

1. To introduce partially observed Markov process (POMP) models as tools for scientific investigation and public health policy.
1. To give students the ability to formulate POMP models of their own.
1. To teach efficient approaches for performing scientific inference using POMP models.
1. To familiarize students with the **pypomp** package.
1. To help students familiar with the **R-pomp** package to transfer that expertise to **pypomp**.
1. To give students opportunities to work with inference methods, including methodologies that use automatic differentiation and GPUs which are not accessible via **pomp**.
1. To provide documented examples for student re-use.

----------------------

## Lessons

[**0. Instructions for preparing your laptop for the course exercises.**](chapter0)

[**1. Introduction: What is "Simulation-based Inference for Epidemiological Dynamics"?  POMPs and pypomp.**](chapter1)

[**2. Simulation of stochastic dynamic models.**](chapter2)

[**3. Likelihood for POMPs: theory and practice.**](chapter3)

[**4. Iterated filtering: theory and practice.**](chapter4)

[**5. Case study: measles.  Recurrent epidemics, long time series, covariates, extra-demographic stochasticity, interpretation of parameter estimates.**](chapter5)

[**6. Case study: polio. Workflow for a real research problem.**](chapter6)

[**7. Case study: Ebola. Model diagnostics and forecasting.**](chapter7)

**8. Case study: HIV and fluctuating sexual contact rates. Panel data.** Not yet implemented.


