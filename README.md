SMCPy - **S**equential **M**onte **C**arlo with **Py**thon 
==========================================================================
[![Build](https://github.com/nasa/SMCPy/actions/workflows/tests.yml/badge.svg)](https://github.com/nasa/SMCPy/actions)

## Description
SMCPy is an open-source package for performing uncertainty quantification using
a parallelized sequential Monte Carlo sampler.

## Key Features
* Alternative to Markov chain Monte Carlo for Bayesian inference problems
* Unbiased estimation of marginal likelihood for Bayesian model selection
* Parallelization through either numpy vectorization or mpi4py

# Quick Start

## Installation
To install SMCPy, use pip.
```sh
pip install smcpy
```

## Overview
To operate the code, the user supplies a computational model built in Python
3.6+, defines prior distributions for each of the model parameters to be
estimated, and provides data to be used for probabilistic model calibration. SMC
sampling of the parameter posterior distribution can then be conducted with ease
through instantiation of a sampler class and a call to the sample() method.

The two primary sampling algorithms implemented in this package are MPI-enabled
versions of those presented in the following articles, respectively:
> Nguyen, Thi Le Thu, et al. "Efficient sequential Monte-Carlo samplers for Bayesian
> inference." IEEE Transactions on Signal Processing 64.5 (2015): 1305-1319.
[Link to Article](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7339702) | [BibTeX Reference](https://scholar.googleusercontent.com/scholar.bib?q=info:L7AZJvppx1MJ:scholar.google.com/&output=citation&scisdr=CgUT24-FENXorVVNYK0:AAGBfm0AAAAAXYJIeK1GJKW947imCXoXAkfc7yZjQ7Oo&scisig=AAGBfm0AAAAAXYJIeNYSGEVCrlauowP6jMwVMHB_blTp&scisf=4&ct=citation&cd=-1&hl=en)


> Buchholz, Alexander, Nicolas Chopin, and Pierre E. Jacob. "Adaptive tuning of
> hamiltonian monte carlo within sequential monte carlo." Bayesian Analysis
> 1.1 (2021): 1-27.
[Link to Article](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Adaptive-Tuning-of-Hamiltonian-Monte-Carlo-Within-Sequential-Monte-Carlo/10.1214/20-BA1222.full) | [BibTeX Reference](https://scholar.googleusercontent.com/scholar.bib?q=info:wkjyyAN3q3UJ:scholar.google.com/&output=citation&scisdr=CgUA1gUaENXokaHu_K0:AAGBfm0AAAAAYXbr5K0e7EUBTRYw-hgqrmqC-G0ghzIo&scisig=AAGBfm0AAAAAYXbr5FfqGNe5PbrfGSvhMKzBoUbwdXDH&scisf=4&ct=citation&cd=-1&hl=en)

The first is a simple likelihood tempering approach in which the tempering
sequence is fixed and user-specified
([FixedPhiSampler](https://github.com/nasa/SMCPy/blob/8b7813106de077c80992ba37d2d85944d6cce40c/smcpy/samplers.py#L44)).
The second is an adaptive approach that chooses the tempering steps based on a
target effective sample size ([AdaptiveSampler](https://github.com/nasa/SMCPy/blob/8b7813106de077c80992ba37d2d85944d6cce40c/smcpy/samplers.py#L92)).

This software was funded by and developed under the High Performance Computing
Incubator (HPCI) at NASA Langley Research Center.

## Example Usage

We'll set up a toy example -- estimating the slope an intercept of a line given only noisy observations of that line.
```python
import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import uniform

from smcpy import AdaptiveSampler, VectorMCMC, VectorMCMCKernel
from smcpy.utils.noise_generator import generate_noisy_data


def model(params):
    return params[:, [0]] * np.linspace(0.5, 2.5, 100) + params[:, [1]]


if __name__ == "__main__":

    std_dev = 0.5
    true_params = np.array([[2, 3.5]]) # true but unknown (to be estimated)
    noisy_data = generate_noisy_data(model(true_params), std_dev)
    priors = [uniform(-6, 12.0), uniform(-6, 12.0)]

    vector_mcmc = VectorMCMC(model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, ("slope", "intercept"))

    smc = AdaptiveSampler(mcmc_kernel)
    step_list, mll_list = smc.sample(num_particles=500, num_mcmc_samples=10)

    print(f"marginal log likelihood = {mll_list[-1]}")
    print(f"parameter means = {step_list[-1].compute_mean()}")

    sns.pairplot(pd.DataFrame(step_list[-1].param_dict))
    sns.mpl.pyplot.savefig("pairwise.png")
```
### Output
```bash
[ mutation ratio: 1.0: : 100.00%|███████████████████████████████████████| phi: 1.00000/1.0 [00:00<00:00
marginal log likelihood = -76.27987428009173
parameter means = {'slope': np.float64(2.0148875448007013), 'intercept': np.float64(3.523915137965013)}
```
Plotting is easy with `seaborn`:
<p align="center">
<img src="https://github.com/nasa/SMCPy/blob/main/examples/linear_example/pairwise.png" width="400" alt="Pairwise plot"/>
</p>

To run this example in parallel using MPI, the MCMC kernel just needs to be built with the
`ParallelVectorMCMC` class in place of `VectorMCMC`. More details can be found in the
[MPI example](https://github.com/nasa/SMCPy/blob/main/examples/mpi_example/run_example.py).

Tests
-----

Clone the repo and move into the package directory:

```sh
git clone https://github.com/nasa/SMCPy.git
cd SMCPy
```

Install requirements necessary to use SMCPy:

```sh
pip install -r requirements.txt
```

Optionally, if you'd like to use the MPI-enabled parallel sampler, install the
associated requirements:

```sh
pip install -r requirements_optional.txt
```

Add SMCPy to your Python path. For example:

```sh
export PYTHONPATH="$PYTHONPATH:/path/to/smcpy"
```

Run the tests to ensure proper installation:

```sh
pytest tests
```

## Contributing
1.  Fork (<https://github.com/nasa/SMCPy/fork>)
2.  Create your feature branch (`git checkout -b feature/fooBar`)
3.  Commit your changes (`git commit -am 'Add some fooBar'`)
4.  Push to the branch (`git push origin feature/fooBar`)
5.  Create a Pull Request

# Development
NASA Langley Research Center <br /> 
Hampton, Virginia <br /> 

This software was funded by and developed under the High Performance Computing Incubator (HPCI) at NASA Langley Research Center. <br /> 

## Authors
* Patrick Leser
* Julia Truong
* Michael Wang

# License
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
