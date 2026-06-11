import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from scipy.stats import uniform, multivariate_normal

sys.path.append(os.path.join(os.getcwd(), "../../"))
sys.path.append(os.path.join(os.getcwd(), "../../smcpy"))
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler as Sampler
from smcpy.paths import GeometricPath
from smcpy.mfmc_proposal import MultiFidelityProposal

from toy_helpers import M_HF, M_LF, generate_noisy_data

# Data generation details
STD_DEV = 0.2
theta_0 = 1/20
theta_1 = 1
THETA_TRUE = np.array([[theta_0, theta_1]])
NUM_PARTICLES = 5_000
np.random.seed(42)
noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV)

priors = [uniform(0.001, 2), uniform(-2, 8)]
vector_mcmc = VectorMCMC(M_HF, noisy_data, priors, STD_DEV)

# initialize from prior
mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("theta_0", "theta_1"))
smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
reg_step_list, mll_list = smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=5,
    target_ess=0.5
)
reg_phi_list = smc.phi_sequence

lofi_samples = reg_step_list[-1].params

test_prop = MultiFidelityProposal(lofi_samples, M_LF, noisy_data)
prop_samples = test_prop.rvs(10, random_state=42)

print("Samples: ")
print(prop_samples)

print("LogPdf Values: ")
print(test_prop.logpdf(prop_samples, 0.2))