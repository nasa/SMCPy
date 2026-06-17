"""
This Proposal class references and adapts from the follow two papers:

@InProceedings{pmlr-v32-sejdinovic14,
  title = 	 {Kernel Adaptive Metropolis-Hastings},
  author = 	 {Sejdinovic, Dino and Strathmann, Heiko and Garcia, Maria Lomeli and Andrieu, Christophe and Gretton, Arthur},
  booktitle = 	 {Proceedings of the 31st International Conference on Machine Learning},
  pages = 	 {1665--1673},
  year = 	 {2014},
  editor = 	 {Xing, Eric P. and Jebara, Tony},
  volume = 	 {32},
  number =       {2},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Bejing, China},
  month = 	 {22--24 Jun},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v32/sejdinovic14.pdf},
  url = 	 {https://proceedings.mlr.press/v32/sejdinovic14.html},
  abstract = 	 {A Kernel Adaptive Metropolis-Hastings algorithm is introduced, for the purpose of sampling from a target distribution with strongly nonlinear support. The algorithm embeds the trajectory of the Markov chain into a reproducing kernel Hilbert space (RKHS), such that the feature space covariance of the samples informs the choice of proposal. The procedure is computationally efficient and straightforward to implement, since the RKHS moves can be integrated out analytically: our proposal distribution in the original space is a normal distribution whose mean and covariance depend on where the current sample lies in the support of the target distribution, and adapts to its local covariance structure. Furthermore, the procedure requires neither gradients nor any other higher order information about the target, making it particularly attractive for contexts such as Pseudo-Marginal MCMC. Kernel Adaptive Metropolis-Hastings outperforms competing fixed and adaptive samplers on multivariate, highly nonlinear target distributions, arising in both real-world and synthetic examples.}
}

@InProceedings{10.1007/978-3-319-71249-9_24,
author="Schuster, Ingmar
and Strathmann, Heiko
and Paige, Brooks
and Sejdinovic, Dino",
editor="Ceci, Michelangelo
and Hollm{\'e}n, Jaakko
and Todorovski, Ljup{\v{c}}o
and Vens, Celine
and D{\v{z}}eroski, Sa{\v{s}}o",
title="Kernel Sequential Monte Carlo",
booktitle="Machine Learning and Knowledge Discovery in Databases",
year="2017",
publisher="Springer International Publishing",
address="Cham",
pages="390--409",
abstract="We propose kernel sequential Monte Carlo (KSMC), a framework for sampling from static target densities. KSMC is a family of sequential Monte Carlo algorithms that are based on building emulator models of the current particle system in a reproducing kernel Hilbert space. We here focus on modelling nonlinear covariance structure and gradients of the target. The emulator's geometry is adaptively updated and subsequently used to inform local proposals. Unlike in adaptive Markov chain Monte Carlo, continuous adaptation does not compromise convergence of the sampler. KSMC combines the strengths of sequental Monte Carlo and kernel methods: superior performance for multimodal targets and the ability to estimate model evidence as compared to Markov chain Monte Carlo, and the emulator's ability to represent targets that exhibit high degrees of nonlinearity. As KSMC does not require access to target gradients, it is particularly applicable on targets whose gradients are unknown or prohibitively expensive. We describe necessary tuning details and demonstrate the benefits of the the proposed methodology on a series of challenging synthetic and real-world examples.",
isbn="978-3-319-71249-9"
}
"""

import numpy as np
import warnings

from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_kernels


class LocalMCMCProposal:
    def __call__(self, inputs, scale_factor):
        if inputs.shape[0] < 2:
            raise ValueError("inputs parameter contains less than 2 points")
        orig_inputs = inputs.copy()
        orig_cov = self.compute_covariance(inputs, scale_factor)
        prop_inputs = self.proposal(inputs, orig_cov)
        prop_cov = self.compute_covariance(prop_inputs, scale_factor)
        return orig_inputs, orig_cov, prop_inputs, prop_cov

    def compute_covariance(self, inputs, scale_factor):
        D = inputs.shape[1]
        NU = 2.38 / np.sqrt(D)

        rbfs = self.rbf(inputs, sigma=self.median_distance(inputs))
        R = NU**2 * np.array(
            [np.atleast_2d(np.cov(inputs.T, aweights=r)) for r in rbfs]
        )
        return R * scale_factor

    def rbf(self, inputs, sigma):
        if sigma <= 0:
            raise ValueError(f"Sigma must be postive, got {sigma}")

        gamma = -1 / (2 * sigma**2)
        return pairwise_kernels(inputs.copy(), metric="rbf", gamma=gamma)

    def median_distance(self, inputs):
        pairwise_dist = pdist(inputs, metric="euclidean")
        return np.median(pairwise_dist)

    def proposal(self, inputs, cov):
        chol = np.array([self._ensure_psd_cov_and_do_chol_decomp(mat) for mat in cov])
        z = np.random.default_rng().normal(0, 1, inputs.shape)
        delta = np.einsum("ijk,ik->ij", chol, z)
        return inputs + delta

    def _ensure_psd_cov_and_do_chol_decomp(self, cov):
        """
        Higham NJ. Computing a nearest symmetric positive semidefinite matrix.
        Linear Algebra and its Applications. 1988 May;103(C):103-118.

        Code implementation: https://stackoverflow.com/a/63131250/4733085
        """
        try:
            return np.linalg.cholesky(cov)
        except:
            warnings.warn(
                "Covariance matrix is not positive semi-definite; "
                "forcing negative eigenvalues to zero and rebuilding "
                "covariance matrix."
            )
            eigval, eigvec = np.linalg.eigh(cov)
            eigval[eigval < 0] = 0
            cov = (eigvec @ np.diag(eigval)) @ eigvec.T
            cov += 1e-14 * np.eye(len(eigval))
            return np.linalg.cholesky(cov)
