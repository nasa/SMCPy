import numpy as np
import pytest

from smcpy.smc.updater import Updater
from smcpy import VectorMCMC, VectorMCMCKernel
from smcpy.paths import GeometricPath


class MockedParticles:
    def __init__(self, params, log_likes, log_weights):
        self.param_names = tuple(params.keys())
        self.param_dict = params
        self.params = np.array([params[k] for k in self.param_names]).T
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.weights = np.exp(log_weights)
        self.num_particles = len(log_weights)
        self.total_unnorm_log_weight = 99

    def compute_ess(self):
        return 0.5 * self.params.shape[0]


@pytest.fixture
def mocked_particles(mocker):
    params = {"a": np.array([1, 1, 2]), "b": np.array([2, 3, 4])}
    log_likes = np.array([-1, -2, -3]).reshape(-1, 1)
    log_weights = np.log([0.1, 0.6, 0.3]).reshape(-1, 1)
    particles = MockedParticles(params, log_likes, log_weights)
    return particles


@pytest.fixture
def mocked_particles_resample(mocker):
    params = {"a": np.array([1, 1, 2]), "b": np.array([2, 3, 4])}
    log_likes = np.array([-1, -2, -3]).reshape(-1, 1)
    log_weights = np.array([0, 0, 1]).reshape(-1, 1)
    particles = MockedParticles(params, log_likes, log_weights)
    return particles


@pytest.mark.parametrize("ess_threshold", [-0.1, 1.1])
def test_ess_threshold_valid(ess_threshold):
    with pytest.raises(ValueError):
        Updater(ess_threshold, mcmc_kernel=None)


@pytest.mark.parametrize("particles_warn_threshold", [-99, -1.9, 1.1, 3, 10000])
def test_particles_warn_threshold_valid(particles_warn_threshold):
    ess_threshold = 0.8
    with pytest.raises(ValueError):
        Updater(
            ess_threshold=ess_threshold,
            mcmc_kernel=None,
            particles_warn_threshold=particles_warn_threshold,
        )


def test_update_particles_warning(mocked_particles_resample, mocker):
    mocker.patch("smcpy.smc.updater.Particles", new=MockedParticles)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b"])
    particles_warn_threshold = 1.0
    mocker.patch.object(kernel, "get_log_priors", return_value=np.ones((3, 1)))

    updater = Updater(
        ess_threshold=1.0,
        mcmc_kernel=kernel,
        particles_warn_threshold=particles_warn_threshold,
    )

    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    particles_warn_threshold = 1.0
    with pytest.warns(UserWarning, match="Resampled to less than 100.0% of particles;"):
        updater.update(mocked_particles_resample)


def test_update_no_proposal(mocked_particles, mocker):
    mocker.patch("smcpy.smc.updater.Particles", new=MockedParticles)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b"])
    mocker.patch.object(kernel, "get_log_priors", return_value=np.ones((3, 2)))

    updater = Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    expect_log_weights = (
        mocked_particles.log_likes * delta_phi + mocked_particles.log_weights
    )

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_equal(new_particles.log_weights, expect_log_weights)
    np.testing.assert_array_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_equal(new_particles.log_likes, mocked_particles.log_likes)


def test_update_w_proposal(mocked_particles, mocker):
    mocker.patch("smcpy.smc.updater.Particles", new=MockedParticles)

    proposal = mocker.Mock()
    proposal.logpdf.return_value = np.full((3, 1), 4)

    path = GeometricPath(proposal=proposal)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b"], path)
    mocker.patch.object(kernel, "get_log_priors", return_value=np.ones((3, 2)))

    updater = Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    expect_log_weights = (
        mocked_particles.log_likes + 2 - 4
    ) * delta_phi + mocked_particles.log_weights

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_almost_equal(new_particles.log_weights, expect_log_weights)
    np.testing.assert_array_almost_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_almost_equal(
        new_particles.log_likes, mocked_particles.log_likes
    )


def test_update_w_proposal_and_req_phi(mocked_particles, mocker):
    mocker.patch("smcpy.smc.updater.Particles", new=MockedParticles)

    proposal = mocker.Mock()
    proposal.logpdf.return_value = np.full((3, 1), 4)

    path = GeometricPath(proposal=proposal, required_phi=0.15)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b"], path)
    mocker.patch.object(kernel, "get_log_priors", return_value=np.ones((3, 2)))

    updater = Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    log_numer = 4 * 0 + 2 * 1 + mocked_particles.log_likes * 0.2
    log_denom = 4 * (0.05 / 0.15) + 2 * (0.1 / 0.15) + mocked_particles.log_likes * 0.1
    exp_log_weights = mocked_particles.log_weights + log_numer - log_denom

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_almost_equal(new_particles.log_weights, exp_log_weights)
    np.testing.assert_array_almost_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_almost_equal(
        new_particles.log_likes, mocked_particles.log_likes
    )


def test_update_w_zero_prob_prior(mocked_particles, mocker):
    mocker.patch("smcpy.smc.updater.Particles", new=MockedParticles)

    proposal = mocker.Mock()
    proposal.logpdf.return_value = np.full((3, 1), 4)

    path = GeometricPath(proposal=proposal)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b"], path)
    mocker.patch.object(
        kernel, "get_log_priors", return_value=np.array([[1, 1], [1, 1], [1, -np.inf]])
    )

    updater = Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = delta_phi

    exp_log_weights = (
        mocked_particles.log_likes + 2 - 4
    ) * delta_phi + mocked_particles.log_weights
    exp_log_weights[-1] = -np.inf

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_almost_equal(new_particles.log_weights, exp_log_weights)
    np.testing.assert_array_almost_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_almost_equal(
        new_particles.log_likes, mocked_particles.log_likes
    )


def test_update_w_zero_in_numer_and_denom(mocked_particles, mocker):
    mocker.patch("smcpy.smc.updater.Particles", new=MockedParticles)

    proposal = mocker.Mock()
    proposal.logpdf.return_value = np.array(([4], [4], [-np.inf]))

    path = GeometricPath(proposal=proposal)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b"], path)
    mocker.patch.object(kernel, "get_log_priors", return_value=np.ones((3, 2)))

    updater = Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    exp_log_weights = (
        mocked_particles.log_likes + 2 - 4
    ) * delta_phi + mocked_particles.log_weights
    exp_log_weights[-1] = -np.inf

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_almost_equal(new_particles.log_weights, exp_log_weights)
    np.testing.assert_array_almost_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_almost_equal(
        new_particles.log_likes, mocked_particles.log_likes
    )


@pytest.mark.parametrize(
    "random_samples, resample_indices",
    (
        (np.array([0.3, 0.3, 0.7]), [1, 1, 2]),
        (np.array([0.3, 0.05, 0.05]), [1, 0, 0]),
        (np.array([0.3, 0.05, 0.7]), [1, 0, 2]),
        (np.array([0.05, 0.05, 0.7]), [0, 0, 2]),
        (np.array([0.3, 0.7, 0.3]), [1, 2, 1]),
    ),
)
def test_update_with_standard_resample(
    mocked_particles, random_samples, resample_indices, mocker
):
    mocker.patch("smcpy.smc.updater.Particles", new=MockedParticles)
    rng = mocker.Mock(np.random.default_rng(), autospec=True)
    rng.uniform.return_value = random_samples
    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b"], rng=rng)

    updater = Updater(ess_threshold=1.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    new_weights = np.array([0.1, 0.4, 0.5])
    mocker.patch.object(
        updater, "_compute_new_weights", return_value=np.log(new_weights)
    )

    new_particles = updater.update(mocked_particles)

    assert updater.ess < len(mocked_particles.params)
    assert updater.resampled == True

    np.testing.assert_array_equal(new_particles.log_weights, np.log(np.ones(3) * 1 / 3))
    np.testing.assert_array_equal(
        new_particles.log_likes, mocked_particles.log_likes[resample_indices]
    )
    np.testing.assert_array_equal(
        new_particles.params, mocked_particles.params[resample_indices]
    )

    assert new_particles._total_unlw == 99


def test_resample_using_stratified_sampling_uniform_weights(mocker, mocked_particles):
    mocked_kernel = mocker.Mock()
    mocked_kernel.rng = np.random.default_rng()
    n_particles = mocked_particles.num_particles
    mocked_particles.weights = np.array([[1 / n_particles]] * n_particles)
    updater = Updater(
        ess_threshold=1.0, mcmc_kernel=mocked_kernel, resample_strategy="stratified"
    )

    new_particles = updater.resample_if_needed(mocked_particles)
    assert updater._resampled == True
    np.testing.assert_array_equal(mocked_particles.params, new_particles.params)


@pytest.mark.parametrize(
    "uniform_sample, expected", (([0.3, 0.5, 0.5], [1, 2]), ([0.4, 0.5, 0.5], (1, 3)))
)
def test_resample_using_stratified_sampling_nonuniform_weights(
    mocker, mocked_particles, uniform_sample, expected
):
    mocked_rng = mocker.Mock(np.random.default_rng(), autospec=True)
    mocked_rng.uniform.return_value = uniform_sample
    mocked_kernel = mocker.Mock()
    mocked_kernel.rng = mocked_rng

    expected_params = np.array([expected, [1, 3], [2, 4]])

    updater = Updater(
        ess_threshold=1.0, mcmc_kernel=mocked_kernel, resample_strategy="stratified"
    )

    new_particles = updater.resample_if_needed(mocked_particles)
    assert updater._resampled == True
    np.testing.assert_array_equal(expected_params, new_particles.params)


def test_resample_raises_with_invalid_strategy(mocker, mocked_particles):
    with pytest.raises(ValueError):
        Updater(
            ess_threshold=1.0, mcmc_kernel=mocker.Mock(), resample_strategy="bad-strat"
        )


def test_resample_strategy_case_insensitive(mocker):
    Updater(ess_threshold=1.0, mcmc_kernel=mocker.Mock(), resample_strategy="StAnDaRd")
