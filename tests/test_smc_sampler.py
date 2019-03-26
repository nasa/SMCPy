import pytest
import numpy as np
from mpi4py import MPI
from os import path
from smcpy.hdf5.hdf5_storage import HDF5Storage
from smcpy.particles.smc_step import SMCStep
from smcpy.particles.particle import Particle
from smc_tester import SMCTester, Model

'''
Unit and regression tests for the smc_tester.
'''

arr_alm_eq = np.testing.assert_array_almost_equal


@pytest.fixture
def x_space():
    return np.arange(50)


@pytest.fixture
def error_std_dev():
    return 0.6


@pytest.fixture()
def smc_tester():
    np.random.seed(2)
    return SMCTester()


@pytest.fixture
def mpi_comm_world():
    return MPI.COMM_WORLD


@pytest.fixture
def cloned_comm():
    return MPI.COMM_WORLD.Clone()


@pytest.fixture
def smc_tester_rank_seed(cloned_comm):
    np.random.seed(cloned_comm.Get_rank())
    return SMCTester()


@pytest.fixture
def h5_filename():
    return 'test.h5'


@pytest.fixture
def model():
    x_space = np.arange(50)
    return Model(x_space)


def test_communicator_is_clone(smc_tester, mpi_comm_world):
    assert smc_tester._comm.__class__ is mpi_comm_world.__class__
    assert smc_tester._comm is not mpi_comm_world


def test_communicator_is_right_size(smc_tester, mpi_comm_world):
    assert smc_tester._size == mpi_comm_world.Get_size()


def test_communicator_is_right_rank(smc_tester, mpi_comm_world):
    assert smc_tester._rank == mpi_comm_world.Get_rank()


def test_setup_mcmc_sampler(smc_tester):
    from smcpy.mcmc.mcmc_sampler import MCMCSampler
    assert isinstance(smc_tester._mcmc, MCMCSampler)


@pytest.mark.parametrize("input_,exp_error", [(-1, ValueError),
                                              (0, ValueError),
                                              ('1', TypeError),
                                              (1.0, TypeError),
                                              (dict(), TypeError),
                                              ([1], TypeError)])
def test_pos_integer_input_checks(input_, exp_error, smc_tester):
    with pytest.raises(exp_error):
        smc_tester.num_particles = input_
    with pytest.raises(exp_error):
        smc_tester.num_time_steps = input_
    with pytest.raises(exp_error):
        smc_tester.num_mcmc_steps = input_


@pytest.mark.parametrize("input_,exp_error", [(-1, ValueError),
                                              ('1', TypeError),
                                              (dict(), TypeError),
                                              ([1], TypeError)])
def test_ess_threshold_input_checks(input_, exp_error, smc_tester):
    with pytest.raises(exp_error):
        smc_tester.ess_threshold = input_


def test_ess_threshold_default_is_set(smc_tester):
    smc_tester.num_particles = 50
    smc_tester.ess_threshold = None
    expected = smc_tester.num_particles / 2.
    assert smc_tester.ess_threshold == expected


@pytest.mark.parametrize("input_,exp_error", [(1., TypeError),
                                              (['1'], TypeError),
                                              (dict(), TypeError)])
def test_autosave_file_input_checks(input_, exp_error, smc_tester):
    with pytest.raises(exp_error):
        smc_tester.autosaver = input_


@pytest.mark.parametrize("autosave_file,expect", [(h5_filename(), HDF5Storage),
                                                  (None, None.__class__)])
def test_autosave_behavior_is_set(autosave_file, expect, smc_tester,
                                  cloned_comm):
    smc_tester.autosaver = autosave_file
    if cloned_comm.Get_rank() > 0:
        assert smc_tester.autosaver is None
    else:
        assert isinstance(smc_tester.autosaver, expect)
        smc_tester.cleanup_file(autosave_file)


@pytest.mark.parametrize("prop_center,prop_scales,exp_error",
                         [(None, dict(), ValueError),
                          (1, dict(), TypeError),
                          (dict(), 1, TypeError),
                          (1, 1, TypeError)])
def test_check_set_proposal_distribution_inputs(prop_center, prop_scales,
                                                exp_error, smc_tester):
    with pytest.raises(exp_error):
        smc_tester._check_proposal_dist_inputs(prop_center, prop_scales)


@pytest.mark.parametrize("prop_center,prop_scales,exp_error",
                         [({'jabroni': 1}, {'a': 1, 'b': 2}, KeyError),
                          ({'a': 1, 'b': 2}, {'a': 1, 'jabroni': 2}, KeyError)])
def test_check_set_proposal_distribution_input_keys(prop_center, prop_scales,
                                                    exp_error, smc_tester):
    with pytest.raises(exp_error):
        smc_tester._check_proposal_dist_input_keys(prop_center, prop_scales)


@pytest.mark.parametrize("prop_center,prop_scales,exp_error",
                         [({'a': '1', 'b': 2}, {'a': 1., 'b': '2'}, TypeError),
                          ({'a': [1], 'b': 2}, {'a': 1, 'b': {}}, TypeError)])
def test_check_set_proposal_distribution_input_vals(prop_center, prop_scales,
                                                    exp_error, smc_tester):
    with pytest.raises(exp_error):
        smc_tester._check_proposal_dist_input_vals(prop_center, prop_scales)


def test_set_proposal_distribution_with_scales(smc_tester):
    smc_tester.when_proposal_dist_set_with_scales()
    assert smc_tester.proposal_center == smc_tester.expected_center
    assert smc_tester.proposal_scales == smc_tester.expected_scales


def test_set_proposal_distribution_no_scales(smc_tester):
    smc_tester.when_proposal_dist_set_with_no_scales()
    assert smc_tester.proposal_center == smc_tester.expected_center
    assert smc_tester.proposal_scales == smc_tester.expected_scales


@pytest.mark.parametrize("prop_center,expected", [(None, 1), (dict, 0)])
def test_set_start_time_based_on_proposal(smc_tester, prop_center, expected):
    smc_tester.proposal_center = prop_center
    smc_tester._set_start_time_based_on_proposal()
    assert smc_tester._start_time_step == expected


@pytest.mark.parametrize("params", [{'a': 2.42, 'b': 5.74},
                                    {'a': 2.3, 'b': 5.4},
                                    {'a': 2.43349633, 'b': 5.73716365}])
def test_likelihood_from_pymc(smc_tester, params, error_std_dev):
    std_dev = error_std_dev
    data = smc_tester._mcmc.data
    model_eval = smc_tester._mcmc.model.evaluate(params)
    smc_tester._mcmc.generate_pymc_model(fix_var=True, std_dev0=std_dev)
    log_like = smc_tester._evaluate_likelihood(params)
    calc_log_like = smc_tester.calc_log_like_manually(model_eval, data, std_dev)
    arr_alm_eq(log_like, calc_log_like)


def test_val_error_when_proposal_beyond_prior_support(smc_tester):
    smc_tester.when_sampling_parameters_set()
    with pytest.raises(ValueError):
        smc_tester.when_initial_particles_sampled_from_proposal_outside_prior()


def test_initialize_from_proposal(smc_tester, error_std_dev):
    params = {'a': np.array(2.43349633), 'b': np.array(5.73716365)}
    weight = 0.06836560508406836
    log_like = -706.453419056

    smc_tester.when_sampling_parameters_set()
    smc_tester.when_initial_particles_sampled_from_proposal(error_std_dev)

    first_particle = smc_tester.particles[0]
    first_particle.print_particle_info()
    arr_alm_eq(first_particle.params.values(), params.values())
    arr_alm_eq(first_particle.weight, weight)
    arr_alm_eq(first_particle.log_like, log_like)
    assert len(smc_tester.particles) == 1


def test_initialize_from_prior(smc_tester, error_std_dev):
    params = {'a': np.array(-1.87449914), 'b': np.array(-9.45595268)}
    weight = 0.0
    log_like = -1242179.09405

    error_std_dev = 0.6
    smc_tester.when_sampling_parameters_set()
    smc_tester.when_initial_particles_sampled_from_prior(error_std_dev)

    first_particle = smc_tester.particles[0]
    first_particle.print_particle_info()
    arr_alm_eq(first_particle.params.values(), params.values())
    arr_alm_eq(first_particle.weight, weight)
    arr_alm_eq(first_particle.log_like, log_like)
    assert len(smc_tester.particles) == 1


def test_initialize_step(smc_tester, cloned_comm):
    error_std_dev = 0.6

    smc_tester.when_sampling_parameters_set()
    smc_tester.when_initial_particles_sampled_from_proposal(error_std_dev)
    step = smc_tester._initialize_step(smc_tester.particles)
    if cloned_comm.Get_rank() == 0:
        assert isinstance(step, SMCStep)
        assert len(step.get_particles()) == cloned_comm.Get_size()
        arr_alm_eq(sum(step.get_weights()), 1.)
    else:
        assert step is None


@pytest.mark.parametrize("restart_step", [-1, 3])
def test_raise_value_error_when_restart_step_invalid(smc_tester, restart_step):
    with pytest.raises(ValueError):
        smc_tester.when_sampling(restart_step, hdf5_to_load=None,
                                 autosave_file=None)


def test_restart_step(smc_tester):
    with pytest.raises(ValueError):
        smc_tester.when_sampling(restart_time_step=2, hdf5_to_load=None,
                                 autosave_file=None)


def test_save_step_list(smc_tester, h5_filename, cloned_comm):
    if cloned_comm.Get_rank() == 0:
        assert not path.exists(h5_filename)
    smc_tester.when_sampling_parameters_set()
    smc_tester.when_step_created()
    smc_tester.save_step_list(h5_filename)
    if cloned_comm.Get_rank() == 0:
        assert path.exists(h5_filename)
        smc_tester.cleanup_file(h5_filename)


def test_step_list_trimmer(smc_tester, cloned_comm, h5_filename):
    smc_tester.when_sampling_parameters_set()
    smc_tester.when_step_created()
    smc_tester.save_step_list(h5_filename)
    if cloned_comm.Get_rank() == 0:
        assert path.exists(h5_filename)
        step_list = smc_tester.load_step_list(h5_filename)
        smc_tester.assert_step_lists_almost_equal(step_list,
                                                  smc_tester.step_list)
        smc_tester.cleanup_file(h5_filename)


@pytest.mark.parametrize("restart_time", [0, 1])
def test_set_start_time_equal_to_restart_time_step(smc_tester, restart_time):
    smc_tester.num_time_steps = 2
    smc_tester.restart_time_step = restart_time
    smc_tester._set_start_time_equal_to_restart_time_step()
    assert smc_tester._start_time_step == restart_time


def test_trim_step_list(smc_tester, cloned_comm):
    smc_tester.when_sampling_parameters_set()
    smc_tester.when_step_created()
    if cloned_comm.Get_rank() == 0:
        assert len(smc_tester.step_list) == 1
        trimmed_list = smc_tester._trim_step_list(smc_tester.step_list, -1)
        assert len(trimmed_list) == 0
    else:
        assert smc_tester.step_list is None


@pytest.mark.parametrize("input_", [0, [1], dict()])
def test_set_step_type_error(smc_tester, input_):
    with pytest.raises(TypeError):
        smc_tester.step = SMCStep()
        smc_tester.step.fill_step(input_)


def test_create_new_particles(smc_tester, cloned_comm):
    log_like = -706.4534190556333
    params = {'a': np.array(2.43349633), 'b': np.array(5.73716365)}

    smc_tester.when_sampling_parameters_set()
    smc_tester.when_step_created()
    new_particles = smc_tester._create_new_particles(0.2)
    if cloned_comm.Get_rank() > 0:
        assert isinstance(new_particles, list)
        assert all([isinstance(x, Particle) for x in new_particles])
    else:
        assert isinstance(new_particles, list)
        assert all([isinstance(x, Particle) for x in new_particles])
        arr_alm_eq(new_particles[0].log_like, log_like)
        arr_alm_eq(new_particles[0].params.values(), params.values())


def test_compute_step_covariance(smc_tester):
    '''
    Rank is same on each processor so each processor returns a duplicate set
    of particles, meaning this test will pass regardless of processes. Simply
    here to test the plumbing.
    '''
    cov_test = np.array([[0.91459139, -0.30873497],
                         [-0.30873497, 3.44351062]])

    smc_tester.when_sampling_parameters_set(num_particles_per_processor=10)
    smc_tester.when_step_created()
    cov = smc_tester._compute_step_covariance()
    assert isinstance(cov, np.ndarray)
    arr_alm_eq(cov, cov_test)


def test_mutate_new_particles(smc_tester, cloned_comm):
    params = {'a': np.array(2.43349633), 'b': np.array(5.73716365)}
    log_like = -712.444883603

    smc_tester.when_sampling_parameters_set(num_particles_per_processor=10)
    smc_tester.when_step_created()
    smc_tester.when_particles_mutated()
    mutated_particles = smc_tester.mutated_particles

    if cloned_comm.Get_rank() > 0:
        assert mutated_particles is None
    else:
        assert isinstance(mutated_particles, list)
        assert all([isinstance(x, Particle) for x in mutated_particles])
        arr_alm_eq(mutated_particles[0].log_like, log_like)
        arr_alm_eq(mutated_particles[0].params.values(), params.values())


def test_update_step_with_new_particles(smc_tester, cloned_comm):
    smc_tester.when_sampling_parameters_set()
    smc_tester.when_step_created()
    smc_tester.when_particles_mutated()
    mutated_particles = smc_tester.mutated_particles

    if cloned_comm.Get_rank() == 0:
        pc1 = smc_tester.step.copy()
        pc1.fill_step(mutated_particles)
        smc_tester._update_step_with_new_particles(mutated_particles)
        pc2 = smc_tester.step
        smc_tester.assert_steps_almost_equal(pc1, pc2)
    else:
        assert mutated_particles is None
        assert smc_tester._update_step_with_new_particles(
            mutated_particles) is None


def test_autosave_step(smc_tester, h5_filename, cloned_comm):
    assert not path.exists(h5_filename)
    smc_tester.when_sampling_parameters_set(autosave_file=h5_filename)
    smc_tester.when_step_created()
    smc_tester.when_particles_mutated()
    mutated_particles = smc_tester.mutated_particles
    smc_tester._update_step_with_new_particles(mutated_particles)
    smc_tester._autosave_step()

    if cloned_comm.Get_rank() == 0:
        pc1 = smc_tester.step_list
        pc2 = smc_tester.load_step_list(h5_filename)
        smc_tester.assert_step_lists_almost_equal(pc1, pc2)
        smc_tester.cleanup_file(h5_filename)
    else:
        assert smc_tester.autosaver is None


def test_close_autosaver(smc_tester, h5_filename, cloned_comm):
    smc_tester.when_sampling_parameters_set(autosave_file=h5_filename)
    if cloned_comm.Get_rank() == 0:
        assert path.exists(h5_filename)
        smc_tester.autosaver.close()
        with pytest.raises(ValueError):
            smc_tester.autosaver._h5.mode
        smc_tester.cleanup_file(h5_filename)
        assert not path.exists(h5_filename)
    else:
        assert smc_tester.autosaver is None


def test_error_processing(smc_tester):
    with pytest.raises(TypeError):
        smc_tester.error_processing_args(model)
