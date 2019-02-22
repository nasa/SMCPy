from mpi4py import MPI
import numpy as np
from os import remove
from smcpy.smc.smc_sampler import SMCSampler
from smcpy.model.base_model import BaseModel


class Model(BaseModel):
    '''
    Model used for SMC sampler testing
    '''

    def __init__(self, x):
        self.x = np.array(x)

    def evaluate(self, *args, **kwargs):
        params = self.process_args(args, kwargs)
        a = params['a']
        b = params['b']
        return a * self.x + b


class SMCTester(SMCSampler):
    '''
    Steps through primary smc sample() method in an easily testable fashion.
    '''

    def __init__(self):
        param_priors = self._set_param_priors()
        model = self._instance_model()
        data = self._generate_data(model)

        self.comm = MPI.COMM_WORLD.Clone()

        super(SMCTester, self).__init__(data, model, param_priors)

    @staticmethod
    def _set_param_priors():
        return {'a': ['Uniform', -10., 10.], 'b': ['Uniform', -10., 10.]}

    @staticmethod
    def _instance_model():
        x_space = np.arange(50)
        return Model(x_space)

    @staticmethod
    def _generate_data(model):
        std_dev = 0.6
        true_params = {'a': 2.5, 'b': 1.3}
        return model.generate_noisy_data_with_model(std_dev, true_params)

    def evaluate_model(model):
        return model.evaluate()

    def error_processing_args(model):
        return model.evaluate("test error")

    @staticmethod
    def cleanup_file(filename):
        try:
            remove(filename)
        except:
            pass
        return None

    @staticmethod
    def calc_log_like_manually(model_eval, data, std_dev):
        '''
        Assuming iid normally distributed errors, this function computes the
        likelihood given model evaluations at a specific param vector, data,
        and assumed noise standard deviation. This function is intended to test
        the pymc likelihood computation; i.e., results_rv.logp.
        '''
        M = len(data)
        var = std_dev**2
        diff = data - model_eval
        ssq = np.linalg.norm(diff)**2
        return np.log(1. / (2 * np.pi * var)**(M / 2.) * np.exp(-1. / (2 * var) * ssq))

    @classmethod
    def assert_step_lists_almost_equal(class_, pc1, pc2):
        assert len(pc1) == len(pc2)
        for i in range(len(pc1)):
            class_.assert_steps_almost_equal(pc1[i], pc2[i])
        return None

    @staticmethod
    def assert_steps_almost_equal(pc1, pc2):
        aae = np.testing.assert_array_almost_equal
        aae(pc1.get_log_likes(), pc2.get_log_likes())
        aae(pc1.get_weights(), pc2.get_weights())
        aae(pc1.get_params('a',), pc2.get_params('a',))
        aae(pc1.get_params('b',), pc2.get_params('b',))

    def when_proposal_dist_set_with_scales(self):
        proposal_center = {'a': 1, 'b': 2}
        proposal_scales = {'a': 0.5, 'b': 1}
        self._set_proposal_distribution(proposal_center, proposal_scales)
        self.expected_center = proposal_center
        self.expected_scales = proposal_scales
        return None

    def when_proposal_dist_set_with_no_scales(self):
        proposal_center = {'a': 1, 'b': 2}
        proposal_scales = None
        self._set_proposal_distribution(proposal_center, proposal_scales)
        self.expected_center = proposal_center
        self.expected_scales = {'a': 1, 'b': 1}
        return None

    def when_sampling(self, restart_time_step, hdf5_to_load, autosave_file):
        '''
        Perform SMC sampling with predefined sampling parameters.
        '''
        num_particles = self.comm.Get_size()
        num_time_steps = 2
        num_mcmc_steps = 1
        measurement_std_dev = 0.6
        ess_threshold = 0.8 * num_particles
        proposal_center = {'a': 2.0, 'b': 3.5}
        proposal_scales = {'a': 0.5, 'b': 0.5}
        self.sample(num_particles, num_time_steps, num_mcmc_steps,
                    measurement_std_dev, ess_threshold, proposal_center,
                    proposal_scales, restart_time_step, hdf5_to_load,
                    autosave_file)
        return None

    def when_sampling_parameters_set(self, num_time_steps=2,
                                     num_particles_per_processor=1,
                                     num_mcmc_steps=1, autosave_file=None,
                                     restart_time_step=0):
        '''
        Testing checkpoint. This returns an instance of the SMCSampler class
        that has initialized the sampler parameters and is preparing to
        initialize the particles.
        '''
        num_particles = self.comm.Get_size() * num_particles_per_processor
        ess_threshold = 0.8 * num_particles

        self.num_particles = num_particles
        self.num_time_steps = num_time_steps
        self.temp_schedule = np.linspace(0., 1., self.num_time_steps)
        self.num_mcmc_steps = num_mcmc_steps
        self.ess_threshold = ess_threshold
        self.autosaver = autosave_file
        self.restart_time_step = restart_time_step
        return None

    def when_initial_particles_sampled_from_proposal(self, measurement_std_dev):
        proposal_center = {'a': 2.0, 'b': 3.5}
        proposal_scales = {'a': 0.5, 'b': 0.5}

        self._set_proposal_distribution(proposal_center, proposal_scales)
        self._set_start_time_based_on_proposal()
        self.particles = self._initialize_particles(measurement_std_dev)
        return None

    def when_initial_particles_sampled_from_proposal_outside_prior(self):
        proposal_center = {'a': 1000.0, 'b': 3.5}
        proposal_scales = {'a': 1000.0, 'b': 0.5}

        self._set_proposal_distribution(proposal_center, proposal_scales)
        self._set_start_time_based_on_proposal()
        self.particles = self._initialize_particles(0.1)
        return None

    def when_initial_particles_sampled_from_prior(self, measurement_std_dev):
        proposal_center = None
        proposal_scales = None

        self._set_proposal_distribution(proposal_center, proposal_scales)
        self._set_start_time_based_on_proposal()
        self.particles = self._initialize_particles(measurement_std_dev)
        return None

    def when_step_created(self):
        self.when_initial_particles_sampled_from_proposal(0.6)
        particles = self.particles
        step = self._initialize_step(particles)
        if self.comm.Get_rank() == 0:
            step.fill_step(step.copy_step())
            self.step_list = [step]
            self.step = step
        else:
            self.step_list = None
            self.step = None
        return None

    def when_particles_mutated(self):
        temp_step = 0.2
        std_dev = 0.6

        new_particles = self._create_new_particles(temp_step)
        cov = np.eye(2)
        self.mutated_particles = self._mutate_new_particles(new_particles,
                                                            cov, std_dev,
                                                            temp_step)
        return None
