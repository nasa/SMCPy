from mpi4py import MPI
import numpy as np
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
        return {'a': ['Uniform', -5.0, 5.0], 'b': ['Uniform', -5.0, 5.0]}


    @staticmethod
    def _instance_model():
        x_space = np.arange(50)
        return Model(x_space)


    @staticmethod
    def _generate_data(model):
        std_dev = 0.6
        true_params = {'a': 2.0, 'b': 3.5}
        return model.generate_noisy_data_with_model(std_dev, true_params)


    @staticmethod
    def cleanup_file(filename):
        try:
            os.remove(filename)
        except:
            pass
        return None


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


    def when_sampling_parameters_set(self):
        '''
        Testing checkpoint. This returns an instance of the SMCSampler class
        that has initialized the sampler parameters and is preparing to
        initialize the particles.
        '''
        num_particles = self.comm.Get_size()
        num_time_steps = 2
        num_mcmc_steps = 1
        ESS_threshold = 0.8 * num_particles
        autosave_file = None

        self.num_particles = self._check_num_particles(num_particles)
        self.temp_schedule = self._check_temperature_schedule(num_time_steps)
        self.num_mcmc_steps = self._check_num_mcmc_steps(num_mcmc_steps)
        self.ESS_threshold = self._set_ESS_threshold(ESS_threshold)
        self._autosaver = self._set_autosave_behavior(autosave_file)
        return None


    def when_initial_particles_sampled_from_proposal(self):
        proposal_center = {'a': 2.0, 'b': 3.5}
        proposal_scales = {'a': 0.5, 'b': 0.5}
        measurement_std_dev = 0.6

        self.when_sampling_parameters_set()
        self._set_proposal_distribution(proposal_center, proposal_scales)
        self._set_start_time_based_on_proposal()
        self.particles = self._initialize_particles(measurement_std_dev)
        return None

