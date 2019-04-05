'''
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.

Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRessED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNess FOR A PARTICULAR PURPOSE, OR
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
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLess THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
'''

import os
import warnings
from tqdm import tqdm


from copy import copy

import numpy as np

from mpi4py import MPI
from ..mcmc.mcmc_sampler import MCMCSampler
from ..smc.smc_step import SMCStep
from ..hdf5.hdf5_storage import HDF5Storage
from ..utils.properties import Properties
from particle_initializer import ParticleInitializer
from particle_updater import ParticleUpdater


class SMCSampler(Properties):
    '''
    Class for performing parallel Sequential Monte Carlo sampling
    '''

    def __init__(self, data, model, param_priors):
        self._comm, self._size, self._rank = self._setup_communicator()
        self._mcmc = self._setup_mcmc_sampler(data, model, param_priors)
        self.parameter_names = param_priors.keys()
        super(SMCSampler, self).__init__()

    @staticmethod
    def _setup_communicator():
        comm = MPI.COMM_WORLD.Clone()
        size = comm.Get_size()
        my_rank = comm.Get_rank()
        return comm, size, my_rank

    @staticmethod
    def _setup_mcmc_sampler(data, model, param_priors):
        mcmc = MCMCSampler(data=data, model=model, params=param_priors,
                           storage_backend='ram')
        return mcmc

    def sample(self, num_particles, num_time_steps, num_mcmc_steps,
               measurement_std_dev, ess_threshold=None, proposal_center=None,
               proposal_scales=None, restart_time_step=0, hdf5_to_load=None,
               autosave_file=None):
        '''
        :param num_particles: number of particles to use during sampling
        :type num_particles: int
        :param num_time_steps: number of time steps in temperature schedule that
            is used to transition between prior and posterior distributions.
        :type num_time_steps: int
        :param num_mcmc_steps: number of mcmc steps to take during mutation
        :param num_mcmc_steps: int
        :param measurement_std_dev: standard deviation of the measurement error
        :type measurement_std_dev: float
        :param ess_threshold: threshold equivalent sample size; triggers
            resampling when ess > ess_threshold
        :type ess_threshold: float or int
        :param proposal_center: initial parameter dictionary, which is used to
            define the initial proposal distribution when generating particles;
            default is None, and initial proposal distribution = prior.
        :type proposal_center: dict
        :param proposal_scales: defines the scale of the initial proposal
            distribution, which is centered at proposal_center, the initial
            parameters; i.e. prop ~ MultivarN(q1, (I*proposal_center*scales)^2).
            Proposal scales should be passed as a dictionary with keys and
            values corresponding to parameter names and their associated scales,
            respectively. The default is None, which sets initial proposal
            distribution = prior.
        :type proposal_scales: dict
        :param restart_time_step: time step at which to restart sampling;
            default is zero, meaning the sampling process starts at the prior
            distribution; note that restart_time_step < num_time_steps. The
            step at restart_time is retained, and the sampling begins at the
            next step (t=restart_time_step+1).
        :type restart_time_step: int
        :param hdf5_to_load: file path of a step list
        :type hdf5_to_load: string


        :Returns: A list of SMCStep class instances that contains all particles
        and their past generations at every time step.
        '''

        self.ess_threshold = ess_threshold
        self.autosaver = autosave_file
        self.restart_time_step = restart_time_step
        self.temp_schedule = np.linspace(0., 1., num_time_steps)
        start_time_step = 1
        if self.restart_time_step == 0:
            initializer = ParticleInitializer(self._mcmc, self.temp_schedule,
                                              self._comm)
            initializer.set_proposal_distribution(proposal_center, proposal_scales)
            particles = initializer.initialize_particles(measurement_std_dev,
                                                         num_particles)
            step = self._initialize_step(particles)
            self.step = step
            self.step_list = [step]

        elif 0 < self.restart_time_step <= num_time_steps:
            start_time_step = restart_time_step
            step_list = self.load_step_list(hdf5_to_load)
            step_list = self._trim_step_list(step_list,
                                             self.restart_time_step)
            step = step_list[-1]
            self.step = step
            self.step_list = step_list

        updater = ParticleUpdater(step, self._comm)
        self._autosave_step()
        p_bar = tqdm(range(num_time_steps)[start_time_step + 1:])
        last_ess = 0

        for t in p_bar:
            temperature_step = self.temp_schedule[t] - self.temp_schedule[t - 1]
            new_particles = updater.update_particles(temperature_step)
            covariance = self._compute_step_covariance()
            mutated_particles = self.mutate_new_particles(new_particles,
                                                          covariance,
                                                          measurement_std_dev,
                                                          temperature_step)
            self._update_step_with_new_particles(mutated_particles)
            self.step_list.append(self.step.copy())
            self._autosave_step()
            p_bar.set_description("Step number: {:2d} | Last ess: {:8.2f} | "
                                  "Current ess: {:8.2f} | Samples accepted: "
                                  "{:.1%} | {} | "
                                  .format(t + 1, last_ess, updater._ess,
                                          self._acceptance_ratio,
                                          updater._resample_status))
            last_ess = updater._ess

        self._close_autosaver()
        return self.step_list

    def _initialize_step(self, particles):
        particles = self._comm.gather(particles, root=0)
        if self._rank == 0:
            step = SMCStep()
            step.set_particles(np.concatenate(particles))
            step.normalize_step_weights()
        else:
            step = None
        return step

    def _trim_step_list(self, step_list, restart_time_step):
        if self._rank == 0:
            to_keep = range(0, restart_time_step + 1)
            trimmed_steps = [step_list[i].get_particles() for i in to_keep]
            step_list = trimmed_steps
        return step_list

    @staticmethod
    def _file_exists(hdf5_file):
        return os.path.exists(hdf5_file)

    def _compute_step_covariance(self):
        if self._rank == 0:
            covariance = self.step.calculate_covariance()
        else:
            covariance = None
        covariance = self._comm.scatter([covariance] * self._size, root=0)
        return covariance

    def mutate_new_particles(self, particles, covariance, measurement_std_dev,
                             temperature_step):
        '''
        Predicts next distribution along the temperature schedule path using
        the MCMC kernel.
        '''
        mcmc = copy(self._mcmc)
        step_method = 'smc_metropolis'
        new_particles = []
        acceptance_count = 0
        if self._size == 1:
            particles = particles[0]
        for particle in particles:
            mcmc.generate_pymc_model(fix_var=True, std_dev0=measurement_std_dev,
                                     q0=particle.params)
            mcmc.sample(self.num_mcmc_steps, burnin=0, step_method=step_method,
                        cov=covariance, verbose=-1, phi=temperature_step)
            stochastics = mcmc.MCMC.db.getstate()['stochastics']
            params = {key: stochastics[key] for key in particle.params.keys()}
            if particle.params != params:
                acceptance_count += 1
            particle.params = params
            particle.log_like = mcmc.MCMC.logp
            new_particles.append(particle)
        new_particles = self._comm.gather(new_particles, root=0)
        self._acceptance_ratio = float(acceptance_count) / len(particles)
        # new list of accepted particles

        if self._rank == 0:
            return list(np.concatenate(new_particles))
            # return the other list
        return new_particles

    def _update_step_with_new_particles(self, particles):
        if self._rank == 0:
            self.step.set_particles(particles)
        return None

    def _autosave_step(self):
        if self._rank == 0 and self._autosaver is not None:
            step_index = len(self.step_list) - 1
            self.autosaver.write_step(self.step, step_index)
        return None

    def _close_autosaver(self):
        if self._rank == 0 and self._autosaver is not None:
            self.autosaver.close()
        return None

    def save_step_list(self, h5_file):
        '''
        Saves self.step to an hdf5 file using the HDF5Storage class.

        :param hdf5_to_load: file path at which to save step list
        :type hdf5_to_load: string
        '''
        if self._rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='w')
            hdf5.write_step_list(self.step_list)
            hdf5.close()
        return None

    def load_step_list(self, h5_file):
        '''
        Loads and returns a step list stored using the HDF5Storage
        class.

        :param hdf5_to_load: file path of a step_list saved using the
            self.save_step_list() methods.
        :type hdf5_to_load: string
        '''
        if self._rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='r')
            step_list = hdf5.read_step_list()
            hdf5.close()
            print 'Step list loaded from %s.' % h5_file
        else:
            step_list = None
        return step_list
