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


from ..utils.single_rank_comm import SingleRankComm
from copy import copy
import numpy as np


class ParticleMutator():
    '''
    Class for mutating particles at each step of Sequential Monte Carlo sampling
    with the main `mutate_new_particles` method, which uses the MCMC kernal to
    determine the distribution along the temperature schedule path.
    '''

    def __init__(self, step, mcmc, num_mcmc_steps, mpi_comm=SingleRankComm()):
        self.step = step
        self._comm = mpi_comm
        self._mcmc = mcmc
        self.num_mcmc_steps = num_mcmc_steps
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    def mutate_particles(self, measurement_std_dev=None, temperature_step=1):
        '''
        Predicts next distribution along the temperature schedule path using
        the MCMC kernel.

        :param measurement_std_dev: standard deviation of the measurement error;
            if unknown, set to None and it will be sampled along with other
            model parameters.
        :type measurement_std_dev: float or None
        :param temperature_step: difference in temp schedule between steps
        :type temperature_step: float

        :Returns: An SMCStep class instance that contains all particles after
            mutation.
        '''
        covariance = self._compute_step_covariance()
        particles = self._partition_and_scatter_particles()
        mcmc = copy(self._mcmc)
        step_method = 'smc_metropolis'
        new_particles = []
        mutation_count = 0
        for particle in particles:

            if measurement_std_dev is None:
                std_dev0 = particle.params['std_dev']
            else:
                std_dev0 = measurement_std_dev

            mcmc.generate_pymc_model(fix_var=bool(measurement_std_dev),
                                     std_dev0=std_dev0, q0=particle.params)

            mcmc.sample(self.num_mcmc_steps, burnin=0, step_method=step_method,
                        cov=covariance, verbose=-1, phi=temperature_step)
            stochastics = mcmc.MCMC.db.getstate()['stochastics']
            params = {key: stochastics[key] for key in particle.params.keys()}

            if particle.params != params:
                mutation_count += 1

            particle.params = params
            particle.log_like = mcmc.MCMC.logp
            new_particles.append(particle)

        new_particles = self._gather_and_concat_particles(new_particles)
        self._mutation_ratio = float(mutation_count) / len(particles)
        self.step = self._update_step_with_new_particles(new_particles)
        return self.step

    def _gather_and_concat_particles(self, new_particles):
        new_particles = self._comm.gather(new_particles, root=0)

        if self._rank == 0:
            new_particles = list(np.concatenate(new_particles))

        return new_particles

    def _update_step_with_new_particles(self, particles):
        if self._rank == 0:
            self.step.set_particles(particles)
        else:
            self.step = None
        return self.step

    def _partition_and_scatter_particles(self):
        if self._rank == 0:
            particles = self._partition_new_particles()
        else:
            particles = []
        particles = self._comm.scatter(particles, root=0)
        return particles

    def _partition_new_particles(self):
        partitions = np.array_split(self.step.get_particles(),
                                    self._size)
        return partitions

    def _compute_step_covariance(self):
        if self._rank == 0:
            covariance = self.step.get_covariance()
        else:
            covariance = None
        covariance = self._comm.scatter([covariance] * self._size, root=0)
        return covariance
