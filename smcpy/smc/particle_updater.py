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


def _mpi_decorator(func):
    def wrapper(self, *args, **kwargs):
        if self._rank == 0:
            func(self, *args, **kwargs)
    return wrapper


class ParticleUpdater():
    '''
    Class for updating particles at each step of Sequential Monte Carlo sampling
    with methods for updating log weights and resampling if ess under threshold
    '''

    def __init__(self, step, ess_threshold, mpi_comm=SingleRankComm()):
        self.step = step
        self.ess_threshold = ess_threshold
        self._comm = mpi_comm
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    #@_mpi_decorator
    def update_log_weights(self, temperature_step):
        '''
        Incrementally updates log weights depending on the likelihood and
        temperature step for each particle in a step

        :param temperature_step: change in temperature schedule between steps
        :type temperature_step: float
        '''
        for p in self.step.get_particles():
            temp_weight = p.log_weight
            p.log_weight = temp_weight + p.log_like * temperature_step
        return self.step

    #@_mpi_decorator
    def resample_if_needed(self):
        '''
        Checks if ess below threshold; if yes, resample with replacement.
        '''
        self._ess = self.step.compute_ess()
        if self._ess < self.ess_threshold:
            self._resample_status = "Resampling..."
            self.step.resample()
        else:
            self._resample_status = "No resampling"
        return self.step
