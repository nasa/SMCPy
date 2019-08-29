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

import numpy as np


class ParticleDistributor():
    '''
    Manages partitioning, scattering, and gathering of particles for a given
    particle step in an MPI operating environment.
    '''

    def __init__(self, step, mpi_comm):
        self._comm = mpi_comm
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        self._step = step

    def partition_particles(self):
        '''
        Partition particles according to self._size (available number of
        mpi processes). Performs this operation on all mpi processes for
        which it is called.

        :returns: list of partitions (where partition is a list of particles)
        '''
        partitions = np.array_split(self._step.get_particles(),
                                    self._size)
        return partitions

    def partition_and_scatter_particles(self):
        '''
        Partitions particles and scatters them to all available mpi processes.

        :returns: partition of particles for each mpi process
        '''
        if self._rank == 0:
            particles = self.partition_particles()
        else:
            particles = []
        particles = self._comm.scatter(particles, root=0)
        return particles

    def gather_and_concat_particles(self, particles):
        '''
        Gathers a concatenated list of all particles from all mpi processes
        to rank 0.

        :returns: concatenated list of particles (rank 0)
                  empty list (all other ranks)
        '''
        new_particles = self._comm.gather(particles, root=0)

        if self._rank == 0:
            particles = list(np.concatenate(particles))

        return particles
