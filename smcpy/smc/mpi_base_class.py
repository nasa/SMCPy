import numpy as np


class MPIBaseClass:

    def __init__(self, mpi_comm):
        self._comm = mpi_comm
        self._size = mpi_comm.Get_size()
        self._rank = mpi_comm.Get_rank()

    def get_num_particles_in_partition(self, total_num_particles, rank):
        '''
        Get number of particles in partition based on MPI rank and total
        number of particles.

        :param total_num_particles: number of particles accross all ranks
        :type total_num_particles: int
        :param rank: MPI rank
        :type rank: int
        :returns: number of particles in partition for given rank
        '''
        num_particles_for_partition = int(total_num_particles / self._size)
        remainder = total_num_particles % self._size
        overtime_ranks = range(remainder)
        if rank in overtime_ranks:
            num_particles_for_partition += 1
        return num_particles_for_partition

    def partition_and_scatter_particles(self, particles):
        '''
        Partitions particles and scatters them to available MPI processes.

        :param particles: particles to be partitioned and scattered
        :type particles: list or array
        :returns: partition of particles for each mpi process
        '''
        # MARKED FOR REFACTOR WITH NEW PARTICLES OBJECT
        #if self._rank == 0:
        #    particles = np.array_split(particles, self._size)
        #else:
        #    particles = []
        #particles = self._comm.scatter(particles, root=0)
        return particles

    def gather_and_concat_particles(self, particles):
        '''
        Gathers a particles from all MPI processes and concatenates them on
        rank 0.

        :param particles: list of particles on rank
        :type particles: list or array
        :returns: concatenated list of particles (rank 0) empty list (all
                  other ranks)
        '''
        # MARKED FOR REFACTOR WITH NEW PARTICLES OBJECT
        #particles = self._comm.gather(particles, root=0)
        #if self._rank == 0:
        #    particles = list(np.concatenate(particles))
        return particles
