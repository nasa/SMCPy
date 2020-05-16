import numpy as np
import pytest

from smcpy.smc.mpi_base_class import MPIBaseClass


@pytest.mark.parametrize('rank,expected', [(0, 4), (1, 3), (2, 3)])
def test_get_num_particles_in_partition(stub_comm, rank, expected, mocker):
    mocker.patch.object(stub_comm, 'Get_size', return_value=3)
    mocker.patch.object(stub_comm, 'Get_rank', return_value=rank)
    mpi_base = MPIBaseClass(mpi_comm=stub_comm)
    assert mpi_base.get_num_particles_in_partition(10, rank) == expected


@pytest.mark.parametrize('rank,expected', [(0, [1, 2]), (1, [3, 4]), (2, [5])])
def test_partition_and_scatter_particles(stub_comm, rank, expected, mocker):
    mocker.patch.object(stub_comm, 'Get_size', return_value=3)
    mocker.patch.object(stub_comm, 'Get_rank', return_value=rank)
    mocker.patch.object(stub_comm, 'scatter', return_value=expected)
    if rank == 0:
        stub_comm.scatter = lambda x, root: x[rank]

    mpi_base = MPIBaseClass(mpi_comm=stub_comm)

    particles = [1, 2, 3, 4, 5]
    particles = mpi_base.partition_and_scatter_particles(particles)
    np.testing.assert_array_equal(particles, expected)


@pytest.mark.parametrize('rank,particles,gathered,expected',
                         [(0, [1, 2], [[1, 2], [3, 4], [5]], [1, 2, 3, 4, 5]),
                          (1, [3, 4], [], []),
                          (2, [5], [], [])])
def test_gather_and_concat_particles(stub_comm, rank, particles, gathered,
                                     expected, mocker):
    mocker.patch.object(stub_comm, 'Get_size', return_value=3)
    mocker.patch.object(stub_comm, 'Get_rank', return_value=rank)
    mocker.patch.object(stub_comm, 'gather', return_value=gathered)

    mpi_base = MPIBaseClass(mpi_comm=stub_comm)

    particles = mpi_base.gather_and_concat_particles(particles)
    np.testing.assert_array_equal(particles, expected)
