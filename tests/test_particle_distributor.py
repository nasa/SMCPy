import pytest
from smcpy.particles.particle_distributor import ParticleDistributor

@pytest.fixture(scope='module')
def comm():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        raise Exception
    except:
        class MockComm(): pass
        comm = MockComm()
        comm.Get_rank = lambda: 0
        comm.Get_size = lambda: 1
        comm.scatter = lambda array, root: [array]
        comm.gather = lambda array, root: array[0]
    return comm


@pytest.fixture()
def size(comm):
    return comm.Get_size()


@pytest.fixture
def particle_distributor(comm, filled_step):
    return ParticleDistributor(filled_step, comm)


def test_partition_particles(particle_distributor, size):
    partitions = particle_distributor.partition_particles()
    assert len(partitions) == size


def test_partition_and_scatter_particles(particle_distributor):
    particles = particle_distributor.partition_and_scatter_particles()
    assert len(particles)
