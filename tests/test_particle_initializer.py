import numpy as np
import pytest


def test_initialize_particles_from_prior(part_initer):
    particles = part_initer.initialize_particles(num_particles=5,
                                                 measurement_std_dev=1.0)

    particle_param_vals = np.array([p.params.values() for p in particles])
    particle_loglikes = np.array([p.log_like for p in particles])
    particle_weights = np.array([p.log_weight for p in particles])

    expected_loglikes = np.array([-3. / 2 * np.log(2 * np.pi)] * 5)
    expected_weights = expected_loglikes

    assert all([0. <= val <= 1. for val in particle_param_vals.flatten()])
    np.testing.assert_array_equal(particle_loglikes, expected_loglikes)
    np.testing.assert_array_equal(particle_weights, expected_weights)


def test_set_proposal_distribution_with_scales(part_initer):
    proposal_center = {'a': 1, 'b': 2}
    proposal_scales = {'a': 0.5, 'b': 1}
    part_initer.set_proposal_distribution(proposal_center, proposal_scales)
    assert part_initer.proposal_center == proposal_center
    assert part_initer.proposal_scales == proposal_scales


def test_set_proposal_distribution_with_no_scales(part_initer):
    proposal_center = {'a': 1, 'b': 2}
    part_initer.set_proposal_distribution(proposal_center)
    assert part_initer.proposal_center == proposal_center
    assert part_initer.proposal_scales == {'a': 1, 'b': 1}


def test_initialize_particles_with_proposals(part_initer):
    proposal_center = {'a': 1, 'b': 2}
    proposal_scales = {'a': 0.5, 'b': 1}
    part_initer.set_proposal_distribution(proposal_center, proposal_scales)
    with pytest.raises(Exception):
        particles = part_initer.initialize_particles(num_particles=5,
                                                     measurement_std_dev=1.0)
