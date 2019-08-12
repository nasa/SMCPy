import numpy as np
import pytest


def test_initialize_particles_from_prior_fixed_std(part_initer):
    expected_loglikes = np.array([-3. / 2 * np.log(2 * np.pi)] * 5)
    expected_weights = expected_loglikes
    expected_keys = ['a', 'b']

    particles = part_initer.initialize_particles(measurement_std_dev=1.0,
                                                 num_particles=5)
    particle_param_vals = np.array([p.params.values() for p in particles])
    particle_loglikes = np.array([p.log_like for p in particles])
    particle_weights = np.array([p.log_weight for p in particles])

    np.testing.assert_array_equal(particle_loglikes, expected_loglikes)
    np.testing.assert_array_almost_equal(particle_weights, expected_weights)
    assert sorted(particles[0].params.keys()) == expected_keys


def test_initialize_particles_from_prior_rv_std(part_initer):
    expected_keys = ['a', 'b', 'std_dev']

    particles = part_initer.initialize_particles(measurement_std_dev=None,
                                                 num_particles=5)

    assert sorted(particles[0].params.keys()) == expected_keys


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
