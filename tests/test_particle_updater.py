def test_update_log_weights(part_updater):
    temperature_step = 0.1
    exp_w = 0.2 - 0.2 * temperature_step
    part_updater.update_log_weights(temperature_step)
    assert all([exp_w == p.log_weight for p in
                part_updater.step.get_particles()])


def test_resample_if_needed_no(part_updater):
    part_updater.resample_if_needed()
    assert part_updater._resample_status == "No resampling"


def test_resample_if_needed_yes(part_updater_high_ess_threshold):
    part_updater_high_ess_threshold.resample_if_needed()
    assert part_updater_high_ess_threshold._resample_status == "Resampling..."
