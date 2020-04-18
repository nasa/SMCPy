import pytest

from smcpy.mcmc.step_method_base import SMCStepMethod


def test_phi_set():
    step_method = SMCStepMethod(phi=1)
    assert step_method.phi == 1


@pytest.mark.parametrize('phi', [-10, -0.1, 1.1, 10])
def test_bad_phi(phi):
    with pytest.raises(ValueError):
        SMCStepMethod(phi=phi)
