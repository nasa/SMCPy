import numpy as np
import pytest

from smcpy import MultiSourceNormal


def test_multisource_normal_bad_data_segments():
    inputs = np.ones((5, 6))
    data = np.ones(8)
    model = lambda x: np.ones((x.shape[0], data.shape[0])) * 2
    args = [(4, 3), (None, None)]

    with pytest.raises(ValueError):
        MultiSourceNormal(inputs, data, args)


@pytest.mark.parametrize("args_0,args_1", [((1, 7), (1, 1)),
                                           ((8, 0), (0.5, 0.5)),
                                           ((4, 4), (1, 0.5)),
                                           ((6, 2), (0.5, 1)),
                                           ((1, 2, 5), (2, 2, 2)),
                                           ((2, 2, 4), (2, 2, 2)),
                                           ((2, 2, 4), (1, 2, 3)),
                                           ((4, 0, 4), (1, 2, 0.5)),
                                           ((5, 1, 1, 1), (3, 2, 20, 1))])
def test_multisource_normal_fixed_std(args_0, args_1):
    inputs = np.ones((5, 6))
    data = np.ones(8)
    model = lambda x: np.ones((x.shape[0], data.shape[0])) * 2
    args = [args_0, args_1]

    segment_array = np.array(args_0)
    std_array = np.array(args_1)
    exp_term = sum([np.product(std_array ** 2) / (std_array[i] ** 2)
                    * segment_array[i] for i in range(len(std_array))])
    expected_log_like = np.ones(inputs.shape[0])
    expected_log_like *= (np.log(2 * np.pi) * -(len(data) / 2) + \
                         np.sum(np.log(std_array) * -segment_array)) + \
                         -1 / (2 * np.product(std_array ** 2)) * exp_term

    msn = MultiSourceNormal(model, data, args)
    log_like = msn(inputs)

    np.testing.assert_array_almost_equal(log_like, expected_log_like)


@pytest.mark.parametrize("args_1", [(None, None, None), (None, 2, 3),
                                    (1, None, 3), (1, 2, None),
                                    (None, 2, None), (None, None, 3)])
def test_multisource_normal_variable_std(args_1):
    inputs = np.ones((5, 6))
    data = np.ones(8)
    model = lambda x: np.ones((x.shape[0], data.shape[0])) * 2
    args = [(2, 2, 4), args_1]

    none_count = args_1.count(None)
    for i, std in enumerate(args_1):
        if std is None:
            inputs[:, -none_count + i] = i + 1

    expected_log_like = np.ones(inputs.shape[0]) * -14.604474003651934

    msn = MultiSourceNormal(model, data, args)
    log_like = msn(inputs)

    np.testing.assert_array_almost_equal(log_like, expected_log_like)
