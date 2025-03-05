import numpy as np
import pytest

from smcpy import MultiSourceNormal, MVNormal, MVNRandomEffects, Normal


@pytest.mark.parametrize(
    "class_, args",
    [(MultiSourceNormal, [(3,), (1,)]), (MVNormal, [0, 0, 0]), (Normal, 1)],
)
def test_nan_in_model_raises_value_error(class_, args):
    model = lambda x: np.array([[1, 2, np.nan], [np.nan, np.nan, 3]])
    data = np.ones(3)
    like = class_(model, data, args)

    with pytest.raises(ValueError):
        like(np.ones((1, 1)))


def test_multisource_normal_bad_data_segments():
    inputs = np.ones((5, 6))
    data = np.ones(8)
    model = lambda x: np.ones((x.shape[0], data.shape[0])) * 2
    args = [(4, 3), (None, None)]

    with pytest.raises(ValueError):
        MultiSourceNormal(inputs, data, args)


@pytest.mark.parametrize(
    "args_0,args_1",
    [
        ((1, 7), (1, 1)),
        ((8, 0), (0.5, 0.5)),
        ((4, 4), (1, 0.5)),
        ((6, 2), (0.5, 1)),
        ((1, 2, 5), (2, 2, 2)),
        ((2, 2, 4), (2, 2, 2)),
        ((2, 2, 4), (1, 2, 3)),
        ((4, 0, 4), (1, 2, 0.5)),
        ((5, 1, 1, 1), (3, 2, 20, 1)),
    ],
)
def test_multisource_normal_fixed_std(args_0, args_1):
    inputs = np.ones((5, 6))
    data = np.ones(8)
    model = lambda x: np.ones((x.shape[0], data.shape[0])) * 2
    args = [args_0, args_1]

    segment_array = np.array(args_0)
    std_array = np.array(args_1)
    exp_term = sum(
        [
            np.prod(std_array**2) / (std_array[i] ** 2) * segment_array[i]
            for i in range(len(std_array))
        ]
    )
    expected_log_like = np.ones(inputs.shape[0])
    expected_log_like *= (
        np.log(2 * np.pi) * -(len(data) / 2)
        + np.sum(np.log(std_array) * -segment_array)
    ) + -1 / (2 * np.prod(std_array**2)) * exp_term

    msn = MultiSourceNormal(model, data, args)
    log_like = msn(inputs)

    np.testing.assert_array_almost_equal(log_like, expected_log_like)


@pytest.mark.parametrize(
    "args_1",
    [
        (None, None, None),
        (None, 2, 3),
        (1, None, 3),
        (1, 2, None),
        (None, 2, None),
        (None, None, 3),
    ],
)
def test_multisource_normal_variable_std(mocker, args_1):
    num_samples = 5
    inputs = [1, 1, 1]
    data = np.ones(8)
    model = mocker.Mock(return_value=np.ones((num_samples, 8)) * 2)
    args = [(2, 2, 4), args_1]

    expected_model_inputs = [1, 1, 1]

    none_count = args_1.count(None)
    for i, std in enumerate(args_1):
        if std is None:
            inputs.append(i + 1)

    inputs = np.tile(inputs, (num_samples, 1))

    expected_log_like = np.ones(inputs.shape[0]) * -14.604474003651934
    expected_model_inputs = np.ones((5, 3))

    msn = MultiSourceNormal(model, data, args)
    log_like = msn(inputs)

    np.testing.assert_array_almost_equal(log_like, expected_log_like)
    np.testing.assert_array_equal(model.call_args_list[0][0][0], expected_model_inputs)


@pytest.mark.parametrize("n_snapshots", [1, 5, 10])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_mvn_likelihood_single_dimension(mocker, n_samples, n_snapshots):
    model = mocker.Mock(return_value=np.ones((n_samples, 1)))
    data = np.array([[3]] * n_snapshots)
    args = [np.sqrt(1 / (2 * np.pi)) ** 2]
    inputs = np.ones((n_samples, 2))

    expected_log_like = np.array([[-4 * np.pi]] * n_samples) * n_snapshots

    mvn = MVNormal(model, data, args)

    np.testing.assert_array_equal(mvn(inputs), expected_log_like)
    np.testing.assert_array_equal(model.call_args[0][0], inputs)


@pytest.mark.parametrize("n_snapshots", [1, 2, 5])
@pytest.mark.parametrize(
    "args",
    [
        (1, 2, 3, -1, 0, -2),
        (None, 2, 3, -1, 0, None),
        (1, None, None, None, 0, None),
        (None, None, None, None, None, None),
    ],
)
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_mvn_likelihood_ndim(mocker, n_samples, n_snapshots, args):
    n_data_pts = 3
    n_cov_terms = n_data_pts * (n_data_pts + 1) / 2
    input_vector = [1, 2]
    model_output = np.ones((n_samples, n_data_pts))

    base_args = (1, 2, 3, -1, 0, -2)
    for i, arg in enumerate(args):
        if arg is None:
            input_vector.append(base_args[i])

    inputs = np.tile(input_vector, (n_samples, 1))

    model = mocker.Mock(return_value=model_output)
    data = np.ones((n_snapshots, n_data_pts)) * 2

    expected_log_like = np.array([[-4.54482456]] * n_samples) * n_snapshots

    mvn = MVNormal(model, data, args)

    np.testing.assert_array_almost_equal(mvn(inputs), expected_log_like)
    np.testing.assert_array_equal(model.call_args[0][0], inputs[:, :2])


@pytest.mark.parametrize(
    "args",
    [
        ([1, 2, 3], [2, 3, 4, 5, 6]),
        ([None, 2, 3], [2, None, 4, 5, 6]),
        ([1, None, 3], [2, None, 4, None, None]),
        ([None, None, 3], [2, 3, None, 5, 6]),
        ([None] * 3, [None] * 5),
    ],
)
def test_mvnrandeff_likelihood(mocker, args):
    n_params = 2
    n_samples = 4
    n_randeff = len(args[1])
    n_cov_args = len(args[0])
    n_mvn_nones = args[0].count(None)
    n_norm_nones = args[1].count(None)
    n_nones = n_mvn_nones + n_norm_nones

    model_inputs = np.tile(np.arange(n_params * (n_randeff + 1)), (n_samples, 1))
    inputs = model_inputs.copy()
    if n_nones > 0:
        arg_inputs = np.ones((n_samples, n_nones))
        inputs = np.concatenate((model_inputs, arg_inputs), axis=1)

    data = np.ones((n_randeff, 1))
    model = mocker.Mock()

    norm_like = mocker.Mock(return_value=np.ones((n_samples, 1)) * 2)
    norm_like_class = mocker.Mock(return_value=norm_like)
    mocker.patch("smcpy.log_likelihoods.Normal", new=norm_like_class)

    mvn_like = mocker.Mock(return_value=np.ones((1, 1)) * 3)
    mvn_like_class = mocker.Mock(return_value=mvn_like)
    mocker.patch("smcpy.log_likelihoods.MVNormal", new=mvn_like_class)

    split_inputs = np.array_split(model_inputs, n_randeff + 1, axis=1)

    exp_mvn_inputs = split_inputs[0]
    exp_norm_inputs = split_inputs[1:]
    if n_nones > 0:
        mvn_args = np.ones((n_samples, n_mvn_nones))
        norm_args = np.ones((n_samples, n_norm_nones))
        exp_mvn_inputs = np.concatenate((exp_mvn_inputs, mvn_args), axis=1)
        exp_norm_inputs = [
            np.c_[in_, np.ones(n_samples)] if args[1][i] == None else in_
            for i, in_ in enumerate(exp_norm_inputs)
        ]

    expected_log_like = np.ones((n_samples, 1)) * (3 + 2 * n_randeff)

    mvnre = MVNRandomEffects(model, data, args)
    mvnre.set_model_wrapper(99)

    np.testing.assert_array_equal(mvnre(inputs), expected_log_like)

    # total effects likelihood calls
    for i in range(n_samples):
        in_ = np.array([exp_mvn_inputs[i] for exp_mvn_in in exp_mvn_inputs])
        np.testing.assert_array_equal(
            mvn_like.call_args_list[i][0][0], exp_mvn_inputs[i].reshape(1, -1)
        )

        te_data = np.array([model_in[i] for model_in in split_inputs[1:]])

        call = mvn_like_class.call_args_list[i][0]
        assert call[0] == mvnre._total_effects_model
        np.testing.assert_array_equal(call[1], te_data)
        assert call[2] == args[0]
        mvn_like.set_model_wrapper.assert_not_called()

    # random effects likelihood calls
    for i in range(n_randeff):
        call = norm_like_class.call_args_list[i][0]
        assert call[0] == model
        np.testing.assert_array_equal(call[1], data[i])
        assert call[2] == args[1][i]
        np.testing.assert_array_equal(
            norm_like.call_args_list[i][0][0], exp_norm_inputs[i]
        )
        norm_like.set_model_wrapper.assert_called_with(99)


def test_random_effects_dummy_model(mocker):
    model = mocker.Mock()
    data = [mocker.Mock()]
    args = ([mocker.Mock()], [mocker.Mock()])

    mvnre = MVNRandomEffects(model, data, args)

    np.testing.assert_array_equal(
        mvnre._total_effects_model(np.ones((5, 5))), np.ones((5, 5))
    )


def test_random_effects_multi_model(mocker):
    norm_mock = mocker.patch("smcpy.log_likelihoods.Normal")
    model = [0, 1, 2]
    args = ([1], [0, 1, 2])
    data = [0, 1, 2]

    mvnre = MVNRandomEffects(model, data, args)
    mvnre.set_model_wrapper(4)

    for i in model:
        assert norm_mock.call_args_list[i][0] == (i, i, i)
    assert norm_mock._model_wrapper.called_with(4)


def test_random_effects_multi_model_not_match_num_rand_eff():
    model = [1]
    data = []
    args = ([1], list(range(5)))

    with pytest.raises(ValueError):
        MVNRandomEffects(model, data, args)


def test_likelihood_model_wrapper(mocker):
    model = mocker.Mock(return_value=3)
    like = Normal(model, data=1, args=None)

    output1 = like._get_output(1)
    like.set_model_wrapper(lambda model, x: 2 * model(x))
    output2 = like._get_output(1)

    assert output1 == 3
    assert output2 == 6
