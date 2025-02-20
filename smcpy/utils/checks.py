import numpy as np


class Checks(object):
    @classmethod
    def _is_dict(class_, input_):
        return isinstance(input_, dict)

    @classmethod
    def _is_1D_array(class_, input_):
        return input_.ndim != 1

    @classmethod
    def _is_integer_or_float(class_, input_):
        return any((class_._is_integer(input_), class_._is_float(input_)))

    @classmethod
    def _is_string_or_none(class_, input_):
        return any((class_._is_string(input_), class_._is_none(input_)))

    @staticmethod
    def _is_integer(input_):
        return isinstance(input_, int)

    @staticmethod
    def _is_float(input_):
        return isinstance(input_, float)

    @staticmethod
    def _is_string(input_):
        return isinstance(input_, str)

    @staticmethod
    def _is_none(input_):
        return input_ is None

    @staticmethod
    def _is_negative(input_):
        return input_ < 0

    @staticmethod
    def _is_positive(input_):
        return input_ > 0

    @staticmethod
    def _is_zero(input_):
        return input_ == 0

    @staticmethod
    def _is_positive_definite(input_):
        try:
            np.linalg.cholesky(input_)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def _raise_type_error(input_, type_):
        raise TypeError("%s must be %s." % (input_, type_))

    @staticmethod
    def _raise_negative_error(input_):
        raise ValueError("%s must be > 0." % input_)

    @staticmethod
    def _raise_zero_error(input_):
        raise ValueError("%s cannot be zero." % input_)

    @staticmethod
    def _raise_out_of_bounds_error(input_):
        raise ValueError("%s cannot be greater than num_time_steps." % input)
