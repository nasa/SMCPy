class Checks(object):


    @classmethod
    def _is_integer_or_float(class_, input_):
        return any(class_.is_integer(input_), class_._is_float(input_))


    @classmethod
    def _is_string_or_none(class_, input_):
        return any(class_.is_string(input_), class_._is_none(input_))


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
        return input_ >= 0


    @staticmethod
    def _raise_type_error(input_, type_):
        raise TypeError('%s must be %s.' % (input_, type_))


    @staticmethod
    def _raise_negative_error(input_):
        raise ValueError('%s must be > 0.' % input_)
