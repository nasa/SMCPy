import cupy
import numpy as np
import nvtx

from smcpy.utils import global_imports as gi


@cupy.fuse(kernel_name='fused_is_nan_kernel')
def fused_is_nan(output):
    return gi.num_lib.any(gi.num_lib.isnan(output))
 

@cupy.fuse(kernel_name='calc_normal_log_like_kernel')
def calc_normal_log_like(output, data, var):
    term1 = gi.num_lib.log(2 * gi.num_lib.pi * var)
    term2 = (output - data)**2 / var
    return gi.num_lib.sum((term1 + term2) * -0.5, axis=2)


class BaseLogLike:

    def __init__(self, model, data, args):
        self._model = model
        self._data = data
        self._args = args

    def _get_output(self, inputs):
        output = self._model(inputs)
        if fused_is_nan(output):
            raise ValueError
        return output


class Normal(BaseLogLike):

    def __init__(self, model, data, args=None):
        '''
        Likelihood function for data with additive, iid errors sampled from a
        normal distribution with mean = 0 and std_dev = args. If args is None,
        assumes that the last column of inputs contains the std_dev value.
        '''
        super().__init__(model, data, args)


    def __call__(self, inputs):
        with nvtx.annotate(message='convert inputs'):
            inputs = gi.num_lib.asarray(inputs)
            std_dev = gi.num_lib.expand_dims(inputs[:, :, -1], 2)
            inputs = inputs[:, :, :-1]
            var = std_dev ** 2

        with nvtx.annotate(message='call get_output'):
            output = self._get_output(inputs)

        nll = calc_normal_log_like(output, self._data, var)
        nll = gi.num_lib.expand_dims(nll, 2)

        with nvtx.annotate(message='get'):
            out_nll = np.empty((inputs.shape[0], inputs.shape[1], 1))
            if gi.USING_GPU:
                nll.get(out=out_nll) # makes asynch
            else:
                out_nll[:, :, -1] = nll
        return out_nll
