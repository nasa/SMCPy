import abc
import numpy as np


class BaseModel(object):
    '''
    Abstract class defining the required functions to be used/modified
    by a smcpy user. 
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        '''
        User sets model constants in constructor
        '''


    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        '''
        User MUST redefine this function to return outputs for given inputs. 
        The *arg and **kwarg inputs are required for interfacing with the
        pymc module. At the beginning of the evaluate() method, call the class 
        method process_args(args, kwargs) to automatically handle this
        requirement; e.g.,

            param = self.process_args(args, kwargs)

        :param dictionary input_map: map from (string) input name to (float) 
            input value
        :return: 2D numpy array of outputs for each label at each time/location
            Size: len(self.labels) x len(self.times)*len(self.locations)
            Ordering convention is looping over times first then locations
        '''
        return None


    def process_args(self, args, kwargs):
        '''
        Converts args or kwargs into a dictionary mapping input names to their
        values.

        :return: input map (dictionary)
        '''
        if not args and kwargs:
            return kwargs
        elif args and type(args[0]) is dict:
            return args[0]
        else:
            self._raise_error_processing_args()
        return None


    def _raise_error_processing_args(self):
        msg = '%s.evaluate() accepts a single dictionary or keyword args.' \
               % self.__class__.__name__
        raise TypeError(msg)


    def generate_noisy_data_with_model(self, stdv, true_params):
        synth_data = self.evaluate(true_params)
        noisy_data = synth_data + np.random.normal(0, stdv, synth_data.shape)
        return noisy_data
