'''
Notices:
Copyright 2018 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. No copyright is claimed in 
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
'''

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
