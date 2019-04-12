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

from copy import deepcopy


class Particle():
    '''
    Class defining data structure of an SMC particle (a member of an SMC
    particle chain).
    '''

    def __init__(self, params, log_weight, log_like):
        '''
        :param params: parameters associated with particle; keys = parameter
            name and values = parameter value.
        :type params: dictionary
        :param weight: the computed weight of the particle
        :type weight: float or int
        :param log_like: the log likelihood of the particle
        :type log_like: float or int
        '''
        self.params = self._check_params(params)
        self.log_weight = self._check_log_weight(log_weight)
        self.log_like = self._check_log_like(log_like)

    def print_particle_info(self):
        '''
        Prints particle parameters, weight, and log likelihood to screen.
        '''
        info = (self.params, self.weight, self.log_like)
        print('params = %s\nlog_weight = %s\nlog_like = %s' % info)
        return None

    def copy(self):
        '''
        Returns a deep copy of self.
        '''
        return deepcopy(self)

    @staticmethod
    def _check_params(params):
        if type(params) is not dict:
            raise TypeError('Input "params" must be a dictionary.')
        return params

    @staticmethod
    def _check_log_weight(log_weight):
        if not isinstance(log_weight, int) and not isinstance(log_weight, float):
            raise TypeError('Input "log_weight" must be an integer or float')
        if log_weight < 0:
            raise ValueError('Input "log_weight" must be positive.')
        return log_weight

    @staticmethod
    def _check_log_like(log_like):
        if not isinstance(log_like, float) and not isinstance(log_like, int):
            raise TypeError('Input "log_like" must be an integer or float')
        if log_like > 0:
            raise ValueError('Input "log_like" must be negative.')
        return log_like
