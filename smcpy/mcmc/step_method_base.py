import abc


# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()}) 
class SMCStepMethod(ABC):

    def __init__(self, phi):
        self.phi = phi

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi > 1. or phi < 0:
            raise ValueError('Value of phi must be between 0 and 1')
        self._phi = phi

