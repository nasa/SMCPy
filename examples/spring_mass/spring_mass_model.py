from smcpy.model.base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#------------------------------------------------------
#Helper function to use scipy integrator in model class 
def mass_spring(state, t, K, g):
    '''
    Return velocity/acceleration given velocity/position and values for 
    stiffness and mass
    '''

    # unpack the state vector
    x = state[0]
    xd = state[1]

    # compute acceleration xdd
    xdd = -K*x + g

    # return the two state derivatives
    return [xd, xdd]


#------------------------------------------------------

class SpringMassModel(BaseModel):
    '''
    Defines Spring Mass model with 2 free params (spring stiffness, k & mass, m)
    '''
    def __init__(self, state0=None, time_grid=None):
    
        #Give default initial conditions & time grid if not specified
        if state0 is None:
            state0 = [0.0, 0.0]
        if time_grid is None:
            time_grid = np.arange(0.0, 10.0, 0.1)

        self._state0 = state0
        self._t = time_grid
        
    def evaluate(self, *args, **kwargs):
        '''
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid    
        '''
        params = self.process_args(args, kwargs)
        K = params['K']
        g = params['g']
        results = odeint(mass_spring, self._state0, self._t, args=(K, g))
        #acc = -K*results[:, 0]+g
        #return acc
        return results[:, 0]


if __name__ == '__main__':

    k = 2.5 # Newtons per metre
    m = 1.5 # Kilograms
    g = 9.8
    state0 = [0.0, 0.0]  #initial conditions
    t = np.arange(0.0, 10.0, 0.1)  #time grid for simulation

    #Initialize model & simulate 
    model = SpringMassModel(state0, t)
    state = model.evaluate(K=k/m, g=g)

    #plot results
    plt.figure()
    plt.plot(t, state)
    plt.xlabel('TIME (sec)')
    plt.ylabel('States')
    plt.title('Mass-Spring System')
    plt.legend(('$x$ (m)', '$\dot{x}$ (m/sec)'))
    plt.show()
