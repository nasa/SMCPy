SMCPy - **S**equential **M**onte **C**arlo **S**ampling with **Py**thon 
==========================================================================
Python module for uncertainty quantification using a parallel sequential Monte
Carlo sampler.

To operate the code, the user supplies a computational model built in Python
2.7, defines prior distributions for each of the model parameters to be
estimated, and provides data to be used for calibration. SMC sampling can then
be conducted with ease through instantiation of the SMC class and a call to the
sample() method. The output of this process is an approximation of the parameter
posterior probability distribution conditional on the data provided.

This software was funded by and developed under the High Performance Computing 
Incubator (HPCI) at NASA Langley Research Center.

------------------------------------------------------------------------------
## Example Usage

```python
import numpy as np
from smcpy.examples.spring_mass.spring_mass_models import SpringMassModel
from smcpy.smc.smc_sampler import SMCSampler

# Initialize model
state0 = [0., 0.]                        #initial conditions
measure_t_grid = np.arange(0., 5., 0.2)  #time 
model = SpringMassModel(state0, measure_t_grid)

# Load data
noise_stddev = 0.94
displacement_data = np.genfromtxt('noisy_data.txt')

# Define prior distributions
param_priors = {'K': ['Uniform', 0.0, 10.0],
                'g': ['Uniform', 0.0, 10.0]}

# SMC sampling
num_particles = 5000
num_time_steps = 20
num_mcmc_steps = 1
smc = SMCSampler(displacement_data, model, param_priors)
pchain = smc.sample(num_particles, num_time_steps, num_mcmc_steps, noise_stddev,
                    ess_threshold=num_particles*0.5)
if smc._rank == 0:
    pchain.plot_pairwise_weights(save=True)
```

The above code produces probabilistic estimates of K, the spring stiffness divided by mass, and g, the gravitational constant on an unknown planet. These estimates are in the form of weighted particles and can be visualized by plotting the pairwise weights as shown below. The mean of each parameter is marked by the dashed red line. The true values for this example were K = 1.67 and g = 4.62. More details can be found in the spring mass example (smcpy/examples/spring_mass/).

![Pairwise](https://github.com/nasa/SMCPy/blob/master/examples/spring_mass/pairwise.png)

------------------------------------------------------------------------------

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

