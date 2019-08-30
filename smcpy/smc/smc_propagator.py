from .smc_step import SMCStep

class SMCPropagator():

    def __init__(self, model):
        '''
        :param model: predictive model to propagate particles through
        :type model: BaseModel object
        '''
        self._model = model

    def propagate(self, smc_step):
        '''
        Propagates particles in smc_step through self.model to obtain a new
        SMCStep object that stores the model outputs as params. This enables
        calculation of means, etc.

        :param smc_step: the smc step object to propagate (should contain
            particles with params == model parameters)
        :type smc_step: SMCStep object
        '''
        particle_list = smc_step.get_particles()
        for particle in particle_list:
            outputs = self._model.evaluate(particle.params)
            particle.params = {'output_{}'.format(i): out \
                               for i, out in enumerate(outputs)}
        new_step = SMCStep()
        new_step.set_particles(particle_list)
        return new_step
