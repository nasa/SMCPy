from .smc_step import SMCStep

class SMCPropagator():

    def __init__(self, model, output_names=None):
        '''
        :param model: predictive model to propagate particles through
        :type model: BaseModel object
        '''
        self._model = model
        self._output_names = output_names

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
            particle.params = self._create_output_dictionary(outputs)

        new_step = SMCStep()
        new_step.set_particles(particle_list)
        return new_step

    def _create_output_dictionary(self, outputs):
        output_length = len(outputs)
        if self._output_names is None:
            output_names = ['output_{}'.format(i) for i in range(output_length)]
        else:
            self._check_len_of_output_names(output_length)
            output_names = self._output_names
        return {output_names[i]: out for i, out in enumerate(outputs)}

    def _check_len_of_output_names(self, output_length):
        if len(self._output_names) < output_length:
            raise ValueError('not enough names in output_names for output ' + \
                             'array of length {}'.format(output_length))
