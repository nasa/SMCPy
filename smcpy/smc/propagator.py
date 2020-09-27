from .particles import Particles

class Propagator():

    def propagate(self, model, particles, output_names=None):
        '''
        Propagates particles through model to obtain a new particles object
        that stores the model outputs as params. This enables calculation of
        means, etc.

        :param model: computational model that is a function of particles.params
        :type model: callable
        :param particles: contains input parameter particle information
        :type particles: Particles object
        :param output_names: optional names of output dimensions
        :type output_names: list of str
        '''
        outputs = model(particles.params)
        outputs = self._create_output_dictionary(outputs, output_names)
        return Particles(outputs, particles.log_likes, particles.log_weights)

    def _create_output_dictionary(self, outputs, output_names):
        num_outputs = outputs.shape[1]
        if output_names is None:
            output_names = [f'y{i}' for i in range(num_outputs)]
        self._check_output_names(num_outputs, output_names)
        return {k: outputs[:, i].flatten() for i, k in enumerate(output_names)}

    def _check_output_names(self, num_outputs, output_names):
        if len(output_names) != num_outputs:
            raise ValueError('Number of output names does not equal the number'
                             'of outputs.')
