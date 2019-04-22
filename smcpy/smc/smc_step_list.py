from ..hdf5.hdf5_storage import HDF5Storage


class SMCStepList():

    def __init__(self, step_list):
        self.step_list = step_list

    def save_step_list(self, h5_file):
        '''
        Saves self.step to an hdf5 file using the HDF5Storage class.

        :param hdf5_to_load: file path at which to save step list
        :type hdf5_to_load: string
        '''
        if self._rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='w')
            hdf5.write_step_list(self.step_list)
            hdf5.close()
        return None

    def load_step_list(self, h5_file):
        '''
        Loads and returns a step list stored using the HDF5Storage
        class.

        :param hdf5_to_load: file path of a step_list saved using the
            self.save_step_list() methods.
        :type hdf5_to_load: string
        '''
        if self._rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='r')
            step_list = hdf5.read_step_list()
            hdf5.close()
            print 'Step list loaded from %s.' % h5_file
        else:
            step_list = None
        return step_list
