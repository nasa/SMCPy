import h5py

def log_hdf_file(hdf_file):
    """
    Print the groups, attributes and datasets contained in the given HDF file handler to stdout.

    :param h5py.File hdf_file: HDF file handler to log to stdout.
    """
    def _print_item(name, item):
        """Print to stdout the name and attributes or value of the visited item."""
        print name
        # Format item attributes if any
        if item.attrs:
            print '\tattributes:'
            for key, value in item.attrs.iteritems():
                print '\t\t{}: {}'.format(key, str(value).replace('\n', '\n\t\t'))

        # Format Dataset value
        if hasattr(item, 'value'):
            print '\tValue:'
            print '\t\t' + str(item.value).replace('\n', '\n\t\t')

    # Here we first print the file attributes as they are not accessible from File.visititems()
    _print_item(hdf_file.filename, hdf_file)
    # Print the content of the file
    hdf_file.visititems(_print_item)


with h5py.File('restart.hdf5') as hdf_file:
    log_hdf_file(hdf_file)


