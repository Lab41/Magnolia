from hdf5_iterator import Hdf5Iterator 

class DNNfeeder:
    
    def __init_(self, iterators, shape=None, pos=None, seed=0 ):
        '''
        DNNfeeder: DNN Data Feeder 

        Args:
            iterator (list): a list of hdf5 file names or a list of iterators
            shape (tuple): dimension of slices to extract; if None, will yield
                the full size for each dataset in the HDF5 file
            pos (tuple or None): optionally, tuple of ints or None, the same
                length as shape. Where not None, slices will always be extracted
                from the given position on that dimension. If None, slices
                will be extracted from random positions in that dimension.
        '''
        if isinstance(iterators[0], str):
            self.iterator = []
            for h5file_name in iterators:
                self.iterators = Hdf5Iterator(h5file_name,shape=shape,pos=pos,seed=seed)
        else:
            self.iterators = iterators

    def get_sample(self, iterators=None):
        '''Get a single sample from all iterators in the list'''

        # If iterators to get samples from isn't specified, use our own
        if not iterators:
            iterators = self.iterators
        sample = ()
        for iterator in iterators:
            sample += iterator.next()
        return sample

    def get_batch(self, batch_size, iterators=None):
        '''Get a batch from iterators of size batch_size'''
     
        # If iterators  to get samples from isn't specified, use our own
        if not iterators:
            iterators = self.iterators  
        data_batch = []
        for i in range(batch_size):            
            self.get_sample(iterators)
            data_batch += [current_sample]
        return data_batch
            
    def sum_features(self, data_sample):
        '''Sum features'''
        return sum(data_sample)

    def convert2mfcc(self):
        return NotImplemented
