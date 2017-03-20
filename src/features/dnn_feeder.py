#!/usr/bin/env python

from wav_iterator import batcher
from hdf5_iterator import Hdf5Iterator

class DNNfeeder:

    def __init__(self, iterators, shape=None, pos=None, seed=0 ):
        '''
        DNNfeeder: DNN Data Feeder. Retrieve batches over
        examples from arbitrary numbers of HDF5 sources.

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
        sample = next(zip(*iterators))
        return sample

    def get_batch(self, batch_size, iterators=None):
        '''Get a batch from iterators of size batch_size

        Returns:
            tuple, length equal to the number of iterators in iterators or
            self.iterators
        '''

        # If iterators to get samples from isn't specified, use our own
        if not iterators:
             iterators = self.iterators
        batches = batcher(zip(*iterators), batch_size)
        data_batch = next(batches)
        return data_batch

    def sum_features(self, data_sample):
        '''Sum features'''
        return sum(data_sample)

    def convert2mfcc(self):
        raise NotImplementedError

if __name__ == "__main__":
    from features.hdf5_iterator import mock_hdf5
    mock_hdf5()
    h = Hdf5Iterator("._test.h5")
    d = DNNfeeder((h,))
    batch_size=10
    b = d.get_batch(batch_size)

    # Check that b has the right number of features
    assert len(b) == 1
    # Check the shape of the batch
    assert b[0].shape == (batch_size, 15, 20)
    # Make sure it is pulling different batches each time
    assert not (d.get_batch(10)[0] == d.get_batch(10)[0]).all()
