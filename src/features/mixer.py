#!/usr/bin/env python

import logging

import numpy as np

from .wav_iterator import batcher
from .hdf5_iterator import Hdf5Iterator

class FeatureMixer:

    def __init__(self, iterators, mix_method='sum', shape=None, pos=None, seed=0,
                 diffseed=True, return_key=False ):
        '''
        FeatureMixer: Mixes feature iterators together, yielding an
        iterator over both the original input iterators and the mixed
        output.

        Args:
            iterators (list): a list of hdf5 file names or a list of iterators
            mix_method (str): how to mix features together. Supported methods:
		        'sum': add features together over the first axis
                'ident': do not modify inputs
            shape (tuple): dimension of slices to extract; if None, will yield
                the full size for each dataset in the HDF5 file
            pos (tuple or None): optionally, tuple of ints or None, the same
                length as shape. Where not None, slices will always be extracted
                from the given position on that dimension. If None, slices
                will be extracted from random positions in that dimension.

        Optional Arguments:
            seed: (int) random seed to initialize each iterator (default 0)
            diffseed: (bool) do you want a different random seed for each iterator?

        Iterator yields:
            tuple (mix, *features), total length equal to length of iterators + 1,
                and mix is the sum of the input features
        '''
        self.iterators = []
        self.return_key = return_key
        for i, iterator in enumerate(iterators):
            if isinstance(iterator, str):
                # Only add to the seed if `diffseed` is true
                iseed = seed + int(diffseed)*i
                self.iterators.append(Hdf5Iterator(iterator,shape=shape,pos=pos,seed=iseed, return_key=return_key))
            else:
                self.iterators.append(iterator)
        self.mix_method = mix_method

    def __next__(self):
        next_example = next(zip(*self.iterators))
        logger = logging.getLogger(__name__)

        if self.mix_method == 'sum' or self.mix_method == 'add':
            if self.return_key:
<<<<<<< HEAD
                mixed_example = np.array(list(zip(*next_example))[1]).sum(axis=0)
=======
                # If we're returning keys, then the iterator returns a tuple, where the
                # first element is the key itself, the second element is the data. Thus,
                # the mixed sample will be the sum next_example[1]. 
                mixed_example = np.array( list( zip( *next_example ) )[1] ).sum(axis=0)
>>>>>>> 7018b4bccbbae042db796cec0b6225639025d413
            else:
                mixed_example = np.sum(np.array(next_example), axis=0)
            return (mixed_example, *next_example)
        elif self.mix_method=='ident':
            return next_example
        else:
            raise ValueError("Invalid mix_method: '{}'".format(mix_method))


    def __iter__(self):
        return self

    def get_batch(self, batchsize=32):

        batches = ()
        for iterator in self.iterators:
            batches += (iterator.get_batch(batchsize),)

        # Resultant size of `mixed` is the same as any iterator's batch size as it is the
        # sum of all the signals together.
        if self.mix_method=='sum' or self.mix_method=='add':
            mixed = np.sum( ibatch[1] for ibatch in batches )
            batches = (mixed,)+batches

        return batches




if __name__ == "__main__":
    # To install and test, try: pip install -U --no-deps .; python -m src.features.mixer
    # Some tests and examples
    import timeit
    from magnolia.features.hdf5_iterator import mock_hdf5
    mock_hdf5()
    h = Hdf5Iterator("._test.h5")
    d = FeatureMixer((h,'._test.h5'))
    mix = next(d)


    logger = logging.getLogger(__name__)
    logger.debug("Acquired a sample")
    # Check that b has the right number of features
    assert len(mix) == 3

    # Check that summing sums
    d = FeatureMixer((h,h), mix_method='sum')
    mix, h1, h2 = next(d)
    assert np.sum(h1+h2) == np.sum(mix)

    # Check that one example is different from the next
    assert np.sum(mix - next(d)[0]) != 0

    # Try batching
    # Use larger mock-up
    del h; del d; mock_hdf5(scale = 10)
    d = FeatureMixer([Hdf5Iterator('._test.h5', (1,20), return_key=True, seed=41) for i in range(6)], return_key=True)
    # Timing
    nbatches = 10
    batchsize = 1024
    # Time get_batch
    times = timeit.timeit(lambda: d.get_batch(batchsize=batchsize), number=nbatches)
    print("Test get_batch:\nIn {} batches of {}, avg {:0.2f} secs per batch".format(nbatches, batchsize, times/nbatches))
    # Time batcher
    from magnolia.features.wav_iterator import batcher
    b = batcher(d, batchsize, return_key=True)
    times = timeit.timeit(lambda: type(next(b)), number=nbatches)
    print("batcher")
    print("In {} batches of {}, avg {:0.2f} secs per batch".format(nbatches, batchsize, times/nbatches))
    c = next(b)
    print("Batch type:", type(c))
    print("Type of 'columns' in batch:", [type(x) for x in c])
    print("First dim of 'columns' in batch:", [len(x) for x in c])
    print("Done")
