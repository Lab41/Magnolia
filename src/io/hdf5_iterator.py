#!/usr/bin/env python
'''
hdf5_iterator.py

iterate over hdf5 files with one level of grouping in
random order, yielding randomly positioned slices of a given shape
'''
import logging
import copy
import h5py
import numpy as np

class Hdf5Iterator:
    def __init__(self, hdf5_path, shape=None, pos=None, seed=41):
        '''
        Args:
            hdf5_path (str): path to HDF5 file
            shape (tuple): dimension of slices to extract; if None, will yield
                the full size for each dataset in the HDF5 file
            pos (tuple or None): optionally, tuple of ints or None, the same
                length as shape. Where not None, slices will always be extracted
                from the given position on that dimension. If None, slices
                will be extracted from random positions in that dimension.

        Examples:
            Hdf5Iterator("foo.h5", shape=(None, 256), pos=(None, 0))
                returns an iterator over 2d data samples from foo.h5, where the first
                dimension of the sample is determined by the first dimension of the dataset,
                and the second dimension of the sample is always taken from 0:256 in the
                second dimension of the data.
        '''
        self.hdf5_path = hdf5_path
        self.h5 = h5py.File(hdf5_path, 'r')
        self.h5_groups = [key for key in self.h5]
        self.h5_items = []
        for group in self.h5_groups:
            self.h5_items += [ group + '/' + item for item in self.h5[group] ]
        self.rng = np.random.RandomState(seed)

        # Handle unspecified dimensionality for shape and pos
        if shape is None and pos is None:
            # if ndim is None:
            #     #raise ValueError("If shape and pos are both None, must specify ndim")
            #     ndim = 2
            #     # TODO: figure out if ndim = 1 is different from ndim = 2
            #     logger = logging.getLogger(__name__)
            #     logger.warning("Setting ndim to {} automatically".format(ndim))

            shape = (None,)

        if shape is None:
            self.shape = tuple(None for dim in pos)
        else:
            self.shape = shape

        logger = logging.getLogger(__name__)
        logger.debug("self.shape: {}".format(self.shape))

        if pos is None:
            self.pos = tuple(None for dim in self.shape)
        else:
            self.pos = pos

    def __next__(self):
        '''Randomly pick a dataset from the available options'''
        logger = logging.getLogger(__name__)
        num_tries = 500
        for i in range(num_tries):
            next_item = self.h5[self.rng.choice(self.h5_items)]
            logger.debug("next_item.shape: {}".format(next_item.shape))
            # Ensure that Nones in self.shape
            # will yield the maximum size on the given dimension
            shape = list(copy.copy(self.shape))
            # Cover case where neither shape nor pos specified
            if shape == [None]:
                shape = [None for dim in next_item.shape]

            # Replace nones in shape with dims from next_item
            for j, dim in enumerate(next_item.shape):
                if shape[j] is None:
                    shape[j] = dim
                    logger.debug("dim {}: {}".format(j , dim))
            logger.debug("shape: {}".format(shape))

            # fail if this slice is out of bounds
            if any([want_dim > have_dim for have_dim, want_dim in zip(next_item.shape,shape)]):
                continue

            # Choose a random, valid place for the slice
            # to be made and return it
            slices = []
            for have_dim, want_dim, want_pos in zip(next_item.shape, shape, self.pos):
                if want_pos is None:
                    slice_start = self.rng.randint(have_dim - want_dim + 1)
                else:
                    # TODO: warn if want_pos is out of bounds
                    slice_start = want_pos
                slice_end = slice_start + want_dim
                slices.append(slice(slice_start, slice_end))
            output_slice = next_item[tuple(slices)]
            logger.debug("slices: {}".format(slices))
            logger.debug("output_slice: {}".format(output_slice))
            assert output_slice.shape == tuple(shape), "Result shape {} does not match " \
                    "target shape {}".format(output_slice.shape, shape)
            return output_slice
        raise ValueError("Failed to find a slice. Slice size too big?")

    def __iter__(self):
        return self

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # make small test hdf5 object
    datasets = np.random.randn(5, 10, 15, 20)
    with h5py.File('._test.h5', mode='w') as f:
        key_names = list('abcde')
        for i, k in enumerate(key_names):
            grp = f.create_group(k)
            for j, dataset in enumerate(datasets[i]):
                grp.create_dataset(str(j), data=dataset)

    # Tests ##########
    # Sending a number on one dimension of shape and none on the rest
    # returns a slice where the size of the None dimension is drawn from the data
    h = Hdf5Iterator('._test.h5', (2, None))
    a = next(h)
    assert a.shape == (2, 20)

    # Sending 0 in shape returns slices with technically correct shape
    h = Hdf5Iterator('._test.h5', (0, None))
    a = next(h)
    assert a.shape == (0, 20)

    # Sending tuples of Nones for shape==sending None
    h = Hdf5Iterator('._test.h5', (None, None))
    a = next(h)
    assert a.shape == (15, 20)

    # Asking for too big a slice doesn't work
    try:
        fail_test = True
        h = Hdf5Iterator('._test.h5', (100, 100))
        a = next(h)
        assert a.shape == (15, 20)
    except ValueError:
        fail_test = False
    assert not fail_test

    # Sending two Nones returns the whole dataset each time
    h = Hdf5Iterator('._test.h5', None, None)
    a = next(h)
    assert a.shape == (15, 20)

    # Mixing tuples with nones with plan-old None works
    h = Hdf5Iterator('._test.h5', None, (0, None))
    a = next(h)
    assert a.shape == (15,20)

    # Shape specifications work with nones on both shape and pos
    h = Hdf5Iterator('._test.h5', (None,10), (0, None))
    a = next(h)
    assert a.shape == (15,10)

    # Getting vectors back works
    h = Hdf5Iterator('._test.h5', (None,1), (0, None))
    a = next(h)
    assert a.shape == (15,1)

    # And slices from the edges are possible
    h = Hdf5Iterator('._test.h5', (1,None), (14,))
    a = next(h)
    assert a.shape == (1,20)

    # Invalid positions result in bad slices
    try:
        fail_test = True
        h = Hdf5Iterator('._test.h5', (1,None), (15,))
        a = next(h)
    except AssertionError:
        fail_test = False
    assert not fail_test

    # Loop through infinite examples
    h = Hdf5Iterator('._test.h5', (2, 5))
    a = next(h)
    for a in h:
        print(a)
