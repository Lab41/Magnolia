#!/usr/bin/env python
'''
hdf5_iterator.py

iterate over hdf5 files with one level of grouping in
random order, yielding randomly positioned slices of a given shape
'''
import copy
import h5py
import numpy as np

class Hdf5Iterator:
    def __init__(self, hdf5_path, shape, seed=41):
        self.hdf5_path = hdf5_path
        self.h5 = h5py.File(hdf5_path, 'r')
        self.h5_groups = [key for key in self.h5]
        self.h5_items = []
        for group in self.h5_groups:
            self.h5_items += [ group + '/' + item for item in self.h5[group] ]
        self.rng = np.random.RandomState(seed)
        self.shape = shape

    def __next__(self):
        '''Randomly pick a dataset from the available options'''

        num_tries = 500
        for i in range(num_tries):
            next_item = self.h5[self.rng.choice(self.h5_items)]

            # Ensure that Nones is self.shape
            # will yield the maximum size on the given dimension
            shape = list(copy.copy(self.shape))
            for j, dim in enumerate(shape):
                if dim is None:
                    shape[j] = next_item.shape[j]

            # fail if this slice is out of bounds
            if any([want_dim > have_dim for have_dim, want_dim in zip(next_item.shape,shape)]):
                continue

            # Choose a random, valid place for the slice
            # to be made and yield it
            slices = []
            for have_dim, want_dim in zip(next_item.shape,shape):
                slice_start = self.rng.randint(have_dim - want_dim + 1)
                slice_end = slice_start + want_dim
                slices.append(slice(slice_start, slice_end))
            output_slice = next_item[tuple(slices)]
            return output_slice
        raise ValueError("Failed to find a slice. Slice size too big?")

    def __iter__(self):
        return self

if __name__ == "__main__":
    # make small test hdf5 object
    datasets = np.random.randn(5, 10, 15, 20)
    with h5py.File('._test.h5', mode='w') as f:
        key_names = list('abcde')
        for i, k in enumerate(key_names):
            grp = f.create_group(k)
            for j, dataset in enumerate(datasets[i]):
                grp.create_dataset(str(j), data=dataset)

    # Tests
    h = Hdf5Iterator('._test.h5', (2, None))
    a = next(h)
    assert a.shape == (2, 20)


    h = Hdf5Iterator('._test.h5', (0, None))
    a = next(h)
    assert a.shape == (0, 20)

    h = Hdf5Iterator('._test.h5', (None, None))
    a = next(h)
    assert a.shape == (15, 20)

    try:
        fail_test = True
        h = Hdf5Iterator('._test.h5', (100, 100))
        a = next(h)
        assert a.shape == (15, 20)
    except ValueError:
        fail_test = False

    assert not fail_test

    h = Hdf5Iterator('._test.h5', (2, 5))
    a = next(h)
    assert a.shape == (2, 5)

    for a in h:
        print(a)
