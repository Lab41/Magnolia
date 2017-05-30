## Source Code 


### Simple Components Analysis
- Directory: [`factorization`](https://github.com/Lab41/Magnolia/tree/master/src/factorization)
- Description: PCA and ICA code, which show simple examples of how to do linear methods of denoising and signal isolation. These are either monaural or multi-microphone.

### Features
- Directory: [`features`](https://github.com/Lab41/Magnolia/tree/master/src/features)
- Description: Feature extraction code. In here, there is a mix of STFT, MFCC's, and other feature extraction code, to include preemphasis filters.

### Deep Neural Network Monaural Source Separation
- Directory: [`dnnseparate`](https://github.com/Lab41/Magnolia/tree/master/src/dnnseparate)
- Description: Monaural source separation classes used in our [SiPS paper](https://arxiv.org/abs/1705.04662), including our replication of other research papers. 
  - Source Contrastive Estimation (SCE), [the proposed method](https://arxiv.org/abs/1705.04662), 2017
  - [Deep Clustering](https://arxiv.org/abs/1508.04306), an implementation of Hershey et al, 2016
  - [Permutation Invariant Training](https://arxiv.org/abs/1607.00325), cost function implementation of Yu et al, 2016
  - [Deep Attractor Networks](https://arxiv.org/abs/1611.08930), cost function implementation of Chen et al, 2016

### Iterators
- Directory: [`iterate`](https://github.com/Lab41/Magnolia/tree/master/src/iterate)
- Description: Iterator code that will take an HDF5 file and iterate through it, and mixes thereof (of potentially many iterators).

### Notebooks
- Directory: [`notebooks`](notebooks)
- Description: Example notebooks describing how to run any and all of our algorithms.

### Util
- Directory: [`util`](util/)
- Description: Utility functions Contents include evaluation code, reconstruction code, and clustering code

### Demonstration Code
- Directory: [`demo `](https://github.com/Lab41/Magnolia/tree/master/src/demo)
- Description: A web-based application where you can click on buttons to play and denoise a `.wav` file. It uses flask and is browser based.
