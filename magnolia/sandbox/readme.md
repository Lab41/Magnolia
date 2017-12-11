# Sandbox Experiments

Multiple experiments run by labmates to explore methods for source separation in monaural and multi-mic environments. 

- [`cnn-mask`](cnn-mask), takes in stereo input and applies a binary mask by @cstephenson970
- [`rnn-mask`](https://github.com/Lab41/Magnolia/tree/master/sandbox/rnn-mask), implemented from https://arxiv.org/pdf/1502.04149.pdf by @pcallier
- [`time-ica`](https://github.com/Lab41/Magnolia/tree/master/sandbox/time-ica), implemented by @pcallier
- [`freq-ica`](https://github.com/Lab41/Magnolia/tree/master/sandbox/freq-ica), implemented by @pcallier
- [`kt-freq-ica`](https://github.com/Lab41/Magnolia/tree/master/sandbox/kt-freq-ica), multi-microphone separation from [K and T](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.457.7646&rep=rep1&type=pdf), implemented by @pcallier
- [`stereo-micviz`](https://github.com/Lab41/Magnolia/tree/master/sandbox/stereo-micviz), visualization of two microphone gain array, implemented by @cstephenson
- [`mfcc-LR-mask`](mfcc-LR-mask), prediction of ideal binary mask using linear (and to do, logistic regression) masks. Currently training and testing on the same data. Implemented by @kni
- [`mfcc-DNN-mask`](mfcc-DNN-mask), prediction of ideal binary mask using deep neural network, starting with keras. Implemented by @kni
- [`mfcc-multibranch`](mfcc-multibranch), make a forked neural network using output masks. Currently overfitting, based on architecture inspired by permutation-based networks. Implemented by @kni
