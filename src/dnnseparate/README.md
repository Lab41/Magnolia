## DNN Source Separation

Monaural source separation using several deep neural network approaches.  Each
.py file contains the corresponding model.

Current Models:
- _Lab41's model_ - in [L41model.py](L41model.py)  

    Source contrastive estimation. Contribution on [arXiv](https://arxiv.org/abs/1705.04662) and in submission to SiPS 2017.
    See notebooks for how to run these algorithms in the [notebooks folder](../notebooks)

- _Deep clustering_ in [deep_clustering_model.py](deep_clustering_model.py)

    Hershey, John., et al. "Deep Clustering: Discriminative embeddings
    for segmentation and separation." Acoustics, Speech, and Signal
    Processing (ICASSP), 2016 IEEE International Conference on. IEEE,
    2016.

- _Permutation invariant training_ - in [pit.py](pit.py)

    Kolbaek, Morten, et al. "Multi-talker speech separation and tracing with
    permutation invariant training of deep recurrent neural networks." preprint 2017

- _Deep Attractor Networks_ in ['dan.py'](dan.py)

    Zhuo Chen and Yi Luo and Nima Mesgarani, "Deep attractor network for single-microphone 
    speaker separation", preprint 2016, http://arxiv.org/abs/1611.08930
