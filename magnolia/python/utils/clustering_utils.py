"""
Contains functions for working with separation models based on clustering T-F
domain vectors. Assumes that models have a get_vectors method which takes in an
input and returns the T-F vectors for clustering.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.decomposition import PCA

from ..features.spectral_features import istft
from ..features.preprocessing import make_stft_features, \
                                     undo_preemphasis
from magnolia.utils.training import preprocess_l41_batch, preprocess_chimera_batch, preprocess_l41_regression_batch


def softmax(dots):
    max_value = np.amax(dots,axis=-1)
    exps = np.exp(dots-np.expand_dims(max_value,axis=-1))
    return exps/np.expand_dims(np.sum(exps,axis=-1),axis=-1)


def preprocess_signal(signal, sample_rate):
    """
    Preprocess a signal for input into a model

    Inputs:
        signal: Numpy 1D array containing waveform to process
        sample_rate: Sampling rate of the input signal

    Returns:
        spectrogram: STFT of the signal after resampling to 10kHz and adding
                 preemphasis.
        X_in: Scaled STFT input feature for the model
    """

    # Compute the spectrogram of the signal
    spectrogram = make_stft_features(signal, sample_rate)

    # Get the magnitude spectrogram
    mag_spec = np.abs(spectrogram)

    # Scale the magnitude spectrogram with a square root squashing, and percent
    # normalization
    X_in = np.sqrt(mag_spec)
    m = X_in.min()
    M = X_in.max()
    X_in = (X_in - m)/(M - m)

    return spectrogram, X_in

def process_signal(signal, sample_rate, model):
    """
    Compute the spectrogram and T-F embedding vectors for a signal using the
    specified model.

    Inputs:
        signal: Numpy 1D array containing waveform to process
        sample_rate: Sampling rate of the input signal
        model: Instance of model to use to separate the signal

    Returns:
        spectrogram: Numpy array of shape (Timeslices, Frequency) containing
                     the complex spectrogram of the input signal.
        vectors: Numpy array of shape (Timeslices, Frequency, Embedding)
    """

    # Preprocess the signal into an input feature
    spectrogram, X_in = preprocess_signal(signal, sample_rate)

    # Reshape the input feature into the shape the model expects and compute
    # the embedding vectors
    X_in = np.reshape(X_in, (1, X_in.shape[0], X_in.shape[1]))
    vectors = model.get_vectors(X_in)

    return spectrogram, vectors


def get_cluster_masks(vectors, num_sources, binary_mask=True, algo=None):
    """
    Cluster the vectors using k-means with k=num_sources.  Use the cluster IDs
    to create num_sources T-F masks.

    Inputs:
        vectors: Numpy array of shape (Batch, Time, Frequency, Embedding).
                 Only the masks for the first batch are computed.
        num_sources: Integer number of sources to compute masks for
        binary_mask: If true, computes binary masks.  Otherwise computes the
                     soft masks.
        algo: sklearn-compatable clustering algorithm

    Returns:
         masks: Numpy array of shape (Time, Frequency, num_sources) containing
                the estimated binary mask for each of the num_sources sources.
    """

    if algo is None:
        algo = KMeans(n_clusters=num_sources, random_state=0)

    # Get the shape of the input
    shape = np.shape(vectors)
    
    # Preallocate mask array
    masks = np.zeros((shape[1]*shape[2], num_sources))
    
    if algo.__class__.__name__ == 'BayesianGaussianMixture' or algo.__class__.__name__ == 'GaussianMixture':
        vectors = PCA(n_components=max(1, shape[3]//10),
                      random_state=0).fit_transform(vectors[0].reshape((shape[1]*shape[2], shape[3])))
        
        algo.fit(vectors)
        
        # all_probs = algo.predict_proba(vectors[0].reshape((shape[1]*shape[2], shape[3])))
        all_probs = algo.predict_proba(vectors)
        
        if binary_mask:
            for i in range(all_probs.shape[0]):
                probs = all_probs[i]
                label = np.argmax(probs)
                masks[i, label] = 1
        else:
            for i in range(all_probs.shape[0]):
                probs = all_probs[i]
                masks[i, :] = probs/probs.sum()

        masks = masks.reshape((shape[1], shape[2], num_sources))

    else:
        # Do clustering
        algo.fit(vectors[0].reshape((shape[1]*shape[2],shape[3])))

        if binary_mask:
            # Use cluster IDs to construct masks
            labels = algo.labels_
            for i in range(labels.shape[0]):
                label = labels[i]
                masks[i, label] = 1

            masks = masks.reshape((shape[1], shape[2], num_sources))

        else:
            if algo.__class__.__name__ == 'KMeans':
                all_dists = algo.transform(vectors[0].reshape((shape[1]*shape[2],shape[3])))
                for i in range(all_dists.shape[0]):
                    dists = all_dists[i]
                    masks[i, :] = dists/dists.sum()

                masks = masks.reshape((shape[1], shape[2], num_sources))
            # # Get cluster centers
            # centers = algo.cluster_centers_
            # centers = centers.T
            # centers = np.expand_dims(centers, axis=0)
            # centers = np.expand_dims(centers, axis=0)
    
            # # Compute the masks using the cluster centers
            # masks = centers * np.expand_dims(vectors[0], axis=3)
            # # masks = np.sum(masks*1.5, axis=2)
            # masks = np.sum(masks, axis=2)
            # masks = softmax(masks)
            # # masks = 1/(1 + np.exp(-masks))

    return masks

def apply_masks(spectrogram, masks):
    """
    Takes in a signal spectrogram and apply a set of T-F masks to it to recover
    the sources.

    Inputs:
        spectrogram: Numpy array of shape (T, F) containing the complex
                     spectrogram of the signal to mask.
        masks: Numpy array of shape (T, F, sources) containing the T-F masks for
               each source.

    Returns:
        masked_spectrograms: Numpy array of shape (sources, T, F) containing
                             the masked complex spectrograms for each source.
    """
    num_sources = masks.shape[2]

    masked_specs = [masks[:,:,i]*spectrogram for i in range(num_sources)]

    return masked_specs

def clustering_separate(signal, sample_rate, model, num_sources,
                        binary_mask=True):
    """
    Takes in a signal and a model which has a get_vectors method and returns
    the specified number of output sources.

    Inputs:
        signal: Numpy 1D array containing waveform to separate.
        sample_rate: Sampling rate of the input signal
        model: Instance of model to use to separate the signal
        num_sources: Integer number of sources to separate into
        binary_mask: If true, computes the binary mask. Otherwise
                     computes a soft mask

    Returns:
        sources: Numpy ndarray of shape (num_sources, signal_length)
    """

    # Get the T-F embedding vectors for this signal from the model
    spectrogram, vectors = process_signal(signal, sample_rate, model)

    # Run k-means clustering on the vectors with k=num_sources to recover the
    # signal masks
    masks = get_cluster_masks(vectors, num_sources, binary_mask=binary_mask)

    # Apply the masks from the clustering to the input signal
    masked_specs = apply_masks(spectrogram, masks)

    # Invert the STFT to recover the output waveforms, remembering to undo the
    # preemphasis
    waveforms = []
    for i in range(num_sources):
        waveform = istft(masked_specs[i], 1e4, None, 0.0256, two_sided=False,
                         fft_size=512)
        unemphasized = undo_preemphasis(waveform)
        waveforms.append(unemphasized)

    sources = np.stack(waveforms)

    return sources


def l41_clustering_separate(spec, model, num_sources,
                            binary_mask=True):
    """
    Takes in a spectrogram and a model which has a get_vectors method and returns
    the specified number of output sources.

    Inputs:
        spec: Spectrogram (in the format from a MixIterator) to separate.
        model: Instance of model to use to separate the signal
        num_sources: Integer number of sources to separate into
        binary_mask: If true, computes the binary mask. Otherwise
                     computes a soft mask

    Returns:
        sources: Numpy ndarray of shape (num_sources, Spectrogram.shape)
    """


    model_spec = preprocess_l41_batch(spec)
    # Get the T-F embedding vectors for this signal from the model
    vectors = model.get_vectors(model_spec)

    # Clustering algo
    clusterer = KMeans(n_clusters=num_sources, random_state=0)
    # clusterer = SpectralClustering(n_clusters=num_sources, random_state=0)
    # clusterer = AgglomerativeClustering(n_clusters=num_sources)
    # clusterer = BayesianGaussianMixture(n_components=num_sources,
    #                                     covariance_type='full',
    #                                     reg_covar=1e-3,
    #                                     max_iter=1000,
    #                                     random_state=0)
    # clusterer = GaussianMixture(n_components=num_sources,
    #                             covariance_type='full',
    #                             #reg_covar=1e-3,
    #                             max_iter=1000,
    #                             random_state=0)

    # Run clustering algorithm on the vectors with k=num_sources to recover the
    # signal masks
    masks = get_cluster_masks(vectors, num_sources, binary_mask=binary_mask, algo=clusterer)

    # Apply the masks from the clustering to the input signal
    masked_specs = apply_masks(spec[0].T, masks)

    sources = np.stack(masked_specs)

    return sources.transpose(0, 2, 1)


def chimera_clustering_separate(spec, model, num_sources,
                                binary_mask=True):
    """
    Takes in a spectrogram and a model which has a get_vectors method and returns
    the specified number of output sources.

    Inputs:
        spec: Spectrogram (in the format from a MixIterator) to separate.
        model: Instance of model to use to separate the signal
        num_sources: Integer number of sources to separate into
        binary_mask: If true, computes the binary mask. Otherwise
                     computes a soft mask

    Returns:
        sources: Numpy ndarray of shape (num_sources, Spectrogram.shape)
    """


    model_spec = preprocess_chimera_batch(spec)[1]
    # Get the T-F embedding vectors for this signal from the model
    vectors = model.get_vectors(model_spec)

    # Clustering algo
    clusterer = KMeans(n_clusters=num_sources, random_state=0)

    # Run clustering algorithm on the vectors with k=num_sources to recover the
    # signal masks
    masks = get_cluster_masks(vectors, num_sources, binary_mask=binary_mask, algo=clusterer)

    # Apply the masks from the clustering to the input signal
    masked_specs = apply_masks(spec[0].T, masks)

    sources = np.stack(masked_specs)

    return sources.transpose(0, 2, 1)


def chimera_mask(spec, model):
    model_spec = preprocess_chimera_batch(spec)[1]
    soft_masks = model.get_masks(model_spec)

    masked_specs = soft_masks*np.expand_dims(spec.transpose(0, 2, 1), axis=-1)

    return masked_specs.transpose(0, 2, 1, 3)


def l41_regression_clustering_separate(spec, model, num_sources,
                                       binary_mask=True):
    """
    Takes in a spectrogram and a model which has a get_vectors method and returns
    the specified number of output sources.

    Inputs:
        spec: Spectrogram (in the format from a MixIterator) to separate.
        model: Instance of model to use to separate the signal
        num_sources: Integer number of sources to separate into
        binary_mask: If true, computes the binary mask. Otherwise
                     computes a soft mask

    Returns:
        sources: Numpy ndarray of shape (num_sources, Spectrogram.shape)
    """


    model_spec = preprocess_l41_regression_batch(spec)
    # Get the T-F embedding vectors for this signal from the model
    vectors = model.get_vectors(model_spec)

    # Clustering algo
    clusterer = KMeans(n_clusters=num_sources, random_state=0)

    # Run clustering algorithm on the vectors with k=num_sources to recover the
    # signal masks
    masks = get_cluster_masks(vectors, num_sources, binary_mask=binary_mask, algo=clusterer)

    # Apply the masks from the clustering to the input signal
    masked_specs = apply_masks(spec[0].T, masks)

    sources = np.stack(masked_specs)

    return sources.transpose(0, 2, 1)


def l41_regression_signal(spec, model):
    model_spec = preprocess_l41_regression_batch(spec)
    signal = model.get_signal(model_spec)

    return signal.transpose(0, 2, 1, 3)
