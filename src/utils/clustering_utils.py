"""
Contains functions for working with separation models based on clustering T-F
domain vectors. Assumes that models have a get_vectors method which takes in an
input and returns the T-F vectors for clustering.
"""

import numpy as np
from sklearn.cluster import KMeans

from ..features.spectral_features import istft
from ..features.data_preprocessing import make_stft_features, \
                                                 undo_preemphasis

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


def get_cluster_masks(vectors, num_sources):
    """
    Cluster the vectors using k-means with k=num_sources.  Use the cluster IDs
    to create num_sources T-F masks.

    Inputs:
        vectors: Numpy array of shape (Batch, Time, Frequency, Embedding).
                 Only the masks for the first batch are computed.
        num_sources: Integer number of sources to compute masks for

    Returns:
         masks: Numpy array of shape (Time, Frequency, num_sources) containing
                the estimated binary mask for each of the num_sources sources.
    """

    # Get the shape of the input
    shape = np.shape(vectors)

    # Do k-means clustering
    kmeans = KMeans(n_clusters=num_sources, random_state=0)
    kmeans.fit(vectors[0].reshape((shape[1]*shape[2],shape[3])))

    # Preallocate mask array
    masks = np.zeros((shape[1]*shape[2], num_sources))

    # Use cluster IDs to construct masks
    labels = kmeans.labels_
    for i in range(labels.shape[0]):
        label = labels[i]
        masks[i,label] = 1

    masks = masks.reshape((shape[1], shape[2], num_sources))

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

def clustering_separate(signal, sample_rate, model, num_sources):
    """
    Takes in a signal and a model which has a get_vectors method and returns
    the specified number of output sources.

    Inputs:
        signal: Numpy 1D array containing waveform to separate.
        sample_rate: Sampling rate of the input signal
        model: Instance of model to use to separate the signal
        num_sources: Integer number of sources to separate into

    Returns:
        sources: Numpy ndarray of shape (num_sources, signal_length)
    """

    # Get the T-F embedding vectors for this signal from the model
    spectrogram, vectors = process_signal(signal, sample_rate, model)

    # Run k-means clustering on the vectors with k=num_sources to recover the
    # signal masks
    masks = get_cluster_masks(vectors, num_sources)

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
