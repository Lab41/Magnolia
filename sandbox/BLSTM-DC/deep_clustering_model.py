import numpy as np
import tensorflow as tf

from sklearn.cluster import KMeans

from magnolia.features.spectral_features import istft
from magnolia.features.data_preprocessing import make_stft_features
from magnolia.utils import tf_utils

class DeepClusteringModel:
    def __init__(self, F=257,
                 layer_size=600, embedding_size=40,
                 nonlinearity='logistic'):
        """
        Initializes the deep clustering model from [1].  Defaults correspond to
        the parameters used by the best performing model in the paper.

        [1] Hershey, John., et al. "Deep Clustering: Discriminative embeddings
            for segmentation and separation." Acoustics, Speech, and Signal
            Processing (ICASSP), 2016 IEEE International Conference on. IEEE,
            2016.

        Inputs:
            F: Number of frequency bins in the input data
            layer_size: Size of BLSTM layers
            embedding_size: Dimension of embedding vector
            nonlinearity: Nonlinearity to use in BLSTM layers
        """

        self.F = F
        self.layer_size = layer_size
        self.embedding_size = embedding_size
        self.nonlinearity = nonlinearity

        graph = tf.Graph()
        with graph.as_default():

            # Placeholder tensor for the input data
            self.X = tf.placeholder("float", [None, None, self.F])

            # Placeholder tensor for the labels/targets
            self.y = tf.placeholder("float", [None, None, self.F, 2])

            # Model methods
            self.network
            self.cost
            self.optimizer

            # Saver
            self.saver = tf.train.Saver()


        # Create a session to run this graph
        self.sess = tf.Session(graph = graph)

    def __del__(self):
        """
        Close the session when the model is deleted
        """

        self.sess.close()

    @tf_utils.scope_decorator
    def network(self):
        """
        Construct the op for the network used in [1].  This consists of two
        BLSTM layers followed by a dense layer giving a set of T-F vectors of
        dimension embedding_size
        """

        # Get the shape of the input
        shape = tf.shape(self.X)

        # BLSTM layer one
        BLSTM_1 = tf_utils.BLSTM(self.X, self.layer_size, 'one',
                                 nonlinearity=self.nonlinearity)

        # BLSTM layer two
        BLSTM_2 = tf_utils.BLSTM(BLSTM_1, self.layer_size, 'two',
                                 nonlinearity=self.nonlinearity)

        # Feedforward layer
        feedforward = tf_utils.conv1d_layer(BLSTM_2,
                              [1, self.layer_size, self.embedding_size*self.F])

        # Reshape the feedforward output to have shape (T,F,K)
        embedding = tf.reshape(feedforward,
                             [shape[0], shape[1], self.F, self.embedding_size])

        # Normalize the T-F vectors to get the network output
        embedding = tf.nn.l2_normalize(embedding, 3)

        return embedding

    @tf_utils.scope_decorator
    def cost(self):
        """
        Constuct the cost function op for the cost function used in the deep
        clusetering model
        """

        # Get the shape of the input
        shape = tf.shape(self.y)

        # Reshape the targets to be of shape (batch, T*F, c) and the vectors to
        # have shape (batch, T*F, K)
        Y = tf.reshape(self.y, [shape[0], shape[1]*shape[2], shape[3]])
        V = tf.reshape(self.network,
                       [shape[0], shape[1]*shape[2], self.embedding_size])

        # Compute the partition size vectors
        ones = tf.ones([shape[0], shape[1]*shape[2], 1])
        mul_ones = tf.matmul(tf.transpose(Y, perm=[0,2,1]), ones)
        diagonal = tf.matmul(Y, mul_ones)
        D = 1/tf.sqrt(diagonal)
        D = tf.reshape(D, [shape[0], shape[1]*shape[2]])

        # Compute the matrix products needed for the cost function.  Reshapes
        # are to allow the diagonal to be multiplied across the correct
        # dimensions without explicitly constructing the full diagonal matrix.
        DV  = D * tf.transpose(V, perm=[2,0,1])
        DV = tf.transpose(DV, perm=[1,2,0])
        VTV = tf.matmul(tf.transpose(V, perm=[0,2,1]), DV)

        DY = D * tf.transpose(Y, perm=[2,0,1])
        DY = tf.transpose(DY, perm=[1,2,0])
        VTY = tf.matmul(tf.transpose(V, perm=[0,2,1]), DY)

        YTY = tf.matmul(tf.transpose(Y, perm=[0,2,1]), DY)

        # Compute the cost by taking the Frobenius norm for each matrix
        cost = tf.norm(VTV, axis=[-2,-1]) -2*tf.norm(VTY, axis=[-2,-1]) + \
               tf.norm(YTY, axis=[-2,-1])

        return tf.reduce_mean(cost)

    @tf_utils.scope_decorator
    def optimizer(self):
        """
        Constructs the optimizer op used to train the network
        """
        opt = tf.train.AdamOptimizer()
        return opt.minimize(self.cost)

    def save(self, path):
        """
        Saves the model to the specified path.
        """
        self.saver.save(self.sess, path)

    def load(self, path):
        """
        Load the model from the specified path.
        """
        self.saver.restore(self.sess, path)

    def train_on_batch(self, X_train, y_train):
        """
        Train the model on a batch with input X and target y. Returns the cost
        computed on this batch.
        """

        cost, _ = self.sess.run([self.cost, self.optimizer],
                                {self.X: X_train, self.y: y_train})

        return cost

    def get_vectors(self, X_in):
        """
        Compute the embedding vectors for the input spectrograms
        """

        vectors = self.sess.run(self.network, {self.X: X_in})
        return vectors

    def get_cost(self, X_in, y_in):
        """
        Computes the cost of a batch, but does not update any model parameters.
        """
        cost = self.sess.run(self.cost, {self.X: X_in, self.y: y_in})
        return cost

def preprocess_signal(signal, sample_rate):
    """
    Preprocess a signal for input into  DeepClusteringModel

    Inputs:
        signal: Numpy 1D array containing waveform to process
        sample_rate: Sampling rate of the input signal

    Returns:
        spectrogram: STFT of the signal after resampling to 10kHz and adding
                 preemphasis.
        X_in: Scaled STFT input feature for DeepClusteringModel
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

def get_cluster_masks(vectors, num_sources):
    """
    Cluster the vectors using k-means with k=num_sources.  Use the cluster IDs
    to create num_sources T-F masks.
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
    for i in range(labels):
        label = labels[i]
        masks[i,label] = 1

    masks = masks.reshape((shape[1], shape[2], num_sources))

    return masks

def deep_clustering_separate(signal, sample_rate, model, num_sources):
    """
    Takes in a signal and an instance of DeepClusterModel and returns the
    specified number of output sources.

    Inputs:
        signal: Numpy 1D array containing waveform to separate.
        sample_rate: Sampling rate of the input signal
        model: Instance of DeepClusterModel to use to separate the signal
        num_sources: Integer number of sources to separate into

    Returns:
        sources: Numpy ndarray of shape (num_sources, signal_length)
    """

    # Preprocess the signal into an input feature
    spectrogram, X_in = preprocess_signal(signal, sample_rate)

    # Reshape the input feature into the shape the model expects and compute
    # the embedding vectors
    X_in = np.reshape(X_in, (1, X_in.shape[0], X_in.shape[1]))
    vectors = model.get_vectors(X_in)

    # Run k-means clustering on the vectors with k=num_sources to recover the
    # signal masks
    masks = get_cluster_masks(vectors, num_sources)

    # Apply the masks from the clustering to the input signal
    masked_specs = [masks[:,:,i]*spectrogram for i in range(num_sources)]

    # Invert the STFT to recover the output waveforms
    waveforms = [istft(masked_specs[i], 1e4, None, 0.0256, two_sided=False,
                       fft_size=512) for i in range(num_sources)]

    sources = np.stack(waveforms)

    return sources
