import numpy as np
import tensorflow as tf

from ..utils import tf_utils

class L41Model:
    def __init__(self, F=257, num_speakers=251,
                 layer_size=600, embedding_size=40,
                 nonlinearity='logistic',normalize=False,
                 device='/cpu:0'):
        """
        Initializes Lab41's clustering model.  Default architecture comes from
        the parameters used by the best performing model in the paper[1].
        [1] Hershey, John., et al. "Deep Clustering: Discriminative embeddings
            for segmentation and separation." Acoustics, Speech, and Signal
            Processing (ICASSP), 2016 IEEE International Conference on. IEEE,
            2016.
        Inputs:
            F: Number of frequency bins in the input data
            num_speakers: Number of unique speakers to train on. only use in
                          training.
            layer_size: Size of BLSTM layers
            embedding_size: Dimension of embedding vector
            nonlinearity: Nonlinearity to use in BLSTM layers (default logistic)
            normalize: Do you normalize vectors coming into the final layer?
                       (default False)
            device: Which device to run the model on
        """

        self.F = F
        self.num_speakers = num_speakers
        self.layer_size = layer_size
        self.embedding_size = embedding_size
        self.nonlinearity = nonlinearity
        self.normalize = normalize

        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.device(device):
                # Placeholder tensor for the input data
                self.X = tf.placeholder("float", [None, None, self.F])

                # Placeholder tensor for the labels/targets
                self.y = tf.placeholder("float", [None, None, self.F, None])

                # Placeholder for the speaker indicies
                self.I = tf.placeholder(tf.int32, [None,None])

                # Define the speaker vectors to use during training
                self.speaker_vectors = tf_utils.weight_variable(
                                       [self.num_speakers,self.embedding_size],
                                       tf.sqrt(2/self.embedding_size))

                # Model methods
                self.network
                self.cost
                self.optimizer

            # Saver
            self.saver = tf.train.Saver()


        # Create a session to run this graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph = self.graph, config = config)

    def __del__(self):
        """
        Close the session when the model is deleted
        """

        self.sess.close()

    def initialize(self):
        """
        Initialize variables in the graph
        """

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

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
        if self.normalize:
            embedding = tf.nn.l2_normalize(embedding, 3)

        return embedding

    @tf_utils.scope_decorator
    def cost(self):
        """
        Constuct the cost function op for the negative sampling cost
        """

        # Get the embedded T-F vectors from the network
        embedding = self.network

        # Reshape I so that it is of the correct dimension
        I = tf.expand_dims( self.I, axis=2 )

        # Normalize the speaker vectors and collect the speaker vectors
        # correspinding to the speakers in batch
        if self.normalize:
            speaker_vectors = tf.nn.l2_normalize(self.speaker_vectors, 1)
        else:
            speaker_vectors = self.speaker_vectors
        Vspeakers = tf.gather_nd(speaker_vectors, I)

        # Expand the dimensions in preparation for broadcasting
        Vspeakers_broad = tf.expand_dims(Vspeakers, 1)
        Vspeakers_broad = tf.expand_dims(Vspeakers_broad, 1)
        embedding_broad = tf.expand_dims(embedding, 3)

        # Compute the dot product between the emebedding vectors and speaker
        # vectors
        dot = tf.reduce_sum(Vspeakers_broad * embedding_broad, 4)

        # Compute the cost for every element
        cost = -tf.log(tf.nn.sigmoid(self.y * dot))

        # Average the cost over all speakers in the input
        cost = tf.reduce_mean(cost, 3)

        # Average the cost over all batches
        cost = tf.reduce_mean(cost, 0)

        # Average the cost over all T-F elements.  Here is where weighting to
        # account for gradient confidence can occur
        cost = tf.reduce_mean(cost)

        return cost

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

    def train_on_batch(self, X_train, y_train, I_train):
        """
        Train the model on a batch with input X and target y. Returns the cost
        computed on this batch.
        """

        cost, _ = self.sess.run([self.cost, self.optimizer],
                            {self.X: X_train, self.y: y_train, self.I:I_train})

        return cost

    def get_vectors(self, X_in):
        """
        Compute the embedding vectors for the input spectrograms
        """

        vectors = self.sess.run(self.network, {self.X: X_in})
        return vectors

    def get_cost(self, X_in, y_in, I_in):
        """
        Computes the cost of a batch, but does not update any model parameters.
        """
        cost = self.sess.run(self.cost,
                            {self.X: X_in, self.y: y_in, self.I: I_in})

        return cost
