import numpy as np
import tensorflow as tf

from magnolia.models.model_base import ModelBase
from magnolia.utils import tf_utils

class Chimera(ModelBase):
    def __init__(self, F=257,
                 num_sources=2,
                 layer_size=500, embedding_size=10,
                 alpha=0.1,
                 nonlinearity=tf.tanh,
                 device='/cpu:0'):
        """
        Initializes the Chimera network from [1].  Defaults correspond to
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
            device: Which device to run the model on
        """

        self.F = F
        self.num_sources = num_sources
        self.layer_size = layer_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.nonlinearity = nonlinearity

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(device):
                # Placeholder tensor for the input data
                self.X = tf.placeholder("float", [None, None, self.F])
                # Placeholder tensor for the unscaled input data
                self.X_clean = tf.placeholder("float", [None, None, self.F])

                # Placeholder tensor for the labels/targets
                self.y = tf.placeholder("float", [None, None, self.F, None])
                # Placeholder tensor for the unscaled labels/targets
                self.y_clean = tf.placeholder("float", [None, None, self.F, None])

                # Model methods
                self.network
                self.cost
                self.optimizer

            # Saver
            self.saver = tf.train.Saver()


        # Create a session to run this graph
        self.sess = tf.Session(graph = self.graph)

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
        Construct the op for the network used in [1].  This consists of four
        BLSTM layers followed by a dense layer giving a set of T-F vectors of
        dimension embedding_size
        """

        # Get the shape of the input
        shape = tf.shape(self.X)

        # BLSTM layer one
        BLSTM_1 = tf_utils.BLSTM_(self.X, self.layer_size, 'one',
                                  activation=self.nonlinearity)

        # BLSTM layer two
        BLSTM_2 = tf_utils.BLSTM_(BLSTM_1, self.layer_size, 'two',
                                  activation=self.nonlinearity)

        # BLSTM layer three
        BLSTM_3 = tf_utils.BLSTM_(BLSTM_2, self.layer_size, 'three',
                                  activation=self.nonlinearity)

        # BLSTM layer four
        BLSTM_4 = tf_utils.BLSTM_(BLSTM_3, self.layer_size, 'four',
                                  activation=self.nonlinearity)

        # Feedforward layer
        feedforward = tf_utils.conv1d_layer(BLSTM_4,
                              [1, self.layer_size, self.embedding_size*self.F])

        # Reshape the feedforward output to have shape (T,F,D)
        z = tf.reshape(feedforward,
                        [shape[0], shape[1], self.F, self.embedding_size])

        # DC head
        embedding = self.nonlinearity(z)
        # Normalize the T-F vectors to get the network output
        embedding = tf.nn.l2_normalize(embedding, 3)

        # MI head
        # Feedforward layer
        feedforward_fc = tf_utils.conv2d_layer(z,
                              [1, 1, self.embedding_size, self.num_sources])
        # perform a softmax along the source dimension
        mi_head = tf.nn.softmax(feedforward_fc, dim=3)

        return embedding, mi_head

    @tf_utils.scope_decorator
    def cost(self):
        """
        Constuct the cost function op for the cost function used in the deep
        clusetering model and the mask inference head
        """

        # Get the shape of the input
        shape = tf.shape(self.y)

        dc_output, mi_output = self.network

        # Reshape the targets to be of shape (batch, T*F, c) and the vectors to
        # have shape (batch, T*F, K)
        Y = tf.reshape(self.y, [shape[0], shape[1]*shape[2], shape[3]])
        V = tf.reshape(dc_output,
                       [shape[0], shape[1]*shape[2], self.embedding_size])

        # Compute the partition size vectors
        ones = tf.ones([shape[0], shape[1]*shape[2], 1])
        mul_ones = tf.matmul(tf.transpose(Y, perm=[0,2,1]), ones)
        diagonal = tf.matmul(Y, mul_ones)
        # D = 1/tf.sqrt(diagonal)
        # D = tf.sqrt(1/diagonal)
        D = tf.sqrt(tf.where(tf.is_inf(1/diagonal), tf.ones_like(diagonal) * 0, 1/diagonal))
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
        dc_cost = tf.norm(VTV, axis=[-2,-1]) -2*tf.norm(VTY, axis=[-2,-1]) + \
                  tf.norm(YTY, axis=[-2,-1])

        # broadcast product along source dimension
        mi_cost = tf.square(self.y_clean - mi_output*tf.expand_dims(self.X_clean, -1))

        return self.alpha*tf.reduce_mean(dc_cost) + (1.0 - self.alpha)*tf.reduce_mean(mi_cost)

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

    def train_on_batch(self, X_train, X_train_clean, y_train, y_train_clean):
        """
        Train the model on a batch with input X and target y. Returns the cost
        computed on this batch.
        """

        cost, _ = self.sess.run([self.cost, self.optimizer],
                                {self.X: X_train, self.y: y_train,
                                 self.X_clean: X_train_clean,
                                 self.y_clean: y_train_clean})

        return cost

    def get_masks(self, X_in):
        """
        Compute the masks for the input spectrograms
        """

        masks = self.sess.run(self.network, {self.X: X_in})[1]
        return masks

    def get_vectors(self, X_in):
        """
        Compute the embedding vectors for the input spectrograms
        """

        vectors = self.sess.run(self.network, {self.X: X_in})[0]
        return vectors

    def get_cost(self, X_in, X_clean_in, y_in, y_clean_in):
        """
        Computes the cost of a batch, but does not update any model parameters.
        """
        cost = self.sess.run(self.cost, {self.X: X_in, self.y: y_in,
                             self.X_clean: X_clean_in,
                             self.y_clean: y_clean_in})
        return cost
