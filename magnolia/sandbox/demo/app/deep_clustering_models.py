import numpy as np
import tensorflow as tf

from .tf_utils import scope_decorator,BLSTM,conv1d_layer

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

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Placeholder tensor for the input data
            self.X = tf.placeholder("float", [None, None, self.F])

            # Placeholder tensor for the labels/targets
            self.y = tf.placeholder("float", [None, None, self.F, None])

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

    @scope_decorator
    def network(self):
        """
        Construct the op for the network used in [1].  This consists of two
        BLSTM layers followed by a dense layer giving a set of T-F vectors of
        dimension embedding_size
        """

        # Get the shape of the input
        shape = tf.shape(self.X)

        # BLSTM layer one
        BLSTM_1 = BLSTM(self.X, self.layer_size, 'one',
                                 nonlinearity=self.nonlinearity)

        # BLSTM layer two
        BLSTM_2 = BLSTM(BLSTM_1, self.layer_size, 'two',
                                 nonlinearity=self.nonlinearity)

        # Feedforward layer
        feedforward = conv1d_layer(BLSTM_2,
                              [1, self.layer_size, self.embedding_size*self.F])

        # Reshape the feedforward output to have shape (T,F,K)
        embedding = tf.reshape(feedforward,
                             [shape[0], shape[1], self.F, self.embedding_size])

        # Normalize the T-F vectors to get the network output
        embedding = tf.nn.l2_normalize(embedding, 3)

        return embedding

    @scope_decorator
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

    @scope_decorator
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