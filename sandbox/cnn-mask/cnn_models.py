"""
Define a 1D convolutional model to separate input signals into two component
sources
"""

import tensorflow as tf
from tf_utils import scope_decorator, leaky_relu, conv1d_layer

class Conv1DModel:
    def __init__(self, input_shape, output_shape,
                 filter_length, num_filters, embedding_size):
        """
        Create a model consisting of a 1D convolution, followed by two
        dense layers.

        Inputs:
            input_shape:  List containing the shape of the inputs
            output_shape: List containing the shape of the outputs

            filter_length:  Integer number of previous time slices to include in
                            convolution
            num_filters:    Integer number of convolutional filters to use
            embedding_size: Integer length of embedding vectors
        """
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder("float", input_shape)
            self.y_input = tf.placeholder("float", output_shape)
            self.F = input_shape[2]

            self.filter_length = filter_length
            self.num_filters = num_filters
            self.embedding_size = embedding_size

            self.learning_rate = tf.placeholder("float")

            self.network
            self.cost
            self.optimizer

        self.sess = tf.Session(graph=graph)
        self.saver = tf.train.Saver()

    def __del__(self):
        """
        Deletes the model
        """
        self.sess.close()

    @scope_decorator
    def network(self):
        """
        Constructs the network
        """

        # 1D convolution with num_filters filters
        shape = tf.shape(self.X_input)
        reshaped = tf.reshape(self.X_input, [shape[0],shape[1],self.F])
        conv_layer = conv1d_layer(reshaped,
                                  [self.filter_length,self.F,self.num_filters])
        conv_layer = leaky_relu(conv_layer)

        # Map the convolutional filters to an embedding vector for each T-F
        # element
        vectors = conv1d_layer(conv_layer,
                               [1,self.num_filters,self.embedding_size*self.F])
        vectors = tf.reshape(vectors,
                             [shape[0],shape[1],self.F,self.embedding_size])
        # Normalize the vectors
        vectors = tf.nn.l2_normalize(vectors, 3)

        # Compute the mask from the T-F vectors
        flat_vectors = tf.reshape(vectors,
                                  [shape[0],shape[1],self.F*self.embedding_size])
        logits = conv1d_layer(flat_vectors,
                              [1,self.F*self.embedding_size,2*self.F])
        logits = tf.reshape(logits, [shape[0],shape[1],self.F,2])
        mask = tf.nn.sigmoid(logits)

        # Multiply the mask times the input to generate the prediction
        prediction = tf.mul(mask,self.X_input)

        return logits, mask, prediction

    @scope_decorator
    def cost(self):
        """
        Computes the cross entropy cost using the network output
        """
        # Get the logits from the network
        logits, _, _ = self.network

        # Shape the network output and target values for use in computing the
        # sigmoid cross entropy
        shape = tf.shape(logits)
        flat_logits = tf.reshape(logits,
                                 [shape[0]*shape[1]*shape[2],shape[3]])
        flat_labels = tf.reshape(self.y_input,
                                 [shape[0]*shape[1]*shape[2],shape[3]])

        cost = tf.contrib.losses.sigmoid_cross_entropy(logits=flat_logits,
                                               multi_class_labels=flat_labels)

        return cost

    @scope_decorator
    def optimizer(self):
        """
        Defines the optimization method to use in training
        """
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return opt.minimize(self.cost)

    def predict(self, X):
        """
        Gets the prediction from the inputs to the network
        """
        _, _, prediction = self.sess.run(self.network, feed_dict={X_input: X})

        return prediction 

    def save(self, path):
        """
        Saves the model to the path specified
        """
        self.saver.save(self.sess, path)

    def load(self, path):
        """
        Loads the model from the path specified
        """
        self.saver.restore(self.sess, path)
