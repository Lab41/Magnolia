import logging.config
import numpy as np
import tensorflow as tf

from magnolia.models.model_base import ModelBase
from magnolia.utils import tf_utils


logger = logging.getLogger('model')


class JFLEC(ModelBase):
    """
    """

    def initialize(self):
        self.num_sources = self.config['model_params']['num_sources']
        self.num_samples = self.config['model_params']['num_samples']
        self.num_encoding_layers = self.config['model_params']['num_encoding_layers']
        self.embedding_size = self.config['model_params']['embedding_size']
        self.num_decoding_layers = self.config['model_params']['num_decoding_layers']
        self.alpha_init = self.config['model_params']['alpha']
        nl = self.config['model_params']['nonlinearity']
        self.nonlinearity = eval('{}'.format(nl))

        self.batch_count = 0
        self.nbatches = []
        self.costs = []
        self.t_costs = []
        self.v_costs = []
        self.last_saved = 0


    def build_graph(self, graph):
        with graph.as_default():
            with tf.device(self.config['device']):
                # Placeholder tensor for the input data
                self.X = tf.placeholder(tf.float32, [None, self.num_samples])

                # Placeholder tensor for UIDs for sources
                self.X_uids = tf.placeholder(tf.int32, [None, 2])

                # Placeholder scalar for relative cost factor
                self.Alpha = tf.placeholder(tf.float32)

                # Placeholder tensor for the labels/targets
                self.Y = tf.placeholder(tf.float32, [None, self.num_samples])

                # Model methods
                self.network
                self.cost
                self.optimizer

        return graph


    def learn_from_epoch(self, epoch_id,
                         validate_every,
                         stop_threshold,
                         training_mixer,
                         validation_mixer,
                         batch_formatter,
                         model_save_base):
        # FIXME:
        # Find the number of batches already elapsed (Useful for resuming training)
        start = 0
        if len(self.nbatches) != 0:
            start = self.nbatches[-1]

        batch_count = self.batch_count
        # Training epoch loop
        for batch in iter(training_mixer):
            unscaled_spectral_sum_batch, scaled_spectral_sum_batch, spectral_masks_batch, spectral_sources_batch = batch_formatter(batch[0], batch[1], batch[2])
            # should be dimensions of (batch size, source)
            uids_batch = batch[3]

            # Train the model on one batch and get the cost
            c = self.train_on_batch(scaled_spectral_sum_batch, unscaled_spectral_sum_batch,
                                    spectral_masks_batch, spectral_sources_batch)

            # Store the training cost
            self.costs.append(c)

            # Store the current batch_count number

            # Evaluate the model on the validation data
            if (batch_count + 1) % validate_every == 0:
                # Store the training cost
                self.t_costs.append(np.mean(self.costs))
                # Reset the cost over the last 10 batches
                self.costs = []

                # Compute average validation score
                all_c_v = []
                for vbatch in iter(validation_mixer):
                    unscaled_spectral_sum_batch, scaled_spectral_sum_batch, spectral_masks_batch, spectral_sources_batch = batch_formatter(vbatch[0], vbatch[1], vbatch[2])
                    # dimensions of (batch size, source)
                    uids_batch = vbatch[3]

                    # Get the cost on the validation batch
                    c_v = self.get_cost(scaled_spectral_sum_batch, unscaled_spectral_sum_batch,
                                        spectral_masks_batch, spectral_sources_batch)
                    all_c_v.append(c_v)

                ave_c_v = np.mean(all_c_v)

                # Check if the validation cost is below the minimum validation cost, and if so, save it.
                if len(self.v_costs) > 0 and ave_c_v < min(self.v_costs) and len(self.nbatches) > 0:
                    logger.info("Saving the model because validation score is {} below the old minimum.".format(min(self.v_costs) - ave_c_v))

                    # Save the model to the specified path
                    self.save(model_save_base)

                    # Record the batch that the model was last saved on
                    self.last_saved = self.nbatches[-1]

                # Store the validation cost
                self.v_costs.append(ave_c_v)

                # Store the current batch number
                self.nbatches.append(batch_count + 1 + start)

                # Compute scale quantities for plotting
                length = len(self.nbatches)
                cutoff = int(0.5*length)
                lowline = [min(self.v_costs)]*length

                logger.info("Training cost on batch {} is {}.".format(self.nbatches[-1], self.t_costs[-1]))
                logger.info("Validation cost on batch {} is {}.".format(self.nbatches[-1], self.v_costs[-1]))
                logger.info("Last saved {} batches ago.".format(self.nbatches[-1] - self.last_saved))

                # Stop training if the number of iterations since the last save point exceeds the threshold
                if self.nbatches[-1] - self.last_saved > stop_threshold:
                    logger.info("Early stopping criteria met!")
                    break

            batch_count += 1

        self.batch_count = batch_count


    def infer(self, **kw_args):
        pass


    @tf_utils.scope_decorator
    def network(self):
        """
        """

        # Get the shape of the input
        input_shape = tf.shape(self.X)

        # encoder
        l = tf.expand_dims(self.X)
        for i in range(self.num_encoding_layers):
            if i == 0:
                nfilters = 2**3
                filter_size = 2**3
            else:
                nfilters *= 2
            layer_num = i + 1
            l = self.encoding_layer(l, filter_size, nfilters, layer_num)

        # feature + embeddings
        embeddings = self.compute_embeddings(l)

        # clustering
        cc = self.make_cluster_centers()
        # compute fuzzy assignments
        # batch, feature 1, feature 2, nsources
        squared_diffs = tf.reduce_sum(tf.square(tf.expand_dims(embeddings, 3) - tf.expand_dims(tf.expand_dims(tf.expand_dims(cc, 0), 0), 0)), -1)
        squared_diffs_pow = tf.pow(squared_diffs, 1./(m - 1.))
        W = tf.reciprocal(squared_diffs_pow*tf.expand_dims(tf.reduce_sum(tf.reciprocal(squared_diffs_pow), -1), -1))

        WT = tf.transpose(W, perm=[0, 3, 1, 2])
        clustering_factors = tf.gather_nd(WT, self.X_uids)
        
        # NOTE: I'm making an explicit choice to multiply the embedding vectors
        #       by the fuzzy c-means coefficients
        # batch, feature 1, feature 2, nsources, embedding_size
        scaled_embeddings = tf.expand_dims(clustering_factors, -1)*embeddings

        # decoder
        # collapse embedding dimension with convolution (should revisit later)
        with tf.variable_scope('embedding_decoder', reuse=tf.AUTO_REUSE):
            filters = tf.truncated_normal([1, 1, 1, self.embedding_size, 1], mean=0.0, stddev=0.1)
            bias = tf.truncated_normal([1], mean=0.0, stddev=0.1)
            l = tf.nn.convolution(input=scaled_embeddings,
                                  filter=tf.get_variable(name='weights',
                                                         initializer=filters),
                                  padding='VALID') + \
                tf.get_variable(name='bias', initializer=bias)
            l = self.nonlinearity(tf.squeeze(l))
            
        #for i in range(self.num_encoding_layers):

        return embeddings, W

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

    # def save(self, path):
    #     """
    #     Saves the model to the specified path.
    #     """
    #     self.saver.save(self.sess, path)

    # def load(self, path):
    #     """
    #     Load the model from the specified path.
    #     """
    #     self.saver.restore(self.sess, path)

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

    def encoding_layer(self, prev_layer, filter_size, nfilters, layer_num):
        shape = tf.shape(prev_layer)
        prev_nfilters = shape[-1]

        with tf.variable_scope('encoding_layer_{}'.format(layer_num),
                               reuse=tf.AUTO_REUSE):
            filters = tf.truncated_normal([filter_size, prev_nfilters, nfilters],
                                          mean=0.0, stddev=0.1)
            bias = tf.truncated_normal([nfilters],
                                       mean=0.0, stddev=0.1)
            l = tf.nn.convolution(input=prev_layer,
                                  filter=tf.get_variable(name='weights',
                                                         initializer=filters),
                                  padding='VALID',
                                  strides=[1, 1, 2]) + \
                 tf.get_variable(name='bias',
                                 initializer=bias)
            
            l = self.nonlinearity(l)
        
        return l

    def compute_embeddings(self, prev_layer):
        feature_shape = tf.shape(prev_layer)
        l = tf.expand_dims(prev_layer)

        with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
            # NOTE: This convolution will scan over all "time bins" with
            #       different weights for each embedding dimension.
            #       The filter window is divised such that the temporal
            #       correlations are preserved
            filters = tf.truncated_normal([2*feature_shape[1] - 1, 1,
                                           1, self.embedding_size], mean=0.0, stddev=0.1)
            bias = tf.truncated_normal([self.embedding_size], mean=0.0, stddev=0.1)
            embeddings = tf.nn.convolution(input=l,
                                           filter=tf.get_variable(name='weights',
                                                                  initializer=filters),
                                           padding='SAME') + \
                         tf.get_variable(name='bias',
                                         initializer=bias)
            
            embeddings = self.nonlinearity(embeddings)
            
        return embeddings

    def make_cluster_centers(self):
        with tf.variable_scope('cluster_centers', reuse=tf.AUTO_REUSE):
            init = tf.truncated_normal([self.num_sources, self.embedding_size],
                                       mean=0.0, stddev=0.1)

            w = tf.get_variable(name='weights', initializer=init)

        return w
