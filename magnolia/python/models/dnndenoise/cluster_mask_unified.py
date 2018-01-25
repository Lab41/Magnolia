import logging.config
import numpy as np
import tensorflow as tf

from magnolia.models.model_base import ModelBase
from magnolia.utils import tf_utils


logger = logging.getLogger('model')


class RatioMaskClusterUnified(ModelBase):
    """
    Chimera network from [1] but (implicitly) uses the soft C-means loss.
    Defaults correspond to the parameters used by the best
    performing model in the paper.

    [1] Luo, Yi., et al. "Deep Clustering and Conventional Networks for Music
        Separation: Stronger Together" Published in Acoustics, Speech, and
        Signal Processing (ICASSP) 2017; doi:10.1109/ICASSP.2017.7952118

    Hyperparameters:
        F: Number of frequency bins in the input data
        num_reco_sources: Number sources to reconstruct
        num_training_sources: Number sources in the training set
        layer_size: Size of BLSTM layers
        embedding_size: Dimension of embedding vector
        alpha: Relative mixture of cost terms
        nonlinearity: Nonlinearity to use in BLSTM layers
        device: Which device to run the model on
    """

    def initialize(self):
        self.F = self.config['model_params']['F']
        # should always be 2
        self.num_reco_sources = self.config['model_params']['num_reco_sources']
        self.num_training_sources = self.config['model_params']['num_training_sources']
        self.layer_size = self.config['model_params']['layer_size']
        self.fuzzifier = self.config['model_params']['fuzzifier']
        self.embedding_size = self.config['model_params']['embedding_size']
        self.auxiliary_size = self.config['model_params']['auxiliary_size']
        self.normalize = self.config['model_params']['normalize']
        self.alpha = self.config['model_params']['alpha']
        self.nonlinearity = eval(self.config['model_params']['nonlinearity'])
        self.collapse_sources = self.config['model_params']['collapse_sources']

        self.batch_count = 0
        self.costs = []
        self.t_costs = []
        self.v_costs = []
        self.last_saved = 0

    def build_graph(self, graph):
        with graph.as_default():
            with tf.device(self.config['device']):
                # Placeholder tensor for the input data
                self.X = tf.placeholder("float", [None, None, self.F])
                # Placeholder tensor for the unscaled input data
                self.X_clean = tf.placeholder("float", [None, None, self.F])

                # Placeholder tensor for the labels/targets
                #self.y = tf.placeholder("float", [None, None, self.F, None])
                # Placeholder tensor for the unscaled labels/targets
                self.y_clean = tf.placeholder(
                    "float", [None, None, self.F, None])

                # Placeholder for the speaker indicies
                self.I = tf.placeholder(tf.int32, [None, None])

                # Define the speaker vectors to use during training
                self.speaker_vectors = tf_utils.weight_variable(
                    [self.num_training_sources, self.embedding_size],
                    tf.sqrt(2 / self.embedding_size))
                if self.auxiliary_size > 0:
                    # Define the auxiliary vectors to use during training
                    self.auxiliary_vectors = tf_utils.weight_variable(
                        [self.auxiliary_size, self.auxiliary_size],
                        tf.sqrt(2 / self.auxiliary_size))
                else:
                    self.auxiliary_vectors = None

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

        batch_count = self.batch_count
        # Training epoch loop
        for batch in iter(training_mixer):
            unscaled_spectral_sum_batch, scaled_spectral_sum_batch, spectral_masks_batch, spectral_sources_batch = batch_formatter(
                batch[0], batch[1], batch[2])
            # should be dimensions of (batch size, source)
            uids_batch = batch[3]

            # override ids for simply signal/noise
            if self.collapse_sources:
                uids_batch[:, 0] = 0
                uids_batch[:, 1] = 1

            # Train the model on one batch and get the cost
            c = self.train_on_batch(scaled_spectral_sum_batch, unscaled_spectral_sum_batch,
                                    spectral_sources_batch, uids_batch)

            # Store the training cost
            self.costs.append(c)

            # Evaluate the model on the validation data
            if (batch_count + 1) % validate_every == 0:
                # Store the training cost
                self.t_costs.append(np.mean(self.costs))
                # Reset the cost over the last 10 batches
                self.costs = []

                # Compute average validation score
                all_c_v = []
                for vbatch in iter(validation_mixer):
                    unscaled_spectral_sum_batch, scaled_spectral_sum_batch, spectral_masks_batch, spectral_sources_batch = batch_formatter(
                        vbatch[0], vbatch[1], vbatch[2])
                    # dimensions of (batch size, source)
                    uids_batch = vbatch[3]

                    # override ids for simply signal/noise
                    if self.collapse_sources:
                        uids_batch[:, 0] = 0
                        uids_batch[:, 1] = 1

                    # Get the cost on the validation batch
                    c_v = self.get_cost(scaled_spectral_sum_batch, unscaled_spectral_sum_batch,
                                        spectral_sources_batch, uids_batch)
                    all_c_v.append(c_v)

                ave_c_v = np.mean(all_c_v)

                # Check if the validation cost is below the minimum validation cost, and if so, save it.
                # and len(self.nbatches) > 0:
                if len(self.v_costs) > 0 and ave_c_v < min(self.v_costs):
                    logger.info("Saving the model because validation score is {} below the old minimum.".format(
                        min(self.v_costs) - ave_c_v))

                    # Save the model to the specified path
                    self.save(model_save_base)

                    # Record the batch that the model was last saved on
                    self.last_saved = batch_count  # self.nbatches[-1]

                # Store the validation cost
                self.v_costs.append(ave_c_v)

                # Store the current batch number
                # self.nbatches.append(batch_count)

                logger.info("Training cost on batch {} is {}.".format(
                    batch_count, self.t_costs[-1]))
                logger.info("Validation cost on batch {} is {}.".format(
                    batch_count, self.v_costs[-1]))
                logger.info("Last saved {} batches ago.".format(
                    batch_count - self.last_saved))

                # Stop training if the number of iterations since the last save point exceeds the threshold
                if batch_count - self.last_saved > stop_threshold:
                    logger.info("Early stopping criteria met!")
                    break

            batch_count += 1

        self.batch_count = batch_count

    def infer(self, **kw_args):
        pass

    @tf_utils.scope_decorator
    def network(self):
        """
        Construct the op for the network used in [1].  This consists of four
        BLSTM layers followed by a dense layer giving a set of T-F vectors of
        dimension embedding_size
        """

        m = self.fuzzifier
        reduced_contrast = True

        # Get the shape of the input
        shape = tf.shape(self.X)
        shapeI = tf.shape(self.I)

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
                                            [1, self.layer_size, self.embedding_size * self.F])

        # Reshape the feedforward output to have shape (T,F,D)
        z = tf.reshape(feedforward,
                       [shape[0], shape[1], self.F, self.embedding_size])

        # indices helpers for fuzzy c-means
        flattened_I = tf.reshape(self.I, shape=[shapeI[0] * shapeI[1]])
        batch_range = tf.range(shape[0] * shapeI[1])

        current_sources_indices, current_sources_indices_batch = tf.unique(
            flattened_I)
        known_sources_indices = tf.get_variable('known_sources_indices',
                                                initializer=tf.constant(
                                                    [], dtype=tf.int32),
                                                dtype=tf.int32,
                                                validate_shape=False, trainable=False)
        known_sources_indices = tf.sets.set_union(tf.expand_dims(current_sources_indices, 0),
                                                  tf.expand_dims(known_sources_indices, 0)).values

        # clustering head
        embedding = self.nonlinearity(z)
        # Normalize the T-F vectors to get the network output
        embedding = tf.nn.l2_normalize(embedding, 3)

        # batch, all features, embedding
        embeddings = tf.reshape(embedding,
                                [shape[0], shape[1] * self.F, self.embedding_size])

        # compute fuzzy assignments
        # batch, nsource in mix, nfeatures
        if self.auxiliary_vectors is None:
            # batch, nsource in mix, nfeatures
            squared_diffs_batch = tf.reduce_sum(tf.square(tf.expand_dims(embeddings, 1) -
                                                          tf.reshape(tf.expand_dims(tf.gather(self.speaker_vectors, flattened_I),  1),
                                                                     [shape[0], shapeI[1], 1, self.embedding_size])),
                                                -1)
            diffs_pow_matrix_batch = tf.pow(squared_diffs_batch, 1. / (m - 1.))
            # batch, 1, nfeatures
            W_denom = tf.expand_dims(tf.reduce_sum(tf.reciprocal(diffs_pow_matrix_batch), 1), 1)
        else:
            # NOTE: true/aux refers to both the coordinates and cluster centers
            true_embeddings = embeddings[:, :, :-self.auxiliary_size]
            aux_embeddings = embeddings[:, :, -self.auxiliary_size:]
            true_embeddings_l2 = tf.reduce_sum(
                tf.square(true_embeddings), axis=-1)
            aux_embeddings_l2 = tf.reduce_sum(
                tf.square(aux_embeddings), axis=-1)
            # batch, nsource in mix, nfeatures
            true_squared_diffs_batch = tf.reduce_sum(tf.square(tf.expand_dims(true_embeddings, 1) -
                                                               tf.reshape(tf.expand_dims(tf.gather(self.speaker_vectors, flattened_I),  1),
                                                                          [shape[0], shapeI[1], 1, self.embedding_size])),
                                                     -1)
            squared_diffs_batch = true_squared_diffs_batch + tf.expand_dims(aux_embeddings_l2, 1)
            diffs_pow_matrix_batch = tf.pow(squared_diffs_batch, 1. / (m - 1.))
            # batch, nfeatures, nsources (aux)
            aux_squared_diffs = tf.reduce_sum(tf.square(tf.expand_dims(
                aux_embeddings, 2) - tf.expand_dims(tf.expand_dims(self.auxiliary_vectors, 0), 0)), -1)
            aux_diffs_pow_matrix = tf.pow(aux_squared_diffs + tf.expand_dims(true_embeddings_l2, -1), 1. / (m - 1.))

            W_denom = tf.reduce_sum(tf.reciprocal(
                tf.concat([
                    tf.transpose(diffs_pow_matrix_batch, perm=[0, 2, 1]),
                    aux_diffs_pow_matrix
                ], axis=2)
            ), -1)
            # batch, 1, nfeatures
            W_denom = tf.expand_dims(W_denom, 1)

        # batch, nsource in mix, nfeatures
        W = tf.reciprocal(diffs_pow_matrix_batch * W_denom, name='W')
        # batch, nfeatures, nsource in mix
        clustering_factors = tf.transpose(W, perm=[0, 2, 1])

        # MI head
        # Feedforward layer
        # feedforward_fc = tf_utils.conv2d_layer(z,
        #                      [1, 1, self.embedding_size, self.num_reco_sources])
        # perform a softmax along the source dimension
        #mi_head = tf.nn.softmax(feedforward_fc, dim=3)

        # MI head
        mi_head = tf.reshape(clustering_factors,
                             [shape[0], shape[1], self.F, shapeI[1]])

        return embedding, mi_head

    @tf_utils.scope_decorator
    def cost(self):
        """
        Constuct the cost function op for the cost function used in sce
        and the mask inference head
        """

        # Get the shape of the target
        shape = tf.shape(self.y_clean)

        #cluster_output, mi_output, W, squared_diffs = self.network
        cluster_output, mi_output = self.network

        # broadcast product along source dimension
        mi_cost = tf.square(self.y_clean - mi_output *
                            tf.expand_dims(self.X_clean, -1))

        return tf.reduce_mean(mi_cost)

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

    def train_on_batch(self, X_train, X_train_clean, y_train_clean, I_train):
        """
        Train the model on a batch with input X and target y. Returns the cost
        computed on this batch.
        """

        cost, _ = self.sess.run([self.cost, self.optimizer],
                                {self.X: X_train,
                                 self.X_clean: X_train_clean,
                                 self.y_clean: y_train_clean,
                                 self.I: I_train})

        return cost

    def get_masks(self, X_in, nsources=2, nclustering_iterations_max=500, iterations_stop=10):
        """
        Compute the masks for the input spectrograms
        """
        
        nspectrograms = len(X_in)
        #I = np.arange(nspectrograms * nsources, dtype=np.int32).reshape(nspectrograms, nsources)
        I = np.tile(np.arange(nsources, dtype=np.int32), reps=(nspectrograms, 1))
        
        opt = tf.train.AdamOptimizer()
        clustering_minimize = opt.minimize(self.clustering_cost, var_list=[self.speaker_vectors])
        
        previous_cost = np.finfo('float').max
        iterations_count = 0
        for i in range(nclustering_iterations_max):
            cost, _ = self.sess.run([self.clustering_cost,
                                     clustering_minimize],
                                    {self.X: X_in,
                                     self.I: I})
            if cost < previous_cost:
                previous_cost = cost
            else:
                iterations_count += 1
            
            if iterations_count >= iterations_stop:
                break

        masks = self.sess.run(self.network, {self.X: X_in, self.I: I})[1]
        return masks

    def get_vectors(self, X_in):
        """
        Compute the embedding vectors for the input spectrograms
        """

        vectors = self.sess.run(self.network, {self.X: X_in})[0]
        return vectors

    def get_cost(self, X_in, X_clean_in, y_clean_in, I_in):
        """
        Computes the cost of a batch, but does not update any model parameters.
        """
        cost = self.sess.run(self.cost, {self.X: X_in,
                                         self.X_clean: X_clean_in,
                                         self.y_clean: y_clean_in,
                                         self.I: I_in})
        return cost
