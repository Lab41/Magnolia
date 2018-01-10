import logging.config
import numpy as np
import tensorflow as tf

from magnolia.models.model_base import ModelBase
from magnolia.utils import tf_utils


logger = logging.getLogger('model')


class RatioMaskCluster(ModelBase):
    """
    Chimera network from [1] but uses the soft C-means loss.
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
        self.num_reco_sources = self.config['model_params']['num_reco_sources']  # should always be 2
        self.num_training_sources = self.config['model_params']['num_training_sources']
        self.layer_size = self.config['model_params']['layer_size']
        self.embedding_size = self.config['model_params']['embedding_size']
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
                self.y = tf.placeholder("float", [None, None, self.F, None])
                # Placeholder tensor for the unscaled labels/targets
                self.y_clean = tf.placeholder("float", [None, None, self.F, None])
                
                # Placeholder for the speaker indicies
                self.I = tf.placeholder(tf.int32, [None, None])
                
                # Define the speaker vectors to use during training
                self.speaker_vectors = tf_utils.weight_variable(
                                       [self.num_training_sources, self.embedding_size],
                                       tf.sqrt(2/self.embedding_size))

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
            unscaled_spectral_sum_batch, scaled_spectral_sum_batch, spectral_masks_batch, spectral_sources_batch = batch_formatter(batch[0], batch[1], batch[2])
            # should be dimensions of (batch size, source)
            uids_batch = batch[3]
            
            # override ids for simply signal/noise
            if self.collapse_sources:
                uids_batch[:, 0] = 0
                uids_batch[:, 1] = 1

            # Train the model on one batch and get the cost
            c = self.train_on_batch(scaled_spectral_sum_batch, unscaled_spectral_sum_batch,
                                    spectral_masks_batch, spectral_sources_batch,
                                    uids_batch)

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
                    unscaled_spectral_sum_batch, scaled_spectral_sum_batch, spectral_masks_batch, spectral_sources_batch = batch_formatter(vbatch[0], vbatch[1], vbatch[2])
                    # dimensions of (batch size, source)
                    uids_batch = vbatch[3]
                    
                    # override ids for simply signal/noise
                    if self.collapse_sources:
                        uids_batch[:, 0] = 0
                        uids_batch[:, 1] = 1

                    # Get the cost on the validation batch
                    c_v = self.get_cost(scaled_spectral_sum_batch, unscaled_spectral_sum_batch,
                                        spectral_masks_batch, spectral_sources_batch,
                                        uids_batch)
                    all_c_v.append(c_v)

                ave_c_v = np.mean(all_c_v)

                # Check if the validation cost is below the minimum validation cost, and if so, save it.
                if len(self.v_costs) > 0 and ave_c_v < min(self.v_costs):# and len(self.nbatches) > 0:
                    logger.info("Saving the model because validation score is {} below the old minimum.".format(min(self.v_costs) - ave_c_v))

                    # Save the model to the specified path
                    self.save(model_save_base)

                    # Record the batch that the model was last saved on
                    self.last_saved = batch_count#self.nbatches[-1]

                # Store the validation cost
                self.v_costs.append(ave_c_v)

                # Store the current batch number
                #self.nbatches.append(batch_count)

                logger.info("Training cost on batch {} is {}.".format(batch_count, self.t_costs[-1]))
                logger.info("Validation cost on batch {} is {}.".format(batch_count, self.v_costs[-1]))
                logger.info("Last saved {} batches ago.".format(batch_count - self.last_saved))

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

        # SCE head
        embedding = self.nonlinearity(z)
        # Normalize the T-F vectors to get the network output
        embedding = tf.nn.l2_normalize(embedding, 3)

        # MI head
        # Feedforward layer
        feedforward_fc = tf_utils.conv2d_layer(z,
                              [1, 1, self.embedding_size, self.num_reco_sources])
        # perform a softmax along the source dimension
        mi_head = tf.nn.softmax(feedforward_fc, dim=3)

        return embedding, mi_head

    @tf_utils.scope_decorator
    def cost(self):
        """
        Constuct the cost function op for the cost function used in sce
        and the mask inference head
        """

        # Get the shape of the input
        shape = tf.shape(self.y)

        sce_output, mi_output = self.network

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
        embedding_broad = tf.expand_dims(sce_output, 3)

        # Compute the dot product between the emebedding vectors and speaker
        # vectors
        dot = tf.reduce_sum(Vspeakers_broad * embedding_broad, 4)

        # Compute the cost for every element
        sce_cost = -tf.log(tf.nn.sigmoid(self.y * dot))

        # Average the cost over all speakers in the input
        sce_cost = tf.reduce_mean(sce_cost, 3)

        # Average the cost over all batches
        sce_cost = tf.reduce_mean(sce_cost, 0)

        # Average the cost over all T-F elements.  Here is where weighting to
        # account for gradient confidence can occur
        sce_cost = tf.reduce_mean(sce_cost)

        # broadcast product along source dimension
        mi_cost = tf.square(self.y_clean - mi_output*tf.expand_dims(self.X_clean, -1))

        return self.alpha*sce_cost + (1.0 - self.alpha)*tf.reduce_mean(mi_cost)

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

    def train_on_batch(self, X_train, X_train_clean, y_train, y_train_clean, I_train):
        """
        Train the model on a batch with input X and target y. Returns the cost
        computed on this batch.
        """

        cost, _ = self.sess.run([self.cost, self.optimizer],
                                {self.X: X_train, self.y: y_train,
                                 self.X_clean: X_train_clean,
                                 self.y_clean: y_train_clean,
                                 self.I: I_train})

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

    def get_cost(self, X_in, X_clean_in, y_in, y_clean_in, I_in):
        """
        Computes the cost of a batch, but does not update any model parameters.
        """
        cost = self.sess.run(self.cost, {self.X: X_in, self.y: y_in,
                                         self.X_clean: X_clean_in,
                                         self.y_clean: y_clean_in,
                                         self.I: I_in})
        return cost
