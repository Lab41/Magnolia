import abc
import os
import copy
import logging.config

import tensorflow as tf


logger = logging.getLogger('model')


class ModelBase(abc.ABC):
    """Abstract interface for all models

    All models should handle the construction of their TensorFlow graphs, how to
    train them on an epoch of training data, and how to perform inference.

    Functions that must be overridden (call signatures are given below):
        build_graph: must return the TensoFlow computational graph of the model,
                     called during `__init__`
        learn_from_epoch: update the weights of a model on an ephoch of data
        infer: perform inference
    Functions that can be overwritten (call signatures are given below):
        initialize: function called before `build_graph` during `__init__`
        train: function for start training, calls `learn_from_epoch`
    Functions not likely to be overwritten (call signatures are given below):
        __init__:
        __del__:
        update_config: update model configuration
        reset_config: reset model configuration to be a given configuration
        load_config: load model configuration from file
        write_config: write model configuration to a file
        save: saves the model weights to a specified file
        load: loads the model weights from a specified file
    """


    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
        """Common initialization for all models

        Only needs a dictionary containing the configuration of parameters.

        Configurations:
            best: flag to indicate whether or not to use the best set hyperparameters
            best_config_file: file to load the best configuration
            debug: flag to indicate whether or not to print debugging information
            random_seed: random seed
            model_params: custom hyperparameters of model
            learning_params: parameters related to training model
            saver_config: TensorFlow Saver object initialization parameters
            summary_config: TensorFlow FileWriter object initialization parameters

        Attatches the following properties to the 'self' object:
            config: copy of the configuration passed in at object creation
            debug_flag: if debugging messages are requested
            random_seed: seed for RNG passed in through the 'config'
            graph: TensorFlow computational graph
            saver: TensorFlow Saver object for this graph
            sess: TensorFlow session for this graph
            make_summaries: flag to indicate if TensorBoard summaries are requested
            sw: TensorFlow FileWriter object to which summary information is written
        """

        # Make a `deepcopy` of the configuration before using it to avoid any
        # potential mutation if iterating asynchronously over configurations
        self.config = copy.deepcopy(config)

        # Keep the best hyperparameters found so far inside the model itself
        # This is a mechanism to load the best hyperparameters and override the configuration
        if self.config.get('best', False):
            self.config.update(self.get_config(self.config['best_config_file']))

        self.debug_flag = self.config.get('debug', False)
        if self.debug_flag:
            logger.debug('configuration:', self.config)

        # Use it in your TF graph with tf.set_random_seed()
        self.random_seed = self.config.get('random_seed', None)

        # Do any model initialize before the graph is made
        self.initialize()

        # Again, child Model should provide its own build_grap function
        self.graph = self.build_graph(tf.Graph())

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            saver_config = self.config.get('saver_config', {})
            self.saver = tf.train.Saver(**saver_config)

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.make_summaries = 'summary_config' in self.config
        if self.make_summaries:
            # This sub-dictionary has arguments listed here:
            # https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter
            self.sw = tf.summary.FileWriter(graph=self.sess.graph, **self.config['summary_config'])

        # At the end of this function, you want your model to be ready!
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())



    def __del__(self):
        # Close the session when the model is deleted
        self.sess.close()


    def update_config(self, props):
        self.config.update(props)


    def reset_config(self, props):
        self.config = copy.deepcopy(props)


    def load_config(self, config_file):
        # Returns a dictionary representing the configuration of the model
        with open(config_file, 'r') as json_file:
            return json.load(json_file)


    def write_config(self, config_filename):
        # Record the configuration
        with open(config_filename, 'w') as f:
            json.dump(self.config, f)


    def save(self, save_path):
        # Save model weights
        if self.debug_flag:
            global_step_t = tf.train.get_global_step(self.graph)
            global_step = self.sess.run(global_step_t)
            logger.debug('Saving at global_step {}'.format(global_step))
        self.saver.save(self.sess, save_path)


    def load(self, path):
        self.saver.restore(self.sess, path)
        # TODO: fix this
        #model_name = checkpoint_dir.split(os.sep)[-1]
        #dir_name = os.path.join(checkpoint_dir.split(os.sep)[:-1])
        #checkpoint = tf.train.get_checkpoint_state(dir_name, latest_filename=model_name)
        #if checkpoint is None:
        #    raise RuntimeError("Couldn't find checkpoint files at {}".format(checkpoint_dir))
        #path = checkpoint.model_checkpoint_path
        #step = int(path.split(os.sep)[-1].split('-')[-1])
        #if self.debug_flag:
        #    logger.debug('Loading the model (step {}) from folder: {}'.format(step, checkpoint_dir))
        #self.saver.restore(self.sess, path)


    ###########################################################
    # Commonly Overwritten Functions
    ###########################################################

    def initialize(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be
        # overidden cleanly
        pass


    def train(self, max_epoch=10, **kw_args):
        # This function is usually common to all your models, Here is an example:
        for epoch_id in range(0, max_epoch):
            self.learn_from_epoch(epoch_id, **kw_args)


    ###########################################################
    # Required Overwritten Functions
    ###########################################################

    @abc.abstractmethod
    def build_graph(self, graph):
        pass


    @abc.abstractmethod
    def learn_from_epoch(self, epoch_id, **kw_args):
        # Function to train per epoch
        pass


    @abc.abstractmethod
    def infer(self, **kw_args):
        pass
