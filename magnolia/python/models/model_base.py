import abc
import os
import copy
import logging.config
import tensorflow as tf


logger = logging.getLogger('model')


class ModelBase(abc.ABC):
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
        """Common initialization for all models

        Only needs a dictionary containing the configuration of parameters.
        Configurations:
            best: flag to indicate whether or not to use the best set hyperparameters
            debug: flag to indicate whether or not to print debugging information
            random_seed: random seed
        """
        # Make a `deepcopy` of the configuration before using it to avoid any
        # potential mutation if iterating asynchronously over configurations
        self.config = copy.deepcopy(config)

        # Keep the best hyperparameters found so far inside the model itself
        # This is a mechanism to load the best hyperparameters and override the configuration
        if self.config.get('best', False):
            self.config.update(self.get_best_config())

        self.debug_flag = self.config.get('debug', False)
        if self.debug_flag:
            logger.debug('configuration:', self.config)

        # Use it in your TF graph with tf.set_random_seed()
        self.random_seed = self.config.get('random_seed', None)

        # All models share some basics hyper parameters, this is the section where we
        # copy them into the model
        self.max_epoch = self.config.get('max_epoch', 10)
        self.learning_rate = self.config.get('learning_rate', 1e-4)

        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_model_props()

        # Again, child Model should provide its own build_grap function
        self.graph = self.build_graph(tf.Graph())

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.saver = tf.train.Saver(**self.config.get('saver_config', {}))

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.make_summaries = 'summary_config' in self.config
        if self.make_summaries:
            # This sub-dictionary has arguments listed here:
            # https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter
            self.sw = tf.summary.FileWriter(graph=self.sess.graph, **self.config['summary_config'])

        # Do any remaining model initialize
        self.initialize()

        # At the end of this function, you want your model to be ready!


    @abc.abstractmethod
    def set_model_props(self, props):
        # When you look at your model, you want to know exactly which custom options it needs.
        self.config.update(self.get_best_config())
        pass


    @abc.abstractmethod
    def get_best_config(self):
        # It returns a dictionary used to update the initial configuration (see __init__)
        return {}

    @staticmethod
    @abc.abstractmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overriden by the model')

    @abc.abstractmethod
    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden by the model')

    @abc.abstractmethod
    def infer(self):
        raise Exception('The infer function must be overriden by the model')

    @abc.abstractmethod
    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the model')

    def train(self, save_every=1):
        # This function is usually common to all your models, Here is an example:
        for epoch_id in range(0, self.max_iter):
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
        if self.config['debug']:
            logger.debug('Saving to {} with global_step {}'.format(self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(episode_id), global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)


    @abc.abstractmethod
    def initialize(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:

            if self.config['debug']:
                logger.debug('Loading the model from folder: {}'.format(self.result_dir))
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    @abc.abstractmethod
    def infer(self):
        # This function is usually common to all your models
        pass
