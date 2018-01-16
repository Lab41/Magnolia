# Generic imports
import argparse
import logging.config
import json

import numpy as np
import pandas as pd
import tensorflow as tf

# Import the RatioMaskCluster separation model
from magnolia.models import make_model

# Import utilities for using the model
from magnolia.training.data_iteration.mix_iterator import MixIterator
from magnolia.utils.training import preprocess_chimera_batch
#from magnolia.utils.tf_utils import double_learnable_relu


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train the k-means clustering + ratio mask network.')
    # parser.add_argument('--model_settings', '-s',
    #                     default='../../../../data/models_settings/chimera_template.json',
    #                     help='model settings JSON file')
    # parser.add_argument('--_settings', '-s',
    #                     default='../../../../data/models_settings/chimera_template.json',
    #                     help='model settings JSON file')
    parser.add_argument('--logger_settings', '-l',
                        default='../../../../data/logging_settings/logging.conf',
                        help='logging configuration file')
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger('model')

    # Number of epochs
    num_epochs = 20 # try 20
    # Threshold for stopping if the model hasn't improved for this many consecutive batches
    stop_threshold = 10000
    # validate every number of these batches
    validate_every = 100
    train_batchsize = 8
    train_mixes = ['/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/settings/mixing_LibriSpeech_UrbanSound8K_train.json']
    train_from_disk = False
    validate_batchsize = 5
    validate_mixes = ['/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/settings/mixing_LibriSpeech_UrbanSound8K_validate.json']
    validate_from_disk = False
    model_params = {
        'layer_size': 500,
        'embedding_size': 10,
        'auxiliary_size': 0,
        'alpha': 0.1, # try 0.9
        'nonlinearity': 'tf.tanh',
        'fuzzifier': 2,
        'num_reco_sources': 2,
        'normalize': False,
        'collapse_sources': False,
    }
    model_location = '/gpu:0'
    uid_settings = '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/settings/assign_uids_LibriSpeech_UrbanSound8K.json'
    model_save_base = '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/model_saves/mask_cluster'


    training_mixer = MixIterator(mixes_settings_filenames=train_mixes,
                                 batch_size=train_batchsize,
                                 read_waveform=False,
                                 from_disk=train_from_disk)

    validation_mixer = MixIterator(mixes_settings_filenames=validate_mixes,
                                   batch_size=validate_batchsize,
                                   read_waveform=False,
                                   from_disk=validate_from_disk)

    # get frequency dimension
    frequency_dim = training_mixer.sample_dimensions()[0]
    # TODO: throw an exception
    assert(frequency_dim == validation_mixer.sample_dimensions()[0])

    # get number of sources
    settings = json.load(open(uid_settings))
    uid_file = settings['output_file']
    uid_csv = pd.read_csv(uid_file)
    number_of_sources = uid_csv['uid'].max() + 1

    model_params['F'] = frequency_dim
    model_params['num_training_sources'] = number_of_sources
    config = {'model_params': model_params,
              'device': model_location}
    model = make_model('RatioMaskCluster', config)

    model.train(validate_every=validate_every,
                stop_threshold=stop_threshold,
                training_mixer=training_mixer,
                validation_mixer=validation_mixer,
                batch_formatter=preprocess_chimera_batch,
                model_save_base=model_save_base)


if __name__ == '__main__':
    main()
