"""Script for running the preprocessing pipeline.

This script takes in a directory containing either the LibriSpeech or
UrbanSound8K datasets and produces an HDF5 file containing the fully-
preprocessed short-time Fourier transforms of each sound file.
"""


import os.path
import argparse
import logging.config
import json
import pandas as pd
from magnolia.features.preprocessing import make_stft_dataset


# These classes must have names of the form <dataset_type>_key_maker
class LibriSpeech_key_maker:
    """Creates keys for the speech dataset given its metadata and a filename"""
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path, sep='|', skiprows=12,
                              index_col=0, usecols=[0, 1],
                              names=['ID', 'SEX'])
        self.df['SEX'] = self.df['SEX'].map(str.strip)

    def __call__(self, filename):
        file_name_segments = filename.split(os.path.sep)
        sex = self.df.get_value(int(file_name_segments[-3]), 'SEX')
        return '{}/{}/{}'.format(sex, file_name_segments[-3], file_name_segments[-2])


class UrbanSound8K_key_maker:
    """Creates keys for the noise dataset given its metadata and a filename"""
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path, index_col=0, usecols=[0, 1, 4, 7])

    def __call__(self, filename):
        filename = filename.split(os.path.sep)[-1]
        salience = self.df.get_value(filename, 'salience')
        class_ = self.df.get_value(filename, 'class')
        return '{}/{}'.format(salience, class_)


def main():
    """Run the pipeline."""

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline.')
    parser.add_argument('--settings', '-s',
                        default='../../settings/preprocess_template.json',
                        help='preprocessing settings JSON file')
    parser.add_argument('--logger_settings', '-l',
                        default='../../settings/logging.conf',
                        help='logging configuration file')
    parser.add_argument('--logger_name', '-n',
                        default='preprocessing',
                        help='name of logger')
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger(args.logger_name)

    # Make the data set
    with open(args.settings) as settings_file:
        settings = json.load(settings_file)


        logger.debug('settings {}'.format(settings))
        logger.info(('looking in {} for raw audio data with file '
                     'extension {}').format(settings['data_directory'],
                                            settings['file_type']))
        logger.info('will store output HDF5 file in {}'.format(settings['output_file']))


        key_maker = None
        key_maker_name = '{}_key_maker'.format(settings['dataset_type'])
        if key_maker_name in globals():
            logger.debug('using the {} as the output key maker'.format(key_maker_name))
            key_maker = globals()[key_maker_name](settings['metadata_file'])


        print('Starting preprocessing...')
        make_stft_dataset(data_dir=settings['data_directory'],
                          file_type=settings['file_type'],
                          output_file=settings['output_file'],
                          key_maker=key_maker,
                          **settings['processing_parameters'])
        print('\nFinished!!')


if __name__ == '__main__':
    main()
