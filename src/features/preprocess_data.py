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


# These classes must have names of the form <dataset_type>_metadata_handler
class LibriSpeech_metadata_handler:
    """Creates keys and a metadata table for the speech dataset given its metadata and a filename"""
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path, sep='|', skiprows=12,
                              index_col=0, usecols=[0, 1, 3],
                              names=['ID', 'SEX', 'TIME'])
        self.df['SEX'] = self.df['SEX'].map(str.strip)
        self.new_metadata = {"id": [], "sex": [], "key": [], "duration": []}

    def process_file_metadata(self, fullfilename, y, D, **preprocess_params):
        target_sample_rate = preprocess_params['target_sample_rate']
        signal_length = float(y.size)/target_sample_rate
        self.new_metadata['duration'].append(signal_length)
        file_name_segments = fullfilename.split(os.path.sep)
        sex = self.df.get_value(int(file_name_segments[-3]), 'SEX')
        self.new_metadata['id'].append(int(file_name_segments[-3]))
        self.new_metadata['sex'].append(sex)
        key = '{}/{}/{}'.format(sex, file_name_segments[-3], file_name_segments[-2])
        dataset_name = os.path.splitext(os.path.split(fullfilename)[-1])[0]
        self.new_metadata['key'].append('{}/{}'.format(key, dataset_name))
        return key, dataset_name

    def save_metadata(self, metadata_file_name):
        pd.DataFrame(data=self.new_metadata).to_csv(metadata_file_name)


class UrbanSound8K_metadata_handler:
    """Creates keys and a metadata table for the noise dataset given its metadata and a filename"""
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path, index_col=0)
        self.df['duration'] = self.df['end'] - self.df['start']
        self.new_metadata = {"duration": [], "salience": [], "Class": [], "key": []}

    def process_file_metadata(self, fullfilename, y, D, **preprocess_params):
        filename = os.path.split(fullfilename)[-1]
        salience = self.df.get_value(filename, 'salience')
        class_ = self.df.get_value(filename, 'class')
        self.new_metadata['duration'].append(self.df.get_value(filename, 'duration'))
        self.new_metadata['salience'].append(salience)
        self.new_metadata['Class'].append(class_)
        key = '{}/{}'.format(salience, class_)
        dataset_name = os.path.splitext(filename)[0]
        self.new_metadata['key'].append('{}/{}'.format(key, dataset_name))
        return key, dataset_name

    def save_metadata(self, metadata_file_name):
        pd.DataFrame(data=self.new_metadata).to_csv(metadata_file_name)


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
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger('preprocessing')

    # Make the data set
    with open(args.settings) as settings_file:
        settings = json.load(settings_file)


        logger.debug('settings {}'.format(settings))
        logger.info('looking in {} for raw audio data with file'.format(settings['data_directory']))
        logger.info('will store spectrogram-HDF5 file in {}'.format(settings['spectrogram_output_file']))
        logger.info('will store waveform-HDF5 file in {}'.format(settings['waveform_output_file']))


        key_maker = None
        key_maker_name = '{}_metadata_handler'.format(settings['dataset_type'])
        if key_maker_name in globals():
            logger.debug('using the {} as the metadata handler'.format(key_maker_name))
            key_maker = globals()[key_maker_name](settings['metadata_file'])


        print('Starting preprocessing...')
        make_stft_dataset(data_dir=settings['data_directory'],
                          spectrogram_output_file=settings['spectrogram_output_file'],
                          waveform_output_file=settings['waveform_output_file'],
                          compression=settings['compression'],
                          compression_opts=settings['compression_opts'],
                          key_maker=key_maker,
                          **settings['processing_parameters'])
        print('\nFinished!!')


        if key_maker is not None:
            logger.info('will store metadata file in {}'.format(settings['metadata_output_file']))
            key_maker.save_metadata(settings['metadata_output_file'])


if __name__ == '__main__':
    main()
