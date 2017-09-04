"""Script for partitioning preprocessed data"""


import os.path
import argparse
import logging.config
import json
import numpy as np
import pandas as pd
import h5py
from magnolia.utils.partition_optimizer import PartitionOptimizer


def split_categroies(category_labels, category_populations,
                     desired_fractions,
                     rng, logger, **opt_args):
    results = {}
    category_groups, actual_fractions = PartitionOptimizer(category_populations,
                                                           desired_fractions,
                                                           rng, logger).optimize(**opt_args)
    for i in range(len(desired_fractions)):
        results[i] = category_labels[category_groups == i]

    return results, actual_fractions


def main():
    """Partition preprocessed data."""

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Partition preprocessed data.')
    parser.add_argument('--settings', '-s',
                        default='../../settings/partition_template.json',
                        help='partitioning settings JSON file')
    parser.add_argument('--logger_settings', '-l',
                        default='../../settings/logging.conf',
                        help='logging configuration file')
    parser.add_argument('--logger_name', '-n',
                        default='partitioning',
                        help='name of logger')
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger(args.logger_name)

    # Make the partitions
    with open(args.settings) as settings_file:
        settings = json.load(settings_file)

        rng = np.random.RandomState(settings['rng_seed'])
        logger.debug('settings {}'.format(settings))

        input_file = h5py.File(settings["preprocessed_file"], "r")
        metadata = pd.read_csv(settings['metadata_file'], index_col=0)

        metadata = metadata[(metadata['salience'] == 1) &
                            (metadata['duration'] >= 2.) &
                            (metadata['class'] != 'children_playing')]

        class_summary = metadata['class'].value_counts()
        categories, actual_split = split_categroies(class_summary.index.values,
                                                    class_summary.values,
                                                    np.array([0.8, 0.2]),
                                                    rng, logger,
                                                    niter=100000,
                                                    niter_success=50000)
        for category_number in categories:
            print(metadata.loc[metadata['class'].isin(categories[category_number])]['key'])


if __name__ == '__main__':
    main()
