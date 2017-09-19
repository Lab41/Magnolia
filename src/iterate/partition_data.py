"""Script for partitioning preprocessed data

TODO: need to add logging, documentation, and error handling
"""


import os.path
import argparse
import logging.config
import json
import numpy as np
import pandas as pd
from magnolia.utils.partition_graph import build_partition_graph


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
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger('partitioning')

    # Make the partitions
    with open(args.settings) as settings_file:
        settings = json.load(settings_file)
        preprocessing_settings = json.load(open(settings['preprocessing_settings']))
        partition_graph_desc = json.load(open(settings['partition_graphs_file']))

        rng = np.random.RandomState(settings['rng_seed'])
        logger.debug('settings {}'.format(settings))

        if 'description' in settings:
            logger.info('description = {}'.format(settings['description']))

        metadata = pd.read_csv(preprocessing_settings['metadata_output_file'], index_col=0)

        for graph in partition_graph_desc['partition_graphs']:
            root_node = build_partition_graph(settings['output_directory'],
                                              graph)
            root_node.apply(df=metadata, key=graph['data_label'], rng=rng,
                            niter=10000, niter_success=5000)


if __name__ == '__main__':
    main()
