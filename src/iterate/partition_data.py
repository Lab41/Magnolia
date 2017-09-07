"""Script for partitioning preprocessed data

TODO: need to add logging, documentation, and error handling
"""


import os.path
import argparse
import logging.config
import json
import numpy as np
import pandas as pd
import h5py
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
        partition_graph_desc = json.load(open(settings['partition_graphs_file']))

        rng = np.random.RandomState(settings['rng_seed'])
        logger.debug('settings {}'.format(settings))

        input_file = h5py.File(settings["preprocessed_file"], "r")
        metadata = pd.read_csv(settings['metadata_file'], index_col=0)

        for graph in partition_graph_desc['partition_graphs']:
            root_node = build_partition_graph(settings['output_directory'],
                                              graph)
            root_node.apply(df=metadata, key=graph['data_label'], rng=rng,
                            niter=10000, niter_success=5000)


if __name__ == '__main__':
    main()
