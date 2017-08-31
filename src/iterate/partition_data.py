"""Script for partitioning preprocessed data"""


import os.path
import argparse
import logging.config
import json


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

        logger.debug('settings {}'.format(settings))


if __name__ == '__main__':
    main()
