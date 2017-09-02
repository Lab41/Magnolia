"""Script for partitioning preprocessed data"""


import os.path
import argparse
import logging.config
import json
import h5py


class PartitionTreeFilter:
    """Holds a filter (node) structure of the partition tree"""

    def __init__(self, filter_desc):
        self._id = filter_desc["id"]
        self._only = [] if "only" not in filter_desc else filter_desc["only"]
        self._except = [] if "except" not in filter_desc else filter_desc["except"]
        self._only_is_list = isinstance(self._only, list)
        self._except_is_list = isinstance(self._except, list)
        self._pass_all = (self._only is [] and self._except is [])
        self._splits = []

    def add_split(self, split):
        self._splits.append(split)

    def get_splits(self):
        return self._splits

    def apply(self, categories):
        # returned matched list of categories
        pass


class PartitionTreeGroup:
    """Holds a group (node) structure of the partition tree"""

    def __init__(self, group_desc):
        self._name = group_desc["name"]
        self._only = [] if "only" not in group_desc else group_desc["only"]
        self._except = [] if "except" not in group_desc else group_desc["except"]
        self._only_is_list = isinstance(self._only, list)
        self._except_is_list = isinstance(self._except, list)
        self._pass_all = (self._only is [] and self._except is [])
        self._file_names = []

    def add_file_name(self, file_name):
        self._file_names.append(file_name)

    def get_file_names(self):
        return self._file_names

class PartitionTreeSplit:
    """Holds a split (edge) structure of the partition tree"""

    def __init__(self, split_desc):
        self._source = split_desc["source"]
        self._target = split_desc["target"]
        self._file_names = []

    def add_file_name(self, file_name):
        self._file_names.append(file_name)

    def get_file_names(self):
        return self._file_names


def build_partition_graphs(fsg):
    """Returns the partition graphs from lists of filters, splits, and groups"""

    graphs = []
    for graph_elements in fsg:
        filters = graph_elements["filters"]
        splits = graph_elements["splits"]
        groups = graph_elements["groups"]

        graph = {}



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

        input_file = h5py.File(settings["preprocessed_file"], "r")
        # metadata = pd.read_csv(settings['metadata_file'])
        # metadata['duration'] = metadata['end'] - metadata['start']
        # metadata = metadata[(metadata['salience'] == 1) & (metadata['duration'] >= 2.)]
        # print(metadata[metadata['class'] != 'children_playing'].head())

        graphs = build_partition_graphs(settings["partition_graphs"])


if __name__ == '__main__':
    main()
