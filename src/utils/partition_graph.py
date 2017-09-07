"""Class and functions for building partition graph

TODO: need to add logging, documentation, and error handling
"""


import os.path
import numpy as np
import logging.config
from magnolia.utils.partition_optimizer import split_categories


def _recursively_build_tree_split(split, filters, groups, splits, path):
    for filter_desc in filters:
        if split._target == filter_desc['id']:
            node = PartitionGraphFilter(filter_desc)
            split.set_destination(node)
            _recursively_build_tree_node(node, filters, groups, splits, path)
            return

    for group_name in groups:
        if split._target == group_name:
            group = PartitionGraphGroup(group_name, path)
            split.set_destination(group)
            return

def _recursively_build_tree_node(node, filters, groups, splits, path):
    new_path = os.path.join(path, node._id)

    for split_desc in splits:
        if split_desc['source'] == node._id:
            split = PartitionGraphSplit(split_desc)
            node.add_split(split)
            _recursively_build_tree_split(split, filters, groups, splits, new_path)


def build_partition_graph(output_directory, partition_graph):
    filters = partition_graph['filters']
    groups = partition_graph['groups']
    splits = partition_graph['splits']
    sources = set()
    targets = set()

    for split in splits:
        sources.add(split['source'])
        targets.add(split['target'])

    root_id = sources - sources.intersection(targets)
    # TODO: throw a proper exception
    assert(len(root_id) == 1)
    root_id = root_id.pop()

    root_node = None
    for filter in filters:
        if root_id == filter['id']:
            root_node = PartitionGraphFilter(filter)

    _recursively_build_tree_node(root_node, filters, groups,
                                 splits, output_directory)

    return root_node


class PartitionGraphFilter:
    def __init__(self, filter_desc):
        self._id = filter_desc['id']
        self._pass_through = ('pandas_query' not in filter_desc)
        if not self._pass_through:
            self._query = filter_desc['pandas_query']
        self._splits = []

    def apply(self, df, key, **kwargs):
        if not self._pass_through:
            df = df.query(self._query)

        fractions = np.zeros(len(self._splits))
        split_category = None
        for i, split in enumerate(self._splits):
            fractions[i] = split.fraction
            if split.has_split_category():
                split_category = split.split_category()

        if split_category is not None:
            class_summary = df[split_category].value_counts()
            # TODO: throw a proper exception
            assert(len(class_summary.index.values) >= len(fractions))
            categories, actual_splits = split_categories(class_summary.index.values,
                                                         class_summary.values,
                                                         fractions,
                                                         **kwargs)
            for category_number in categories:
                split_df = df.loc[df[split_category].isin(categories[category_number])]
                self._splits[category_number].apply(df=split_df, key=key,
                                                    actual_split=actual_splits[category_number],
                                                    **kwargs)
        else:
            count = len(df.index)
            indices = kwargs['rng'].permutation(count)
            lower_index = 0
            for split in self._splits:
                upper_index = lower_index + int(count*split.fraction)
                selected_indices = indices[lower_index:upper_index]
                split.apply(df=df.iloc[selected_indices], key=key,
                            actual_split=float(len(selected_indices))/count,
                            **kwargs)
                lower_index = upper_index


    def add_split(self, split):
        total_fraction = 0.0
        for s in self._splits:
            total_fraction += s.fraction
        total_fraction += split.fraction
        # TODO: throw a proper exception
        assert(total_fraction <= 1.0)
        self._splits.append(split)

    def splits(self):
        return self._splits


class PartitionGraphGroup:
    def __init__(self, id, filename_base):
        self._id = id
        self._filename_base = filename_base

    def apply(self, df, **kwargs):
        filename = '{}.csv'.format(self._id)
        os.makedirs(self._filename_base, exist_ok=True)
        full_file_path = os.path.join(self._filename_base, filename)
        df.to_csv(full_file_path)


class PartitionGraphSplit:
    def __init__(self, split_desc):
        self._source = split_desc['source']
        self._target = split_desc['target']
        if "split_on" in split_desc:
            self._split_on = split_desc['split_on']
        self.fraction = split_desc['fraction']
        self._destination = None

    def apply(self, df, key, actual_split, **kwargs):
        self._destination.apply(df=df, key=key, **kwargs)

    def set_destination(self, destination):
        self._destination = destination

    def destination(self):
        return self._destination

    def has_split_category(self):
        return "_split_on" in self.__dict__

    def split_category(self):
        return self._split_on
