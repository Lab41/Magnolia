"""Class and functions for building partition graph

TODO: need to add logging, documentation, and error handling
"""


import os.path
import logging
import numpy as np
import logging.config
from magnolia.utils.partition_optimizer import split_categories


logger = logging.getLogger('partitioning')


def get_all_groups(graph_element, group_list, is_node=True):
    if is_node:
        if graph_element.terminal():
            group_list.append(graph_element)
        else:
            for split in graph_element.splits():
                get_all_groups(split, group_list, False)
    else:
        get_all_groups(graph_element.destination(), group_list, True)


def get_group_path(group_name, graph_element, is_node=True):
    groups = []
    get_all_groups(graph_element, groups, is_node)
    for group in groups:
        if group.id() == group_name:
            return group.compute_filename()


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

    # determine root node
    for split in splits:
        sources.add(split['source'])
        targets.add(split['target'])

    root_id = sources - sources.intersection(targets)
    # ensure only one root node
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
        stratify_categories = []
        for i, split in enumerate(self._splits):
            fractions[i] = split.fraction
            if split.has_split_category():
                if split_category is None:
                    split_category = split.split_category()
                else:
                    # TODO: throw exception
                    # ensure all splits refer to the same category
                    assert(split_category == split.split_category())
            if stratify_categories == []:
                stratify_categories = split.stratify_catagories()
            elif split.stratify_catagories() != []:
                # TODO: throw exception
                # ensure all stratify categories are the same
                assert(frozenset(stratify_categories) == frozenset(stratify_categories))

        if split_category is not None:
            # TODO: throw a proper exception
            # NOTE: cannot split along category and preserve stratification w.r.t. that category
            assert(split_category not in stratify_categories)
            split_category_summary = df[split_category].value_counts()
            # TODO: throw a proper exception
            # must have more categories than number of splits
            assert(len(split_category_summary.index) >= len(fractions))
            grouped_df = None
            if stratify_categories != []:
                grouped_df = df.groupby(stratify_categories)
            # stratify_classes_summaries = []
            # for stratify_category in stratify_categories:
            #     stratify_classes_summaries.append(df[stratify_category].value_counts(normalize=True))
            categories, actual_splits = split_categories(split_category_summary.index.values,
                                                         split_category_summary.values,
                                                         fractions,
                                                         split_category,
                                                         df,
                                                         grouped_df,
                                                         stratify_categories,
                                                        #  stratify_classes_summaries,
                                                         **kwargs)
            for category_number in categories:
                split_df = df.loc[df[split_category].isin(categories[category_number])]
                self._splits[category_number].apply(df=split_df, key=key,
                                                    actual_split=actual_splits[category_number],
                                                    **kwargs)
        elif stratify_categories != []:
            rng = kwargs['rng']
            total_indices = {}
            for keys, group in df.groupby(stratify_categories):
                count = len(group.index)
                indices = rng.permutation(group.index.values)
                lower_index = 0
                for i, fraction in enumerate(fractions):
                    upper_index = lower_index + int(count*fraction)
                    selected_indices = indices[lower_index:upper_index]
                    if i in total_indices:
                        total_indices[i] = np.concatenate((total_indices[i], selected_indices))
                    else:
                        total_indices[i] = selected_indices
                    lower_index = upper_index
            count = len(df.index)
            for i, split in enumerate(self._splits):
                indices = total_indices[i]
                split.apply(df=df.loc[indices], key=key,
                            actual_split=float(len(indices))/count,
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

    def id(self):
        return self._id

    def terminal(self):
        return False


class PartitionGraphGroup:
    def __init__(self, id, filename_base):
        self._id = id
        self._filename_base = filename_base

    def apply(self, df, **kwargs):
        os.makedirs(self._filename_base, exist_ok=True)
        full_file_path = self.compute_filename()
        df.to_csv(full_file_path)

    def compute_filename(self):
        return os.path.join(self._filename_base, '{}.csv'.format(self._id))

    def id(self):
        return self._id

    def terminal(self):
        return True


class PartitionGraphSplit:
    def __init__(self, split_desc):
        self._source = split_desc['source']
        self._target = split_desc['target']
        if "split_on" in split_desc:
            self._split_on = split_desc['split_on']
        self._stratify_wrt = []
        if "stratify_wrt" in split_desc:
            self._stratify_wrt = split_desc['stratify_wrt']
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

    def stratify_catagories(self):
        return self._stratify_wrt

    def terminal(self):
        return False
