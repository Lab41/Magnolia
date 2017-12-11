"""Class and helper function for optimizing the partition of category labels

TODO: need to add more logging and documentation
"""


import logging.config
import numpy as np
from magnolia.utils.MTHM import fuzzy_mthm


logger = logging.getLogger('partitioning')


def split_categories(category_labels, category_populations,
                     desired_fractions,
                     split_category,
                     dataframe,
                     grouped_dataframe,
                     stratify_categories,
                    #  stratify_categories_summaries,
                     rng, **opt_args):
    results = {}
    knapsack_capacities = (category_populations.sum()*desired_fractions).astype(int)
    _, actual_populations, category_groups = fuzzy_mthm(
        np.ones(category_populations.size), category_populations,
        knapsack_capacities)
    actual_fractions = actual_populations/knapsack_capacities.sum()
    # category_groups, actual_fractions = PartitionOptimizer(category_labels,
    #                                                        category_populations,
    #                                                        desired_fractions,
    #                                                        split_category,
    #                                                        dataframe,
    #                                                        grouped_dataframe,
    #                                                        stratify_categories,
    #                                                     #    stratify_categories_summaries,
    #                                                        rng).optimize(**opt_args)
    for i in range(len(desired_fractions)):
        results[i] = category_labels[category_groups == i]

    return results, actual_fractions


class PartitionOptimizer:
    """Helper class that finds an optimal partition of categories through random search"""

    class EnsurePopulation:
        def __init__(self, num_splits=2):
            self._num_splits = num_splits
        def __call__(self, **kwargs):
            return (np.unique(kwargs['x_new']).size == self._num_splits)

    class UnitStep:
        def __init__(self, random_state, stepsize=1, num_splits=2):
            self.stepsize = stepsize
            self._num_splits = num_splits
            self._rng = random_state
        def __call__(self, x):
            return np.clip(x + self._rng.randint(-self.stepsize, self.stepsize + 1, size=x.size),
                           0, self._num_splits - 1)

    def __init__(self, category_labels, category_populations, desired_fractions,
                 split_category, dataframe, grouped_dataframe,
                 stratify_categories,
                #  stratify_categories, stratify_categories_summaries,
                 random_state):
        self._rng = random_state
        self._category_labels = category_labels
        self._category_populations = category_populations
        self._total_population = category_populations.sum()
        self._desired_fractions = desired_fractions
        self._split_category = split_category
        self._dataframe = dataframe
        self._grouped_dataframe = grouped_dataframe
        self._stratify_categories = stratify_categories
        # self._stratify_categories_summaries = stratify_categories_summaries
        self._actual_fractions = np.zeros_like(desired_fractions)
        self._number_of_splits = len(desired_fractions)
        self._accept_test = self.EnsurePopulation(num_splits=self._number_of_splits)
        self._group_indices = np.arange(desired_fractions.size)
        self._initial_vector = self._create_initial_vector()
        self._grouped_dataframe_counts = grouped_dataframe.size()
        self._grouped_dataframe_counts /= self._grouped_dataframe_counts.sum()
        self._grouped_counts = self._grouped_dataframe_counts.values

    def _create_initial_vector(self):
        vec = np.zeros_like(self._category_populations)
        assignments = {}
        for i in range(self._number_of_splits):
            assignments[i] = []
            desired_pop_fraction = int(self._desired_fractions[i]*self._total_population)
            last_positive = False
            min_diff = np.inf
            for j, category_populations in enumerate(self._category_populations):
                if abs(category_populations - desired_pop_fraction) < min_pop:
                    pass
        # vec = self._rng.randint(0, self._number_of_splits, size=self._category_populations.size)
        # while not self._accept_test(x_new=vec):
        #     vec = self._rng.randint(0, self._number_of_splits, size=self._category_populations.size)
        # return vec

    def _calculate_population_fractions(self, x):
        for i in range(self._number_of_splits):
            selected_category_populations = self._category_populations[x == i]
            self._actual_fractions[i] = selected_category_populations.sum()/self._total_population

    def _loss_function(self, x):
        if self._grouped_dataframe is not None:
            self._calculate_population_fractions(x)
            ave_pop_fracs = (np.abs(self._actual_fractions - self._desired_fractions)/self._desired_fractions).mean()

            means = [len(self._group_indices)*ave_pop_fracs]
            # for split_group_index in self._group_indices:
            #     labels_in_group = self._category_labels[x == split_group_index]
            #     split_df = self._dataframe.loc[self._dataframe[self._split_category].isin(labels_in_group)]
            #     grouped_split_df = split_df.groupby(self._stratify_categories)
            #     grouped_split_df_summary = grouped_split_df.size()
            #     grouped_split_df_summary /= grouped_split_df_summary.sum()
            #     diff = self._grouped_dataframe_counts.sub(grouped_split_df_summary, fill_value=0).values
            #     means.append((np.abs(diff)/self._grouped_counts).mean())

            return np.mean(means)
        else:
            self._calculate_population_fractions(x)
            return (np.abs(self._actual_fractions - self._desired_fractions)/self._desired_fractions).sum()

    def optimize(self, stepsize=1, niter=1000, niter_success=None, **kwargs):
        take_step = self.UnitStep(self._rng, stepsize=stepsize, num_splits=self._number_of_splits)

        global_minimum_vec = self._initial_vector
        global_minimum_eval = self._loss_function(global_minimum_vec)
        global_minimum_streak = 0
        current_vec = global_minimum_vec
        current_eval = global_minimum_eval
        early_stopped = False


        for i in range(niter):
            current_vec = take_step(current_vec)
            while not self._accept_test(x_new=current_vec):
                current_vec = take_step(current_vec)
            current_eval = self._loss_function(current_vec)

            if current_eval < global_minimum_eval:
                global_minimum_eval = current_eval
                global_minimum_vec = current_vec
                global_minimum_streak = 0
            global_minimum_streak += 1

            if niter_success is not None and global_minimum_streak > niter_success:
                early_stopped = True
                logger.debug('Optimizer stopping early at {}'.format(i))
                break

        if not early_stopped:
            logger.debug('Optimizer finished all iterations ({})'.format(niter))

        self._calculate_population_fractions(global_minimum_vec)
        return global_minimum_vec, self._actual_fractions
