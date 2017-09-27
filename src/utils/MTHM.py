"""Function to solve the multi-knapsack packing problem with "fuzzy" capacities

Inspired by the approximate MTHM algorithm in http://www.or.deis.unibo.it/kp/Chapter6.pdf

TODO: add logging, error handling, and documentation (the "fuzzy" part is novel)
"""

import numpy as np


def fuzzy_mthm(p, w, c, std_tol=1e-3, max_balance_iter=100, min_mthm_iter=10):
    """Find a solution to the multi-knapsack problem with "fuzzy" capacities

    p: profits (array)
    w: weights (array)
    c: capacities of knapsacks (array)
    """
    # globals
    NO_GROUP = -1

    packed_profit, assignments = mthm(p, w, c, NO_GROUP, min_mthm_iter)
    packsizes = np.zeros_like(c)
    for i in range(c.size):
        packsizes[i] = w[assignments == i].sum()

    # start rearranging knapsacks for items that were left out
    if NO_GROUP in assignments:
        unpacked_profits = p[assignments == NO_GROUP]
        unpacked_weights = w[assignments == NO_GROUP]
        unpacked_indices = np.arange(len(p))[assignments == NO_GROUP]
        knapsack_diffs = c - packsizes

        # overstuff knapsacks with the most room with the leftover items
        for k in range(unpacked_indices.size):
            unpacked_weight = unpacked_weights[k]
            min_knapsack_index = None
            for i, knapsack_diff in enumerate(knapsack_diffs):
                if min_knapsack_index is None:
                    min_knapsack_index = i
                elif np.abs(unpacked_weight - knapsack_diff) < \
                    np.abs(unpacked_weight - knapsack_diffs[min_knapsack_index]):
                    min_knapsack_index = i
            assignments[unpacked_indices[k]] = min_knapsack_index
            packsizes[min_knapsack_index] += unpacked_weight
            packed_profit += unpacked_profits[k]
            knapsack_diffs[min_knapsack_index] = c[min_knapsack_index] - packsizes[min_knapsack_index]

        # print((c - packsizes)/c)

        # make to knapsacks share the loads more evenly by reducing the variance
        # of the relative differences of the knapsacks and their total capacities
        relative_diffs = knapsack_diffs/c
        # print(relative_diffs.std())
        if relative_diffs.std() > std_tol:
            iter_count = 0
            while iter_count < max_balance_iter:
                # find most overstuffed knapsack
                most_overstuffed_knapsack_index = np.argmin(relative_diffs)
                # isolate these items and sort them by weight
                most_overstuffed_knapsack_indices = np.arange(len(p))[assignments == most_overstuffed_knapsack_index]
                most_overstuffed_knapsack_weights = w[assignments == most_overstuffed_knapsack_index]
                argsort_most_overstuffed_knapsack_weights = np.argsort(most_overstuffed_knapsack_weights)
                most_overstuffed_knapsack_indices_sorted = most_overstuffed_knapsack_indices[argsort_most_overstuffed_knapsack_weights]
                most_overstuffed_knapsack_weights_sorted = most_overstuffed_knapsack_weights[argsort_most_overstuffed_knapsack_weights]

                best_weight_index = 0
                best_sub_weight_index = 0
                best_sub_knapsack_index = 0
                prev_ave_rel_diff = None
                for weight_index in range(most_overstuffed_knapsack_weights_sorted.size):
                    overstuffed_weight_sum = most_overstuffed_knapsack_weights_sorted[:weight_index].sum()

                    for knapsack_index in range(packsizes.size):
                        if knapsack_index == most_overstuffed_knapsack_index:
                            continue
                        # isolate these items and sort them by weight
                        knapsack_weights = w[assignments == knapsack_index]
                        argsort_knapsack_weights = np.argsort(knapsack_weights)
                        knapsack_weights_sorted = knapsack_weights[argsort_knapsack_weights]

                        for sub_weight_index in range(knapsack_weights_sorted.size):
                            weight_sum = knapsack_weights_sorted[:sub_weight_index].sum()

                            cur_ave_rel_diff = np.abs(knapsack_diffs[most_overstuffed_knapsack_index] + overstuffed_weight_sum - weight_sum)/c[most_overstuffed_knapsack_index]
                            cur_ave_rel_diff += np.abs(knapsack_diffs[knapsack_index] - overstuffed_weight_sum + weight_sum)/c[knapsack_index]
                            cur_ave_rel_diff *= 0.5
                            if prev_ave_rel_diff is None or cur_ave_rel_diff < prev_ave_rel_diff:
                                best_weight_index = weight_index
                                best_sub_weight_index = sub_weight_index
                                best_sub_knapsack_index = knapsack_index
                                prev_ave_rel_diff = cur_ave_rel_diff

                knapsack_indices = np.arange(len(p))[assignments == best_sub_knapsack_index]
                knapsack_weights = w[assignments == best_sub_knapsack_index]
                argsort_knapsack_weights = np.argsort(knapsack_weights)
                knapsack_indices_sorted = knapsack_indices[argsort_knapsack_weights]

                most_overstuffed_knapsack_indices_sorted = most_overstuffed_knapsack_indices_sorted[:best_weight_index]
                knapsack_indices_sorted = knapsack_indices_sorted[:best_sub_weight_index]

                assignments[most_overstuffed_knapsack_indices_sorted] = best_sub_knapsack_index
                assignments[knapsack_indices_sorted] = most_overstuffed_knapsack_index

                for i in range(c.size):
                    packsizes[i] = w[assignments == i].sum()
                knapsack_diffs = c - packsizes
                relative_diffs = knapsack_diffs/c

                if relative_diffs.std() <= std_tol:
                    break

                iter_count += 1

    # print((c - packsizes)/c)
    # print(relative_diffs.std())
    return packed_profit, packsizes, assignments


def mthm(p, w, c, NO_GROUP, min_iter):
    # pre-sort knapsacks and profits/weights
    c_sorted_args = np.argsort(c) # ascending
    pw_sorted_args = np.argsort(p/w)[::-1] # descending
    p_copy = p[pw_sorted_args]
    w_copy = w[pw_sorted_args]

    # initialize variables
    n = len(p)
    m = len(c)
    # y holds the group assignments for each item
    y = np.empty(n, dtype=int)
    y.fill(NO_GROUP)

    # initial solution
    c_bar = c[c_sorted_args]
    z = 0
    for i in range(m):
        z = greed_ys(p_copy, w_copy, i, c_bar, y, z, NO_GROUP)

    # only continue if greedy solution fails
    if z != p.sum():
        #rearrangement
        z = 0
        c_bar = c[c_sorted_args]
        i = 0
        for j in range(n - 1, -1, -1):
            if y[j] > NO_GROUP:
                l = None
                # for ll in range(i):
                for ll in range(i, m):
                    if w_copy[j] <= c_bar[ll]:
                        l = ll
                        break
                if l is None:
                    # for ll in range(i, m):
                    for ll in range(i):
                        if w_copy[j] <= c_bar[ll]:
                            l = ll
                            break
                if l is None:
                    y[j] = NO_GROUP
                else:
                    y[j] = l
                    c_bar[l] -= w_copy[j]
                    z += p_copy[j]
                    if l < m - 1:
                        i = l + 1
                    else:
                        i = 0
        for i in range(m):
            z = greed_ys(p_copy, w_copy, i, c_bar, y, z, NO_GROUP)

        # first improvement
        for j in range(n):
            if y[j] > NO_GROUP:
                for k in range(j + 1, n):
                    if y[k] > NO_GROUP and y[k] != y[j]:
                        h = None
                        l = None
                        if w_copy[j] > w_copy[k]:
                            h = j
                            l = k
                        else:
                            h = k
                            l = j
                        d = w_copy[h] - w_copy[l]
                        min_null_weight = np.inf
                        for u in range(n):
                            if y[u] == NO_GROUP and min_null_weight > w_copy[u]:
                                min_null_weight = w_copy[u]
                        if d <= c_bar[y[l]] and c_bar[y[h]] + d >= min_null_weight:
                            max_null_profit = -np.inf
                            t = None
                            for u in range(n):
                                if y[u] == NO_GROUP and w_copy[u] <= d + c_bar[y[h]] and p_copy[u] > max_null_profit:
                                    max_null_profit = p_copy[u]
                                    t = u
                            c_bar[y[h]] += d - w_copy[t]
                            c_bar[y[l]] -= d
                            temp_yt = y[t]
                            y[t] = y[h]
                            y[h] = y[l]
                            y[l] = y[temp_yt]
                            z += p_copy[t]

        # repeat second improvement until no further improvement is found
        iter_count = 0
        while True:
            z_prev = z
            iter_count += 1
            # second improvement
            for j in range(n - 1, -1, -1):
                if y[j] > NO_GROUP:
                    c_bar_temp = c_bar[y[j]] + w_copy[j]
                    Y = set()
                    for k in range(n):
                        if y[k] == NO_GROUP and w_copy[k] <= c_bar_temp:
                            Y.add(k)
                            c_bar_temp -= w_copy[k]
                    pk_sum = 0.0
                    for k in Y:
                        pk_sum += p_copy[k]
                    if pk_sum > p_copy[j]:
                        for k in Y:
                            y[k] = y[j]
                            c_bar[y[j]] = c_bar_temp
                            y[j] = NO_GROUP
                            z += pk_sum - p_copy[j]
            if iter_count >= min_iter and z_prev == z:
                break

    # undo sorting
    y_return = np.zeros_like(y)
    for j in range(n):
        if y[j] != NO_GROUP:
            y_return[pw_sorted_args[j]] = c_sorted_args[y[j]]
        else:
            y_return[pw_sorted_args[j]] = NO_GROUP
    return z, y_return


def greed_ys(p, w, i, c_bar, y, z, NO_GROUP):
    for j in range(len(p)):
        if y[j] == NO_GROUP and w[j] <= c_bar[i]:
            y[j] = i
            c_bar[i] -= w[j]
            z += p[j]
    return z


def main():
    n = 250
    m = 10
    # equal item profits
    p = np.ones(n)
    # weights
    w = np.random.randint(1, 51, size=n)
    # capacities
    c = []
    lower_index = 0
    for i in range(m):
        if i < m - 1:
            upper_index = np.random.randint(w.size//(m + 1), w.size//m) + lower_index
            c.append(w[lower_index:upper_index].sum())
        else:
            c.append(w[lower_index:].sum())
        lower_index = upper_index
    c = np.array(c)


    chunked_w = np.empty(n//2)
    for i in range(n//2):
        chunked_w[i] = w[2*i] + w[2*i + 1]
    p = p[:chunked_w.size]


    print(chunked_w, c)
    print(fuzzy_mthm(p, chunked_w, c))


if __name__ == "__main__":
    main()
