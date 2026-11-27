"""Exact group cosines, proper refinements, and size-preserving tree controls."""
from itertools import combinations
import numpy as np


def group_cosines(vectors, groups):
    result = np.full(len(groups), np.nan)
    for index, ids in enumerate(groups):
        if len(ids) < 2:
            continue
        block = np.asarray(vectors[ids], dtype=np.float64)
        total = block.sum(axis=0)
        result[index] = (total @ total - np.einsum('ij,ij->', block, block)) / (len(ids) * (len(ids) - 1))
    return result


class TreeMetrics:
    def __init__(self, codes, epsilon=1e-6):
        self.codes = np.asarray(codes, dtype=np.int64)
        self.n = len(codes)
        self.eps = epsilon
        self.prefixes = {}; self.groups = {}; self.assignment = {}; self.counts = {}
        for depth in [1, 2, 3]:
            prefixes, inverse = np.unique(self.codes[:, :depth], axis=0, return_inverse=True)
            self.prefixes[depth] = [tuple(prefix) for prefix in prefixes.tolist()]
            self.assignment[depth] = inverse
            self.groups[depth] = [np.flatnonzero(inverse == i) for i in range(len(prefixes))]
            self.counts[depth] = np.bincount(inverse)
        self.parents = {}
        self.roots = {}
        for depth in [1, 2, 3]:
            self.roots[depth] = np.array([self.assignment[1][ids[0]] for ids in self.groups[depth]])
            if depth > 1:
                self.parents[depth] = np.array([self.assignment[depth - 1][ids[0]] for ids in self.groups[depth]])
        self.edge_masks = {depth: (self.counts[depth] >= 2) &
                           (self.counts[depth] < self.counts[depth - 1][self.parents[depth]])
                           for depth in [2, 3]}
        self.path_mask = self.edge_masks[3] & self.edge_masks[2][self.parents[3]]
        self.pairs = {}
        for depth in [2, 3]:
            left, right, group = [], [], []
            for index, ids in enumerate(self.groups[depth]):
                for i, j in combinations(ids.tolist(), 2):
                    left.append(i); right.append(j); group.append(index)
            self.pairs[depth] = tuple(np.asarray(a, dtype=np.int64) for a in [left, right, group])

    def observed(self, vectors):
        return {depth: group_cosines(vectors, self.groups[depth]) for depth in [1, 2, 3]}

    def gap_values(self, cosines, child_depth):
        parent_depth = child_depth - 1
        parent_count = len(self.groups[parent_depth])
        pairs = self.counts[child_depth] * (self.counts[child_depth] - 1) / 2
        within_counts = np.bincount(self.parents[child_depth], weights=pairs, minlength=parent_count)
        within_sum = np.bincount(self.parents[child_depth],
                                 weights=pairs * np.nan_to_num(cosines[child_depth]), minlength=parent_count)
        parent_pairs = self.counts[parent_depth] * (self.counts[parent_depth] - 1) / 2
        across_counts = parent_pairs - within_counts
        valid = (within_counts > 0) & (across_counts > 0)
        result = np.full(parent_count, np.nan)
        result[valid] = (within_sum[valid] / within_counts[valid] -
                         (parent_pairs[valid] * cosines[parent_depth][valid] - within_sum[valid]) / across_counts[valid])
        return result

    def units(self, cosines):
        """One value per child edge/path/parent; retain root IDs for cluster CIs."""
        result = {}
        for depth in [2, 3]:
            tag = '12' if depth == 2 else '23'
            mask = self.edge_masks[depth]
            delta = cosines[depth][mask] - cosines[depth - 1][self.parents[depth][mask]]
            roots = self.roots[depth][mask]
            result['delta_' + tag] = (delta, roots)
            result['positive_' + tag] = ((delta > self.eps).astype(float), roots)
            gap = self.gap_values(cosines, depth)
            good = np.isfinite(gap)
            result['gap_' + tag] = (gap[good], self.roots[depth - 1][good])
        path_indices = np.flatnonzero(self.path_mask)
        second = self.parents[3][path_indices]
        first = self.parents[2][second]
        delta12 = cosines[2][second] - cosines[1][first]
        delta23 = cosines[3][path_indices] - cosines[2][second]
        result['path_monotonic'] = (((delta12 > self.eps) & (delta23 >= -self.eps)).astype(float), self.roots[3][path_indices])
        result['path_strict'] = (((delta12 > self.eps) & (delta23 > self.eps)).astype(float), self.roots[3][path_indices])
        return result

    def statistics(self, cosines):
        return {key: float(values.mean()) if len(values) else None
                for key, (values, _) in self.units(cosines).items()}

    def cluster_intervals(self, cosines, rng, repetitions=2000):
        roots = len(self.groups[1])
        sampled = rng.multinomial(roots, np.ones(roots) / roots, size=repetitions)
        result = {}
        for key, (values, root_ids) in self.units(cosines).items():
            if not len(values):
                result[key] = None
                continue
            sums = np.bincount(root_ids, weights=values, minlength=roots)
            counts = np.bincount(root_ids, minlength=roots)
            denominator = sampled @ counts
            eligible = denominator > 0
            means = (sampled[eligible] @ sums) / denominator[eligible]
            result[key] = np.quantile(means, [.025, .975]).tolist()
        return result


class ConditionalControls:
    """Cache only Gram matrices within first-prefix subtrees, not the N×N matrix."""
    def __init__(self, tree, vectors):
        self.tree = tree
        n = tree.n
        self.local = np.empty(n, dtype=np.int64)
        self.offset = np.empty(n, dtype=np.int64)
        self.width = np.empty(n, dtype=np.int64)
        flattened = []
        cursor = 0
        for ids in tree.groups[1]:
            block = np.asarray(vectors[ids], dtype=np.float64)
            gram = block @ block.T
            self.local[ids] = np.arange(len(ids))
            self.offset[ids] = cursor
            self.width[ids] = len(ids)
            flattened.append(gram.ravel())
            cursor += len(ids) ** 2
        self.gram = np.concatenate(flattened)

    def permute(self, parent_depth, rng):
        mapping = np.arange(self.tree.n)
        for ids in self.tree.groups[parent_depth]:
            mapping[ids] = rng.permutation(ids)
        return mapping

    def cosines(self, mapping, observed, first_changed_depth):
        cosines = {depth: observed[depth] for depth in [1, 2, 3]}
        for depth in range(first_changed_depth, 4):
            left, right, group = self.tree.pairs[depth]
            left = mapping[left]; right = mapping[right]
            assert np.array_equal(self.offset[left], self.offset[right])
            values = self.gram[self.offset[left] + self.local[left] * self.width[left] + self.local[right]]
            sums = np.bincount(group, weights=values, minlength=len(self.tree.groups[depth]))
            pairs = self.tree.counts[depth] * (self.tree.counts[depth] - 1) / 2
            cosines[depth] = np.divide(sums, pairs, out=np.full(len(pairs), np.nan), where=pairs > 0)
        return cosines


def holm_adjust(pvalues):
    keys = sorted(pvalues, key=lambda key: pvalues[key])
    result = {}; previous = 0
    for rank, key in enumerate(keys):
        value = min(1.0, (len(keys) - rank) * pvalues[key])
        previous = max(previous, value)
        result[key] = previous
    return result
