"""Check exact formulas and tree-preserving controls against brute force."""
from itertools import combinations
import numpy as np
from semantic_metrics import TreeMetrics, ConditionalControls, group_cosines, holm_adjust


def main():
    rng = np.random.default_rng(41)
    vectors = rng.normal(size=(32, 9))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    codes = np.array([[i//16, (i%16)//8, (i%8)//4] for i in range(32)])
    tree = TreeMetrics(codes)
    observed = tree.observed(vectors)
    for depth in [1,2,3]:
        for index, ids in enumerate(tree.groups[depth]):
            brute = np.mean([vectors[i] @ vectors[j] for i,j in combinations(ids,2)])
            np.testing.assert_allclose(observed[depth][index], brute, atol=1e-12)
    for depth in [2,3]:
        gaps = tree.gap_values(observed, depth)
        for parent, ids in enumerate(tree.groups[depth-1]):
            within, across = [], []
            for i,j in combinations(ids,2):
                bucket = within if tree.assignment[depth][i] == tree.assignment[depth][j] else across
                bucket.append(vectors[i] @ vectors[j])
            np.testing.assert_allclose(gaps[parent], np.mean(within)-np.mean(across), atol=1e-12)
    controls = ConditionalControls(tree, vectors)
    for parent_depth in [1,2]:
        for _ in range(4):
            mapping = controls.permute(parent_depth, rng)
            np.testing.assert_array_equal(tree.assignment[parent_depth][mapping], tree.assignment[parent_depth])
            randomized = controls.cosines(mapping, observed, parent_depth+1)
            brute = tree.observed(vectors[mapping])
            for depth in [1,2,3]:
                np.testing.assert_allclose(randomized[depth], brute[depth], atol=1e-12)
    # A single-item group is undefined; unchanged membership is not a proper refinement.
    singleton_tree = TreeMetrics(np.array([[0,0,0],[0,0,1],[1,1,1]]))
    singleton_scores = singleton_tree.observed(vectors[:3])
    assert np.isnan(singleton_scores[1][1])
    assert not singleton_tree.edge_masks[2].any()
    # Screenshot condition allows a plateau in S2 -> S3, with actual smaller sets.
    scores = {1: np.full(2, .1), 2: np.full(4, .2), 3: np.full(8, .2)}
    assert tree.statistics(scores)['path_monotonic'] == 1
    assert tree.statistics(scores)['path_strict'] == 0
    adjusted = holm_adjust({'a':.01,'b':.03})
    assert adjusted == {'a':.02,'b':.03}
    print('PASS: exact pair means, sibling gaps, both conditional nulls, singleton handling, proper subsets, path inequality, Holm')


if __name__ == '__main__':
    main()
