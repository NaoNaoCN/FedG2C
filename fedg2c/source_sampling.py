"""Reproducible graph coverage across rounds of budgeted source training."""
import torch

from .tensor_shards import IndexedSourceGraphs


def cycle_indices(size, start, count, seed):
    """Read a slice from independently shuffled, consecutive dataset passes."""
    if size < 1 or start < 0 or count < 1:
        raise ValueError("Require a nonempty pool, nonnegative start and positive count")
    indices = []
    while len(indices) < count:
        cycle, position = divmod(start, size)
        generator = torch.Generator().manual_seed(seed + cycle)
        order = torch.randperm(size, generator=generator)
        take = min(size - position, count - len(indices))
        indices.extend(order[position:position + take].tolist())
        start += take
    return indices


class ShuffledGraphCycle:
    """Index source shards; bounded CPU cache instead of an eager graph list."""
    def __init__(self, paths, seed, counts=None):
        self.graphs = IndexedSourceGraphs(paths, counts)
        self.seed = seed
        if not self.graphs:
            raise ValueError("No source graphs")

    def indices(self, start, count):
        return cycle_indices(len(self.graphs), start, count, self.seed)

    def iter_selected(self, indices):
        return (self.graphs[index] for index in indices)
