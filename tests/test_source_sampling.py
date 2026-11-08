"""Coverage and reproducibility guarantees for the formal source budget."""
import unittest

import torch

from fedg2c.source_sampling import cycle_indices


class SourceSamplingTests(unittest.TestCase):
    def test_pcba_is_covered_once_across_ten_rounds(self):
        rounds = [cycle_indices(10000, round_id * 1000, 1000, 2026) for round_id in range(10)]
        indices = [index for chunk in rounds for index in chunk]
        self.assertEqual(len(indices), 10000)
        self.assertEqual(set(indices), set(range(10000)))

    def test_round_slices_match_continuous_stream(self):
        whole = cycle_indices(1113, 0, 10000, 2026)
        sliced = [index for round_id in range(10)
                  for index in cycle_indices(1113, round_id * 1000, 1000, 2026)]
        self.assertEqual(whole, sliced)
        for offset in range(0, 8904, 1113):
            self.assertEqual(set(whole[offset:offset + 1113]), set(range(1113)))

    def test_seeds_are_reproducible_and_do_not_change_global_rng(self):
        before = torch.random.get_rng_state().clone()
        first = cycle_indices(1000, 1000, 1000, 2026)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(first, cycle_indices(1000, 1000, 1000, 2026))
        self.assertNotEqual(first, cycle_indices(1000, 1000, 1000, 2027))


if __name__ == '__main__':
    unittest.main()
