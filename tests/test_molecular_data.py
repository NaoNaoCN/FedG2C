"""Regression checks for molecular tensor export and explicit input vocabularies."""
import io
import unittest

import torch
from torch_geometric.data import Data

from fedg2c.prepare_molecular_data import (
    SchemaStats, graph_record, molecular_schema, stratified_indices,
)


class MolecularExportTests(unittest.TestCase):
    def test_export_detaches_graph_slices_from_dataset_storage(self):
        storage = torch.arange(900000).reshape(-1, 9)
        graph = Data(x=storage[:3], edge_index=torch.tensor([[0, 1], [1, 0]]))
        record = graph_record(graph, "slice")
        serialized = io.BytesIO()
        torch.save(record, serialized)
        self.assertLess(len(serialized.getvalue()), 20000)
        self.assertNotEqual(record["x"].data_ptr(), storage.data_ptr())
        self.assertTrue(torch.equal(record["x"], storage[:3]))

    def test_ogb_vocabulary_does_not_depend_on_observed_categories(self):
        schema = molecular_schema()
        stats = SchemaStats(schema)
        stats.update({"x": torch.zeros(2, 9, dtype=torch.long),
                      "edge_attr": torch.zeros(2, 3, dtype=torch.long)})
        self.assertEqual(stats.schema(), schema)
        invalid = torch.zeros(2, 9, dtype=torch.long)
        invalid[0, 0] = schema["node_cardinalities"][0]
        with self.assertRaises(ValueError):
            stats.update({"x": invalid, "edge_attr": torch.zeros(2, 3, dtype=torch.long)})

    def test_target_selection_stays_inside_official_partition(self):
        dataset = [Data(y=torch.tensor([[int(i % 5 == 0)]])) for i in range(100)]
        partition = torch.arange(20, 80)
        first, counts = stratified_indices(dataset, partition, 20, 2026, "test")
        second, _ = stratified_indices(dataset, partition, 20, 2026, "test")
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 20)
        self.assertTrue(set(first).issubset(set(partition.tolist())))
        self.assertEqual(counts, {"positive": 4, "negative": 16})

    def test_imdb_degree_is_local_and_source_has_no_labels(self):
        graph = Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                     y=torch.tensor([1]), num_nodes=3)
        record = graph_record(graph, "imdb", degree_features=True)
        self.assertNotIn("y", record)
        self.assertTrue(torch.equal(record["x"], torch.tensor([[0.5], [1.0], [0.5]])))


if __name__ == "__main__":
    unittest.main()
