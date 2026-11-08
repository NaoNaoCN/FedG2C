"""Core invariants of the Gefra LODO experiment extension."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from fedg2c.edge_client import EdgePredictionClient
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.gefra_protocol import GRAPH_SOURCES, NODE_SOURCES, TARGET_BUDGETS
from fedg2c.model import SharedBackbone
from fedg2c.prepare_gefra_loo_data import graph_split_indices
from fedg2c.target_data import load_target_manifest, select_support
from fedg2c.target_model import TargetConfig
from fedg2c.tensor_shards import pack_records, record_at


class GefraLooTests(unittest.TestCase):
    def test_all_graph_and_node_lodo_roles_match_reference_cardinality(self):
        self.assertEqual(len(GRAPH_SOURCES), 4)
        self.assertEqual({len(v) for v in GRAPH_SOURCES.values()}, {3})
        self.assertEqual(len(NODE_SOURCES), 10)
        self.assertEqual({len(NODE_SOURCES[k]) for k in ('cora','citeseer','pubmed','photo','computers','wikics')}, {5})
        self.assertEqual(len(NODE_SOURCES['arxiv']), 3)
        self.assertEqual(len(NODE_SOURCES['reddit']), 3)
        self.assertEqual(len(NODE_SOURCES['airport']), 6)
        self.assertEqual(len(NODE_SOURCES['disease']), 6)
        self.assertEqual(TARGET_BUDGETS['arxiv'], (3000, 400, 0, 5e-4))

    def test_node_source_order_matches_gefra_target_configs(self):
        self.assertEqual(NODE_SOURCES['photo'],
                         ['cora','pubmed','citeseer','computers','wikics'])
        self.assertEqual(NODE_SOURCES['computers'],
                         ['cora','pubmed','citeseer','photo','wikics'])

    def test_graph_random_801010_split_is_deterministic_and_disjoint(self):
        first = graph_split_indices(103, 'chemhiv', 17)
        second = graph_split_indices(103, 'chemhiv', 17)
        self.assertEqual(first, second)
        self.assertEqual(tuple(len(first[k]) for k in ('train','valid','test')), (82, 10, 11))
        combined = sum((first[k] for k in ('train','valid','test')), [])
        self.assertEqual(sorted(combined), list(range(103)))

    def test_preparation_excludes_only_ineligible_source_graphs(self):
        from fedg2c.prepare_gefra_loo_data import prepare_gefra_loo_data

        def fake_graphs(cache, name, feature_policy):
            graphs = []
            for index in range(5):
                edges = (torch.empty((2, 0), dtype=torch.long) if index == 0 else
                         torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
                graphs.append(Data(x=torch.full((2, 3), float(index)), edge_index=edges,
                                   y=torch.tensor([index % 2])))
            return graphs, {'input_dim':3, 'edge_dim':0}, {'objective':'binary', 'num_outputs':1}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, output = root/'cache', root/'prepared'
            cache.mkdir()
            with patch('fedg2c.prepare_gefra_loo_data.load_graph', side_effect=fake_graphs):
                source_path, target_path = prepare_gefra_loo_data(cache, output, seed=17,
                                                                  target='proteins')
            source = json.loads(source_path.read_text())
            target = json.loads(target_path.read_text())
            report = json.loads((output/'preparation_report.json').read_text())
            self.assertEqual([client['id'] for client in source['clients']],
                             ['chemhiv','chempcba','imdb-binary'])
            self.assertEqual([client['graph_counts'] for client in source['clients']],
                             [[4],[4],[4]])
            for shard in report['source_shards']:
                payload = torch.load(output/shard['path'], weights_only=True)
                self.assertNotIn('y', payload)
            self.assertEqual([item['excluded_no_nonself_edge_ids'] for item in report['sources']],
                             [[0],[0],[0]])
            self.assertEqual(sum(len(target['splits'][split]) for split in ('train','valid','test')), 5)
            self.assertEqual(report['target']['raw_graphs'], 5)
            from fedg2c.pipeline import validate_prepared_roles
            cfg = {'data':{'prepare':{'kind':'gefra_loo_v1', 'target':'proteins'},
                           'source_manifest':str(source_path), 'target_manifest':str(target_path)},
                   'stage1':{'training':{'sampling_unit':'graph'}, 'module_a':{'num_experts':3}}}
            validate_prepared_roles(cfg)
            source['clients'][0]['id'] = 'proteins'
            source_path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'source client IDs/order'):
                validate_prepared_roles(cfg)

    def test_node_preparation_audits_isolated_roots_and_keeps_all_target_nodes(self):
        from fedg2c.prepare_gefra_loo_data import prepare_gefra_loo_data

        def fake_node(cache, name):
            nodes = torch.arange(9)
            edge = torch.stack([nodes, (nodes+1) % 9])
            labels = torch.arange(10) % 2
            record = {'x':torch.ones(10, 3), 'edge_index':torch.cat([edge,edge.flip(0)], dim=1),
                      'original_node_id':torch.arange(10), 'graph_id':name, 'y':labels}
            return record, labels, {'dataset':name, 'nodes':10, 'input_dim':3}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/'cache'
            cache.mkdir()
            with patch('fedg2c.prepare_gefra_loo_data.load_node', side_effect=fake_node):
                source_path, target_path = prepare_gefra_loo_data(cache, root/'prepared',
                                                                  seed=17, target='cora')
            source = json.loads(source_path.read_text())
            target = json.loads(target_path.read_text())
            report = json.loads((source_path.parent/'preparation_report.json').read_text())
            for client, audit in zip(source['clients'], report['sources']):
                record = torch.load(source_path.parent/client['graphs'][0], weights_only=True)
                self.assertNotIn('y', record)
                self.assertEqual(len(record['x']), 10)
                self.assertEqual(audit['excluded_no_nonself_edge_root_ids'], [9])
            target_ids = [i for entries in target['splits'].values() for i in entries[0]['indices']]
            self.assertEqual(sorted(target_ids), list(range(10)))
            _, samples = load_target_manifest(target_path, TargetConfig(max_nodes=8), include_test=True)
            isolate = next(sample for split in samples.values() for sample in split
                           if sample.sample_id == str(('cora', 9)))
            self.assertEqual(isolate.view.num_nodes, 1)
            self.assertEqual(isolate.root, 0)

    def test_packed_shard_offsets_keep_graph_boundaries_and_labels(self):
        records = [
            {'x': torch.full((2+i, 3), float(i)),
             'edge_index': torch.tensor([[0, 1], [1, 0]]),
             'graph_id': f'g{i}', 'y': torch.tensor([float(i)])}
            for i in range(3)
        ]
        packed = pack_records(records)
        self.assertEqual(packed['count'], 3)
        self.assertTrue(torch.equal(record_at(packed, 2)['x'], records[2]['x']))
        self.assertEqual(record_at(packed, 1)['graph_id'], 'g1')
        self.assertEqual(record_at(packed, 0)['y'].tolist(), [0.0])

    def test_graph_finetune_kshot_is_fixed_first_ids_per_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = {key: [] for key in ('train','valid','test')}
            for split, labels in (('train',[1,0,0,1,1,0,1,0]),
                                  ('valid',[0,1,0,1]), ('test',[0,1,0,1])):
                for index, label in enumerate(labels):
                    name = f'{split}_{index}.pt'
                    torch.save({'x': torch.ones(2, 3),
                                'edge_index': torch.tensor([[0,1],[1,0]]),
                                'graph_id': f'{split}_{index}',
                                'y': torch.tensor([float(label)])}, root/name)
                    splits[split].append({'path': name})
            manifest = {'schema_version':1,'protocol_id':'smoke','visibility':'split_graphs',
                        'task':{'level':'graph','objective':'binary','num_outputs':1},
                        'feature_schema':{'input_dim':3,'edge_dim':0},'splits':splits}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            cfg = TargetConfig(max_nodes=8, evaluate_test=False)
            loaded, samples = load_target_manifest(path, cfg, include_test=False)
            support, ids = select_support(samples['train'], loaded['task'], 1)
            self.assertEqual(ids, [0,1])
            self.assertEqual(len(support), 2)
            self.assertEqual([int(support.label(i).item()) for i in range(2)], [1,0])
            self.assertEqual(set(samples), {'train','valid'})

    def test_node_epoch_visits_each_nonisolated_root_once(self):
        torch.manual_seed(31)
        cfg = EdgePredictionConfig(edge_mask_rate=0.25, feature_mask_rate=0.2,
                                   hidden_dim=8, num_experts=2, num_heads=2,
                                   predictor_dim=8, predictor_heads=2, max_nodes=8,
                                   position_encoding='none', walk_steps=(1,))
        graph = {'x': torch.randn(6, 4),
                 'edge_index': torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,0],
                                             [1,0,2,1,3,2,4,3,5,4,0,5]])}
        from fedg2c.context import GraphView
        client = EdgePredictionClient('node', cfg, SharedBackbone(cfg), {'input_dim':4})
        result = client.pretrain(lambda: iter([GraphView(graph['x'],graph['edge_index'],None,
                                                         torch.arange(6))]),
                                 epochs=1, seed=19, batch_size=2, sampling_unit='node')
        self.assertEqual(result['total_views'], 6)
        self.assertEqual(len(result['steps']), 3)
        self.assertEqual([x['views'] for x in result['steps']], [2,2,2])
        self.assertEqual(result['root_coverage'][0]['roots_visited'], 6)
        self.assertTrue(result['root_coverage'][0]['complete'])


if __name__ == '__main__':
    unittest.main()
