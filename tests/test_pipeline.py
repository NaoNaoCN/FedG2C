"""Exercise stage boundaries, exact budgets and validation-only model selection."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fedg2c.data import write_smoke_manifest
from fedg2c.pipeline import REPO, execute, resolve_config
from fedg2c.target_data import load_target_manifest
from fedg2c.target_model import TargetConfig
from fedg2c.target_smoke import write_smoke_manifest as write_target
from fedg2c.target_train import load_target_model, run_target


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        source = write_smoke_manifest(cls.folder/'source')
        target = write_target(cls.folder/'target')
        cls.raw = json.loads((REPO/'config/fedg2c_molecular_pipeline.json').read_text())
        cls.raw['run'].update(output=str(cls.folder/'steps'), device=os.environ.get('FEDG2C_TEST_DEVICE','cpu'))
        cls.raw['data'] = {'source_manifest':str(source), 'target_manifest':str(target)}
        cls.raw['stage1']['training'].update(rounds=2, local_budget={'unit':'steps','amount':3},
                                            views_per_graph=2, log_every=0)
        cls.raw['stage1']['module_b'].update(steps=1, calibration_graphs=1, diagnostic_graphs=1,
                                            directions=1, readout_pairs=2)
        cls.raw['stage2']['target'].update(epochs=2, evaluate_test=False)
        cls.config_path = cls.folder/'pipeline.json'
        cls.config_path.write_text(json.dumps(cls.raw))
        cls.config = resolve_config(cls.config_path)
        execute(cls.config, config_path=cls.config_path)
        cls.checkpoint = cls.folder/'steps/stage1/round_001/server/shared_backbone.pt'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_steps_two_stages_and_reload(self):
        root = self.folder/'steps'
        self.assertEqual(json.loads((root/'status.json').read_text())['state'], 'completed')
        policy = json.loads((root/'stage1/checkpoint_policy.json').read_text())
        self.assertEqual(policy['interval'], 20)
        self.assertEqual(policy['saved_round_ids'], [1])
        self.assertFalse((root/'stage1/round_000/server/shared_backbone.pt').exists())
        for path in (root/'stage1').glob('round_*/client_*/metrics.json'):
            self.assertEqual(len(json.loads(path.read_text())['steps']), 3)
        summary = json.loads((root/'summary.json').read_text())
        self.assertEqual(set(summary), {'none','lora','adapter','prompt'})
        for mode, report in summary.items():
            self.assertFalse(report['test_evaluated'])
            self.assertIsNone(report['test'])
            self.assertTrue(report['frozen_base_unchanged'])
            _, meta = load_target_model(root/'stage2'/mode/'target_model.pt')
            self.assertEqual(meta['source_round_id'], 1)
        # A completed output is never reused or overwritten.
        with self.assertRaises(FileExistsError):
            execute(self.config)

    def test_full_client_epochs_are_full_passes(self):
        cfg = copy.deepcopy(self.config)
        cfg['run']['output'] = str(self.folder/'epochs')
        cfg['stage1']['training'].update(rounds=1, local_budget={'unit':'epochs','amount':2})
        execute(cfg, 'stage1')
        for path in (self.folder/'epochs/stage1/round_000').glob('client_*/metrics.json'):
            report = json.loads(path.read_text())
            self.assertEqual(len(report['steps']), 3*2*2)
            self.assertEqual([s['epoch'] for s in report['steps']], [1]*6+[2]*6)
            indices = report['sampling']['graph_indices']
            self.assertEqual(set(indices[:3]), {0,1,2})
            self.assertEqual(set(indices[3:]), {0,1,2})

    def test_batched_pipeline_preserves_logical_steps_and_reload_all_adaptations(self):
        cfg = copy.deepcopy(self.config)
        cfg['run']['output'] = str(self.folder/'batched')
        cfg['stage1']['training'].update(rounds=3, batch_size=5, physical_batch_size=3,
                                        checkpoint_interval=2,
                                        local_budget={'unit':'epochs','amount':2}, views_per_graph=1)
        cfg['stage2']['target'].update(microbatch_size=3, eval_batch_size=2)
        execute(cfg)
        root = self.folder/'batched'
        policy = json.loads((root/'stage1/checkpoint_policy.json').read_text())
        self.assertEqual(policy['interval'], 2)
        self.assertEqual(policy['saved_round_ids'], [1, 2])
        saved = [path.parent.parent.name for path in
                 sorted((root/'stage1').glob('round_*/server/shared_backbone.pt'))]
        self.assertEqual(saved, ['round_001', 'round_002'])
        for path in (root/'stage1/round_000').glob('client_*/metrics.json'):
            report = json.loads(path.read_text())
            self.assertEqual([s['views'] for s in report['steps']], [3, 3])
            self.assertEqual(report['physical_batch_size'], 3)
        for mode in ('none','lora','adapter','prompt'):
            model, meta = load_target_model(root/'stage2'/mode/'target_model.pt')
            self.assertTrue(model.frozen_base_unchanged())
            self.assertEqual(meta['config']['microbatch_size'], 3)
            self.assertEqual(meta['config']['eval_batch_size'], 2)

    def test_validation_only_does_not_open_test_files(self):
        manifest = json.loads(Path(self.config['data']['target_manifest']).read_text())
        for entry in manifest['splits']['test']:
            entry['path'] = 'missing_' + entry['path']
        path = self.folder/'target/no_test.json'
        path.write_text(json.dumps(manifest))
        _, splits = load_target_manifest(path, TargetConfig(), include_test=False)
        self.assertEqual(set(splits), {'train','valid'})
        with self.assertRaises(FileNotFoundError):
            load_target_manifest(path, TargetConfig(), include_test=True)
        cfg = copy.deepcopy(self.config)
        cfg['run']['output'] = str(self.folder/'stage2_only')
        cfg['data']['target_manifest'] = str(path)
        cfg['stage2']['checkpoint'] = str(self.checkpoint)
        cfg['stage2']['adaptations'] = ['lora']
        execute(cfg, 'stage2')
        self.assertFalse((self.folder/'stage2_only/stage1').exists())

    def test_early_stop_metric_direction_and_first_tie(self):
        target = TargetConfig(epochs=10, early_stopping_patience=2, evaluate_test=False,
                              selection_metric='roc_auc')
        validation = [{'loss':v,'roc_auc':auc} for v,auc in
                      [(0.4,0.6),(0.5,0.8),(0.3,0.8),(0.2,0.7)]]
        with patch('fedg2c.target_train.evaluate', side_effect=validation) as mocked:
            report = run_target(self.checkpoint, self.config['data']['target_manifest'], target,
                                self.folder/'early_stop')
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(report['best_epoch'], 2)
        self.assertEqual(report['epochs_ran'], 4)
        self.assertTrue(report['stopped_early'])
        self.assertIsNone(report['test'])
        # Undefined ranking metrics must fail instead of silently choosing a model.
        with patch('fedg2c.target_train.evaluate', return_value={'loss':0.2,'roc_auc':None}):
            with self.assertRaises(ValueError):
                run_target(self.checkpoint, self.config['data']['target_manifest'], target,
                           self.folder/'undefined')

    def test_invalid_config_fails_before_output(self):
        for location, key in [('run','threds'), ('stage2','epohs')]:
            raw = copy.deepcopy(self.raw)
            raw[location][key] = 1
            path = self.folder/'invalid.json'
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                resolve_config(path)
        cfg = copy.deepcopy(self.config)
        cfg['run']['output'] = str(self.folder/'invalid_selection')
        cfg['stage2']['target']['selection_metric'] = 'accuracy'
        with self.assertRaises(ValueError):
            execute(cfg)
        self.assertFalse(Path(cfg['run']['output']).exists())

    def test_default_protocol_evaluates_test_once_after_selection(self):
        target = TargetConfig(epochs=2)
        values = [{'loss':0.4}, {'loss':0.3}, {'loss':0.5, 'roc_auc':0.6}]
        with patch('fedg2c.target_train.evaluate', side_effect=values) as mocked:
            result = run_target(self.checkpoint, self.config['data']['target_manifest'], target,
                                self.folder/'test_once')
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(result['best_epoch'], 2)
        self.assertTrue(result['test_evaluated'])
        self.assertEqual(result['test']['roc_auc'], 0.6)

    def test_node_targets_run_all_adaptations_on_identical_fewshot_support(self):
        manifest = write_target(self.folder/'node_target', level='node')
        supports = []
        for mode in ('none','lora','adapter','prompt'):
            target = TargetConfig(adaptation=mode, epochs=2, max_nodes=8, k_shot=1,
                                  selection_metric='accuracy', evaluate_test=False)
            report = run_target(self.checkpoint, manifest, target, self.folder/f'node_{mode}')
            self.assertEqual(report['split_sizes']['train'], 2)
            self.assertIsNotNone(report['validation']['accuracy'])
            self.assertIsNone(report['test'])
            self.assertTrue(report['frozen_base_unchanged'])
            supports.append(report['support_indices_in_train'])
        self.assertTrue(all(support == supports[0] for support in supports))


if __name__ == '__main__':
    unittest.main()
