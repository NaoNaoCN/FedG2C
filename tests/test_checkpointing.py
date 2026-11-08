"""Check retention boundaries, preserved records and unchanged FL results."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from fedg2c.checkpointing import should_save_checkpoint
from pretrain_federated_fedg2c import main as pretrain
from scripts.fedg2c.prune_round_checkpoints import apply_plan, make_plan

REPO = Path(__file__).resolve().parents[1]


class CheckpointRetentionTests(unittest.TestCase):
    def test_round_boundaries_and_short_runs(self):
        self.assertEqual([r for r in range(100) if should_save_checkpoint(r, 100)],
                         [19, 39, 59, 79, 99])
        self.assertEqual([r for r in range(41) if should_save_checkpoint(r, 41)], [19, 39, 40])
        self.assertTrue(should_save_checkpoint(0, 1))
        for arguments in [(0, 10, 0), (10, 10, 20), (0, 0, 20), (0, 1, True)]:
            with self.assertRaises(ValueError):
                should_save_checkpoint(*arguments)

    def test_prune_preserves_json_latest_references_and_other_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / 'experiment/node/cora/stage1'
            for index in range(41):
                folder = stage / f'round_{index:03d}'
                for name in ('client_0', 'server'):
                    child = folder / name
                    child.mkdir(parents=True)
                    (child / 'shared_backbone.pt').write_bytes(b'weights')
                    (child / 'metrics.json').write_text('{}')
                (folder / 'server/server_probes.pt').write_bytes(b'probes')
                (folder / 'server/run_metadata.json').write_text('{}')
            pinned = stage / 'round_007/server/shared_backbone.pt'
            (root / 'execution_plan.json').write_text(json.dumps({'checkpoint': str(pinned)}))
            target = root / 'stage2/target_model.pt'
            target.parent.mkdir()
            target.write_bytes(b'target')
            private = stage / 'client_0_private_training_state.pt'
            private.write_bytes(b'private')
            original_files = set(root.rglob('*'))
            plan = make_plan(root, 20)
            self.assertEqual(plan['groups'][0]['kept_round_ids'], [7, 19, 39, 40])
            self.assertEqual(plan['delete_count'], 37 * 3)
            self.assertTrue(pinned.exists())  # inventory is read-only
            apply_plan(plan)
            remaining = set(root.rglob('*'))
            self.assertEqual(original_files - remaining,
                             {Path(item['path']) for item in plan['delete_files']})
            self.assertTrue(target.exists())
            self.assertTrue(private.exists())
            self.assertEqual(make_plan(root, 20)['delete_count'], 0)

    def test_apply_validates_all_candidates_before_deleting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(3):
                folder = root / f'round_{index:03d}/server'
                folder.mkdir(parents=True)
                (folder / 'shared_backbone.pt').write_bytes(b'weights')
            plan = make_plan(root, 20)
            first, second = [Path(item['path']) for item in plan['delete_files']]
            second.write_bytes(b'changed weights')
            with self.assertRaises(ValueError):
                apply_plan(plan)
            self.assertTrue(first.exists())
            second.write_bytes(b'weights')
            metrics = first.with_name('metrics.json')
            metrics.write_text('{}')
            plan['delete_files'].append({'path': str(metrics), 'bytes': 2})
            with self.assertRaises(ValueError):
                apply_plan(plan)
            self.assertTrue(first.exists())
            self.assertTrue(metrics.exists())

    def test_sparse_saves_preserve_final_parameters_and_every_round_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            server = json.loads((REPO / 'config/fedg2c_server_response.json').read_text())
            server.update(mode='fedavg')
            server_path = base / 'server.json'
            server_path.write_text(json.dumps(server))
            for interval in (1, 20):
                output = base / str(interval)
                pretrain(['--smoke', '--server-config', str(server_path), '--rounds', '21',
                          '--local-steps', '1', '--checkpoint-interval', str(interval),
                          '--batch-size', '3', '--physical-batch-size', '2',
                          '--output', str(output)])
                self.assertEqual(len(list(output.glob('round_*/client_0/metrics.json'))), 21)
                self.assertEqual(len(list(output.glob('round_*/server/run_metadata.json'))), 21)
                saved = [int(path.parent.parent.name.split('_')[1]) for path in
                         sorted(output.glob('round_*/server/shared_backbone.pt'))]
                self.assertEqual(saved, list(range(21)) if interval == 1 else [19, 20])
            states = [torch.load(base / str(i) / 'round_020/server/shared_backbone.pt',
                                 map_location='cpu', weights_only=True)['state_dict'] for i in (1, 20)]
            self.assertEqual(states[0].keys(), states[1].keys())
            for name in states[0]:
                self.assertTrue(torch.equal(states[0][name], states[1][name]), name)


if __name__ == '__main__':
    unittest.main()
