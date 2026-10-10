"""Run with the training Python; small CPU models only."""
from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT))
import jeff_lora_train
from jeff_lora_train import atomic_directory, atomic_report, run_epoch
from coreai.jeff_lora import save_lora
from projects.tetris_training import overnight


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class DeadlineTests(unittest.TestCase):
    def test_final_save_precedes_post_training_evaluation(self):
        source = inspect.getsource(jeff_lora_train.main)
        self.assertLess(source.index('report["save_state"] = "weights_saved"'),
                        source.index('report["adapter"]["train"] = run_epoch'))

    def test_atomic_checkpoint_and_report_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'adapter'
            def fail(path):
                path.mkdir()
                (path/'partial').write_text('incomplete')
                raise RuntimeError('interrupted write')
            with self.assertRaisesRegex(RuntimeError, 'interrupted write'):
                atomic_directory(dest, fail)
            self.assertFalse(dest.exists())
            self.assertFalse((Path(temp)/'adapter.partial').exists())
            atomic_directory(dest, lambda path: (path.mkdir(), (path/'complete').write_text('yes')))
            self.assertEqual((dest/'complete').read_text(), 'yes')
            atomic_report(Path(temp)/'report.json', {'save_state': 'adapter_saved'})
            atomic_report(Path(temp)/'report.json', {'save_state': 'weights_saved'})
            self.assertEqual(json.loads((Path(temp)/'report.json').read_text())['save_state'], 'weights_saved')

    def test_saved_adapter_can_rebuild_merged_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, adapter, merged = (root/name for name in ('base', 'adapter', 'merged'))
            base.mkdir()
            save_file({'linear.weight': torch.eye(2)}, str(base/'model.safetensors'))
            (base/'config.json').write_text('{}')
            layer = SimpleNamespace(key='linear.weight', A=torch.tensor([[1., 0.]]),
                                    B=torch.tensor([[0.], [1.]]), rank=1, scale=2.)
            readout = torch.nn.Linear(2, 2, bias=False)
            save_lora(adapter, [layer], readout, {'base_checkpoint': str(base)})
            atomic_report(root/'report.json', {'save_state': 'adapter_saved'})
            (root/'parity_rows.json').write_text('[]')
            controller = object.__new__(overnight.Controller)
            candidate = {'adapter': str(adapter), 'merged': str(merged)}
            controller.recover_saved_merge(candidate)
            self.assertTrue(torch.equal(load_file(str(merged/'model.safetensors'))['linear.weight'],
                                        torch.tensor([[1., 0.], [2., 1.]])))
            self.assertTrue((merged/'config.json').is_file())
            self.assertFalse((root/'merged.partial').exists())
            self.assertEqual(json.loads((root/'report.json').read_text())['save_state'], 'weights_saved')

    def test_expired_deadline_prevents_optimizer_update(self):
        model, head = Tiny(), torch.nn.Linear(4, 2)
        before = head.weight.detach().clone()
        optimizer = torch.optim.SGD(head.parameters(), lr=.1)
        stats = run_epoch(model, head, [{'ids':[1,2], 'label':0, 'n':2}],
                          1, 0, 'cpu', optimizer, stop_at=time.time()-1)
        self.assertEqual(stats['n'], 0)
        self.assertTrue(stats['stopped_early'])
        self.assertTrue(torch.equal(before, head.weight))

    def test_expired_deadline_skips_the_warmup_optimizer_steps(self):
        model, head = Tiny(), torch.nn.Linear(4, 2)
        layers = [SimpleNamespace(A=torch.zeros(1, 1), B=torch.zeros(1, 1))]
        before = head.weight.detach().clone()
        steps = []
        def make_optimizer():
            optimizer = torch.optim.SGD(head.parameters(), lr=.1)
            optimizer.step = lambda *args, **kwargs: steps.append(1)
            return optimizer
        rows = [{'ids': [1, 2], 'label': 0, 'n': 2}] * 4
        with torch.no_grad():
            timed = jeff_lora_train.timed_warmup(model, head, layers, rows, 2, 0, 'cpu', make_optimizer,
                                                 stop_at=time.time() - 1)
        self.assertEqual((steps, timed['n'], timed['stopped_early']), ([], 0, True))
        self.assertTrue(torch.equal(before, head.weight))
        with torch.no_grad():  # a live deadline still runs both warm-up steps and restores the weights afterwards
            timed = jeff_lora_train.timed_warmup(model, head, layers, rows, 2, 0, 'cpu', make_optimizer,
                                                 stop_at=time.time() + 60)
        self.assertEqual((len(steps), timed['n'], timed['stopped_early']), (2, 4, False))
        self.assertTrue(torch.equal(before, head.weight))

    def test_main_passes_the_deadline_to_the_warmup(self):
        source = inspect.getsource(jeff_lora_train.main)
        self.assertIn('timed_warmup(backbone, readout, layers, train_enc, a.batch, pad_id, device, make_optimizer, stop_at)',
                      source)

    def test_completed_steps_call_recovery_hook(self):
        model, head = Tiny(), torch.nn.Linear(4, 2)
        calls = []
        before = head.weight.detach().clone()
        # Importing inference helpers can leave the process in no-grad mode.
        with torch.no_grad():
            stats = run_epoch(model, head, [{'ids':[1,2], 'label':0, 'n':2}]*3,
                              2, 0, 'cpu', torch.optim.SGD(head.parameters(), lr=.1),
                              stop_at=time.time()+60, step_callback=lambda *args: calls.append(args))
            self.assertFalse(torch.is_grad_enabled())
        self.assertFalse(torch.equal(before, head.weight))
        self.assertEqual(calls, [(1,2), (2,3)])
        self.assertEqual(stats['n'], 3)
        self.assertFalse(stats['stopped_early'])

    def test_controller_deadline_stops_its_child(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=Path(temp)/'run', deadline=(now+timedelta(hours=1)).isoformat(),
                                   train_cutoff=(now+timedelta(minutes=30)).isoformat(), port=8789)
            with patch.object(overnight, 'load', return_value={'summary': {'pieces_mean':1,'lines_mean':0}}):
                controller = overnight.Controller(args)
            with self.assertRaises(TimeoutError):
                controller.run([sys.executable, '-c', 'import time; time.sleep(60)'],
                               'deadline-check', until=time.time()+.2)
            status = json.loads((args.output/'status.json').read_text())
            record = json.loads((args.output/'deadline-check.command.json').read_text())
            self.assertIsNone(status['child_pid'])
            self.assertLess(record['exit_code'], 0)
            self.assertLess(record['seconds'], 15)
            controller.lock.close()

    def test_round_planning_reserves_validation_time(self):
        self.assertEqual(overnight.round_size(1200), 0)
        self.assertEqual(overnight.round_size(10000), 2048)
        self.assertLessEqual(overnight.round_size(2400)*2.3+1200, 2400)

    def test_final_recovery_after_cutoff_preserves_best_and_deadline(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(output=root, deadline=(now+timedelta(minutes=30)).isoformat(),
                                   train_cutoff=(now-timedelta(minutes=2)).isoformat(),
                                   port=8789, resume_final=True)
            best = {'name': 'round-02'}
            for key in ('adapter', 'merged', 'build'):
                (root/key).mkdir()
                best[key] = str(root/key)
            overnight.atomic_json(root/'best.json', best)
            overnight.atomic_json(root/'status.json', {
                'status': 'failed', 'controller_pid': 1, 'deadline': args.deadline,
                'train_cutoff': args.train_cutoff, 'updated_utc': now.isoformat(),
                'error': 'candidate load failed', 'best': best, 'rounds': [{'name': 'round-02'}],
            })
            controller = overnight.Controller(args)
            try:
                self.assertEqual(controller.best, best)
                self.assertEqual(controller.state['deadline'], args.deadline)
                self.assertEqual(controller.state['rounds'], [{'name': 'round-02'}])
                self.assertEqual(controller.state['recovery_history'][0]['previous_error'],
                                 'candidate load failed')
                self.assertNotIn('error', controller.state)
            finally:
                controller.lock.close()

    def test_final_recovery_dispatch_never_trains(self):
        with patch.object(overnight, 'Controller') as constructor, patch.object(overnight.signal, 'signal'):
            overnight.main(['--output', '/unused', '--deadline', 'saved', '--train-cutoff', 'saved',
                            '--resume-final'])
            constructor.return_value.finalize.assert_called_once_with()
            constructor.return_value.recover_pending_round.assert_called_once_with()
            constructor.return_value.execute.assert_not_called()

    def test_saved_round_without_post_eval_can_be_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'round-01'/'model'
            for name in ('adapter', 'merged'):
                (output/name).mkdir(parents=True)
            for name in ('adapter_model.safetensors', 'readout.safetensors', 'adapter_config.json'):
                (output/'adapter'/name).touch()
            for name in ('model.safetensors', 'readout.safetensors'):
                (output/'merged'/name).touch()
            (output/'parity_rows.json').write_text('[]')
            atomic_report(output/'report.json', {'save_state': 'weights_saved', 'adapter': {}, 'history': [{'n': 4}]})
            candidate = {'name': 'round-01', 'adapter': str(output/'adapter'),
                         'merged': str(output/'merged'), 'build': str(root/'round-01'/'build'/'coreai')}
            pending = {'number': 1, 'count': 512, 'lr': 1e-4, 'candidate': candidate}
            controller = object.__new__(overnight.Controller)
            controller.root = root
            controller.best = {'name': 'pilot-v1', 'dev': {'pieces_mean': 1, 'lines_mean': 0}}
            controller.state = {'rounds': [], 'pending_candidate': pending}
            controller.update = lambda **fields: controller.state.update(fields)
            def run(_command, name):
                path = root/f'{name}.log'
                if name.startswith('parity'):
                    path.write_text('{"rows": 4, "argmax_matches": 4}')
                return path
            controller.run = run
            controller.start_server = lambda _candidate: None
            controller.stop_server = lambda: None
            controller.gameplay = lambda *_args: overnight.atomic_json(
                root/'round-01'/'dev'/'report.json', {'summary': {'pieces_mean': 2, 'lines_mean': 0}})
            controller.prune_derived = lambda _candidate: None
            (root/'round-01'/'dev').mkdir()
            self.assertTrue(controller.finish_round(pending))
            self.assertEqual(controller.best['name'], 'round-01')
            self.assertEqual(controller.state['rounds'][0]['training_validation'], 'interrupted_after_save')
            self.assertIsNone(controller.state['pending_candidate'])


if __name__ == '__main__':
    unittest.main()
