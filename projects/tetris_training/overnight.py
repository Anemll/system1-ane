"""Bounded local training rounds, recovery checkpoints, and ANE gameplay selection.

Run with the training Python. All artifacts go under a new --output directory.
The controller stops child process groups it created when their deadline expires.
Recovery checkpoints contain LoRA/head weights; restarting uses a fresh optimizer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .dataset import audit, balanced_sample, read_jsonl, state_key, write_jsonl

REPO = Path(__file__).resolve().parents[2]
# Source always comes from this checkout; environments and artifacts stay external.
STORE = Path(os.environ.get('JEFF_TETRIS_STORE') or Path.home() / 'Models/jeff-tetris-training').expanduser().resolve()
BASE = Path(os.environ.get('JEFF_BASE_MODEL') or Path.home() / 'Models/jeff/jeff-base-v1.3').expanduser().resolve()
TRAIN_PY = sys.executable
SDK_PY = os.environ.get('COREAI_PYTHON') or sys.executable
PILOT_DATA = STORE / 'datasets/pilot-v1'
POOL = STORE / 'datasets/overnight-20261008-pool'
INITIAL = STORE / 'datasets/overnight-20261008-initial-corrections'


def load(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def timestamp(value):
    date = datetime.fromisoformat(value)
    if date.tzinfo is None:
        raise ValueError('timestamps require an explicit timezone')
    return date.timestamp()


def round_size(seconds_remaining, seconds_per_row=2.3):
    """Leave 20 minutes for fixed evaluation, export, compile, and dev games."""
    return max(0, min(2048, int((seconds_remaining - 1200) / seconds_per_row) // 128 * 128))


def metric(summary):
    return summary['pieces_mean'], summary['lines_mean']


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=8)


class Controller:
    def __init__(self, args):
        self.root = args.output.expanduser().resolve()
        resume_final = getattr(args, 'resume_final', False)
        if not resume_final:
            self.root.mkdir(parents=True, exist_ok=False)
        self.lock = (self.root / 'controller.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.deadline = timestamp(args.deadline)
        self.train_cutoff = timestamp(args.train_cutoff)
        if resume_final:
            saved = load(self.root/'status.json')
            if (self.deadline != timestamp(saved['deadline']) or
                    self.train_cutoff != timestamp(saved['train_cutoff'])):
                raise ValueError('recovery must preserve the saved deadlines')
            if saved['status'] not in ('failed', 'deadline_reached') or time.time() >= self.deadline:
                raise ValueError('final evaluation recovery requires a stopped run before its deadline')
        elif not time.time() < self.train_cutoff < self.deadline:
            raise ValueError('need now < train cutoff < final deadline')
        self.port = args.port
        self.env = dict(os.environ, COREAI_PYTHON=SDK_PY, TOKENIZER_PYTHON=TRAIN_PY,
                        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        self.server = None
        self.child = None
        self.used_fresh = set()
        self.corrections = {}
        self.collected = set()
        if resume_final:
            self.best = load(self.root/'best.json')
            for key in ('adapter', 'merged', 'build'):
                if not Path(self.best[key]).is_dir():
                    raise ValueError(f'saved best {key} is missing')
            if (self.root/'final-test').exists() or (self.root/'final-test.log').exists():
                raise ValueError('final evaluation already exists; preserve it for explicit review')
            self.state = saved
            self.state.setdefault('recovery_history', []).append({
                'previous_error': saved.pop('error', None),
                'previous_updated_utc': saved['updated_utc'],
                'previous_controller_pid': saved['controller_pid'],
                'mode': 'recover saved candidate, then final evaluation; no training',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            })
            self.update(status='running', stage='recovering-final-evaluation',
                        controller_pid=os.getpid(), child_pid=None, server_pid=None)
            return
        self.best = {
            'name': 'pilot-v1', 'adapter': str(STORE / 'runs/pilot-v1/adapter'),
            'merged': str(STORE / 'runs/pilot-v1/merged'),
            'build': str(STORE / 'builds/pilot-v1/coreai'),
            'dev': load(STORE / 'evals/pilot-v1-dev/report.json')['summary'],
        }
        self.state = {'status': 'running', 'controller_pid': os.getpid(),
                      'deadline': args.deadline, 'train_cutoff': args.train_cutoff,
                      'root': str(self.root), 'best': self.best, 'rounds': [],
                      'selection': 'mean development pieces, then mean lines; eight original dev seeds',
                      'final_test_note': 'Original 32 test seeds are reused as a benchmark, not a pristine new holdout.',
                      'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__), REPO/'scripts/jeff_lora_train.py', REPO/'forge.py')}}
        self.update(stage='preparing')

    def update(self, **fields):
        self.state.update(fields)
        self.state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        self.state['disk_free_GiB'] = shutil.disk_usage(self.root).free / 1024**3
        atomic_json(self.root / 'status.json', self.state)
        atomic_json(self.root / 'best.json', self.best)

    def check_deadline(self, until=None):
        if (self.root / 'STOP').exists() or time.time() >= min(until or self.deadline, self.deadline):
            raise TimeoutError('deadline or STOP file reached')

    def run(self, command, name, until=None):
        self.check_deadline(until)
        logfile = self.root / f'{name}.log'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        record = {'command': list(map(str, command)), 'started_utc': datetime.now(timezone.utc).isoformat()}
        atomic_json(logfile.with_suffix('.command.json'), record)
        started = time.time()
        with logfile.open('x') as stream:
            self.child = subprocess.Popen(record['command'], cwd=REPO, env=self.env,
                                          stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.update(stage=name, child_pid=self.child.pid, log=str(logfile))
            try:
                while self.child.poll() is None:
                    self.check_deadline(until)
                    self.update()
                    time.sleep(5)
            finally:
                stop_process(self.child)
                record.update(exit_code=self.child.returncode, seconds=time.time()-started)
                atomic_json(logfile.with_suffix('.command.json'), record)
                self.child = None
                self.update(child_pid=None)
        if record['exit_code'] != 0:
            raise RuntimeError(f'{name} failed with exit {record["exit_code"]}; see {logfile}')
        return logfile

    def start_server(self, candidate):
        self.stop_server()
        self.check_deadline()
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', self.port)) == 0:
                raise RuntimeError(f'port {self.port} is occupied; no unrelated server will be stopped')
        path = self.root / f'server-{candidate["name"]}-{int(time.time())}.log'
        command = [SDK_PY, '-u', str(REPO/'scripts/system1_ane_serve.py'), '--model', candidate['merged'],
                   '--build', candidate['build'], '--host', '127.0.0.1', '--port', str(self.port)]
        with path.open('x') as stream:
            self.server = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        self.update(server_pid=self.server.pid, server_candidate=candidate['name'], server_log=str(path))
        for _ in range(60):
            self.check_deadline()
            if self.server.poll() is not None:
                raise RuntimeError(f'server exited; see {path}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2) as response:
                    health = json.load(response)
                if health['status'] == 'ready' and health['checkpoint'] == candidate['merged']:
                    return
            except OSError:
                pass
            time.sleep(1)
        raise RuntimeError('server readiness timeout')

    def stop_server(self):
        stop_process(self.server)
        self.server = None
        if hasattr(self, 'state'):
            self.update(server_pid=None)

    def gameplay(self, dataset, split, output, name, extra=()):
        return self.run([sys.executable, '-u', '-m', 'projects.tetris_training.play',
                         '--dataset', str(dataset), '--split', split, '--adapter', 'base',
                         '--url', f'http://127.0.0.1:{self.port}', '--output', str(output), *extra], name)

    def add_corrections(self, path):
        for row in read_jsonl(path):
            key = state_key(row)
            if key not in self.held_keys and key not in self.anchor_keys:
                self.corrections[key] = row
        self.update(correction_rows=len(self.corrections))

    def data_for_round(self, number, count):
        rng = random.Random(20261008 + number)
        corrections = list(self.corrections.values())
        chosen = balanced_sample(corrections, min(count//4, len(corrections)), number)
        chosen += balanced_sample(self.anchors, count//4, number+100)
        seen = {state_key(row) for row in chosen}
        available = [row for row in self.fresh if state_key(row) not in seen | self.used_fresh]
        if len(available) < count-len(chosen):
            available = [row for row in self.fresh if state_key(row) not in seen]
        fresh = balanced_sample(available, count-len(chosen), number+200)
        self.used_fresh.update(state_key(row) for row in fresh)
        chosen += fresh
        rng.shuffle(chosen)
        keys = [state_key(row) for row in chosen]
        if len(keys) != count or len(set(keys)) != count or set(keys) & self.held_keys:
            raise RuntimeError('training mixture has duplicate or held-out states')
        directory = self.root / f'round-{number:02d}'
        directory.mkdir()
        train = directory / 'train.jsonl'
        write_jsonl(train, chosen)
        atomic_json(directory/'data.json', {
            'rows': count, 'fresh_rows': len(fresh), 'anchor_rows': count//4,
            'correction_rows': count-len(fresh)-count//4,
            'modes': dict(Counter(row['source']['mode'] for row in chosen)),
            'unique_fresh_rows_used_across_rounds': len(self.used_fresh),
            'heldout_overlap': 0, 'sha256': hashlib.sha256(train.read_bytes()).hexdigest(),
        })
        return directory, train

    def prune_derived(self, candidate):
        """Only this run's rebuildable merged/build directories; retain all adapters/reports."""
        for key in ('merged', 'build'):
            path = Path(candidate[key])
            if key == 'build':
                path = path.parent
            if path.is_relative_to(self.root) and path.name in ('merged', 'build') and not path.is_symlink():
                if path.exists():
                    shutil.rmtree(path)

    def saved_training(self, candidate):
        """A report marker is published only after complete adapter and merged renames."""
        output = Path(candidate['adapter']).parent
        report_path = output/'report.json'
        if not report_path.is_file():
            return None
        report = load(report_path)
        if report.get('save_state') not in ('weights_saved', 'complete'):
            return None
        required = ((candidate['adapter'], 'adapter_model.safetensors'),
                    (candidate['adapter'], 'readout.safetensors'),
                    (candidate['adapter'], 'adapter_config.json'),
                    (candidate['merged'], 'model.safetensors'),
                    (candidate['merged'], 'readout.safetensors'),
                    (output, 'parity_rows.json'))
        if not all((Path(folder)/name).is_file() for folder, name in required):
            return None
        return report

    def recover_saved_merge(self, candidate):
        """An atomic adapter/report can be merged if the trainer stopped during merge."""
        output = Path(candidate['adapter']).parent
        report_path = output/'report.json'
        if not report_path.is_file():
            return
        report = load(report_path)
        if report.get('save_state') != 'adapter_saved':
            return
        if not all((output/name).is_file() for name in ('parity_rows.json',)) or not all(
                (Path(candidate['adapter'])/name).is_file() for name in
                ('adapter_model.safetensors', 'readout.safetensors', 'adapter_config.json')):
            return
        merged = Path(candidate['merged'])
        if not merged.exists():
            from coreai.jeff_lora import merge_saved_lora
            config = load(Path(candidate['adapter'])/'adapter_config.json')
            merge_saved_lora(Path(config['base_checkpoint']), Path(candidate['adapter']), merged)
        if not all((merged/name).is_file() for name in ('model.safetensors', 'readout.safetensors')):
            return
        report['save_state'] = 'weights_saved'
        atomic_json(report_path, report)

    def finish_round(self, pending, *, recovery=False):
        number, count, lr = pending['number'], pending['count'], pending['lr']
        candidate = pending['candidate']
        directory = self.root/f'round-{number:02d}'
        output = directory/'model'
        training = self.saved_training(candidate)
        if training is None:
            raise RuntimeError(f'round {number:02d} has no complete saved weights and report')
        if 'heldout' in training['adapter'] and not all(
                math.isfinite(training['adapter']['heldout'][k]) for k in ('loss', 'accuracy')):
            raise RuntimeError('non-finite validation metrics')
        if sum(h['n'] for h in training['history']) == 0:
            self.update(pending_candidate=None, stop_reason='round completed no optimizer steps')
            return False
        build_output = directory/'build'
        dev_output = directory/'dev'
        suffix = ''
        if recovery:
            attempt = 1
            while (directory/f'build-recovery-{attempt:02d}').exists() or (
                    directory/f'dev-recovery-{attempt:02d}').exists() or (
                    self.root/f'export-{number:02d}-recovery-{attempt:02d}.log').exists():
                attempt += 1
            suffix = f'-recovery-{attempt:02d}'
            build_output = directory/f'build-recovery-{attempt:02d}'
            dev_output = directory/f'dev-recovery-{attempt:02d}'
            candidate['build'] = str(build_output/'coreai')
            self.update(pending_candidate=pending)
        self.run([TRAIN_PY, str(REPO/'forge.py'), 'jeff-convert', '--model', candidate['merged'],
                  '--output', str(build_output), '--ctx', '2048', '--prefill', '256', '--quant', 'fp16'],
                 f'export-{number:02d}{suffix}')
        self.run([TRAIN_PY, str(REPO/'forge.py'), 'compile', '--build', candidate['build']],
                 f'compile-{number:02d}{suffix}')
        self.run([sys.executable, str(REPO/'coreai/inspect_coreai_cache.py'), '--model-dir', candidate['build'],
                  '--executable', 'python', '--strict'], f'placement-{number:02d}{suffix}')
        parity = self.run([SDK_PY, str(REPO/'scripts/jeff_lora_parity.py'), '--model', candidate['merged'],
                           '--build', candidate['build'], '--rows', str(output/'parity_rows.json')],
                          f'parity-{number:02d}{suffix}')
        result = json.loads(re.search(r'\{\s*"rows".*?\}', parity.read_text(), re.S).group())
        if result['argmax_matches'] != result['rows'] or result['rows'] != 4:
            raise RuntimeError(f'parity check needs review: {result}')
        self.start_server(candidate)
        self.gameplay(PILOT_DATA, 'dev', dev_output, f'dev-{number:02d}{suffix}')
        candidate['dev'] = load(dev_output/'report.json')['summary']
        accepted = metric(candidate['dev']) > metric(self.best['dev'])
        self.state['rounds'].append({**candidate, 'accepted': accepted, 'training_rows': count,
                                      'lr': lr, 'parity': result, 'training_report': str(output/'report.json'),
                                      'training_validation': 'complete' if 'heldout' in training['adapter'] else 'interrupted_after_save'})
        self.stop_server()
        if accepted:
            previous = self.best
            self.best = candidate
            self.update(best=self.best)
            self.prune_derived(previous)
        else:
            self.prune_derived(candidate)
        self.update(stage=f'round-{number:02d}-complete', pending_candidate=None)
        return accepted

    def recover_pending_round(self):
        pending = self.state.get('pending_candidate')
        if pending and pending['candidate']['name'] not in {row['name'] for row in self.state['rounds']}:
            self.recover_saved_merge(pending['candidate'])
            if self.saved_training(pending['candidate']) is not None:
                self.finish_round(pending, recovery=True)
            else:
                self.update(pending_candidate=None, recovery_note='pending round had no complete saved weights')

    def execute(self):
        audit(PILOT_DATA)
        audit(POOL)
        self.anchors = read_jsonl(PILOT_DATA/'train.jsonl')
        self.anchor_keys = {state_key(row) for row in self.anchors}
        self.fresh = read_jsonl(POOL/'train.jsonl')
        self.held_keys = {state_key(row) for folder in (PILOT_DATA, POOL)
                          for split in ('validation', 'test') for row in read_jsonl(folder/f'{split}.jsonl')}
        self.add_corrections(INITIAL/'corrections.jsonl')
        self.update(fresh_pool_rows=len(self.fresh), anchor_rows=len(self.anchors))
        lr = 1e-4
        for number in range(1, 8):
            if round_size(self.train_cutoff-time.time()) < 512:
                break
            self.check_deadline()
            if number > 1 and self.best['name'] not in self.collected and self.train_cutoff-time.time() > 3600:
                self.start_server(self.best)
                collection = self.root / f'corrections-{number:02d}'
                self.gameplay(POOL, 'train', collection, f'collect-{number:02d}', ('--collect', '--games', '8'))
                self.add_corrections(collection/'corrections.jsonl')
                self.collected.add(self.best['name'])
                self.stop_server()
            count = round_size(self.train_cutoff-time.time())
            if count < 512:
                break
            if shutil.disk_usage(self.root).free < 5 * 1024**3:
                self.update(stop_reason='disk reserve reached; keeping saved adapters and best build')
                break
            directory, train = self.data_for_round(number, count)
            output = directory/'model'
            candidate = {'name': f'round-{number:02d}', 'adapter': str(output/'adapter'),
                         'merged': str(output/'merged'), 'build': str(directory/'build/coreai')}
            command = [TRAIN_PY, '-u', str(REPO/'forge.py'), 'jeff-train-lora', '--model', str(BASE),
                       '--task', 'tetris', '--dataset', str(train), '--heldout-dataset', str(PILOT_DATA/'validation.jsonl'),
                       '--init-adapter', self.best['adapter'], '--output', str(output), '--device', 'mps',
                       '--rank', '16', '--alpha', '32', '--epochs', '1', '--batch', '4', '--lr', str(lr),
                       '--readout-lr', '5e-6', '--eval-batch', '16', '--eval-train-rows', '64',
                       '--checkpoint-every', '64', '--seed', str(100+number),
                       '--train-until', datetime.fromtimestamp(self.train_cutoff, timezone.utc).isoformat()]
            pending = {'number': number, 'count': count, 'lr': lr, 'candidate': candidate}
            self.update(pending_candidate=pending)
            try:
                self.run(command, f'train-{number:02d}', until=min(self.train_cutoff+1200, self.deadline))
            except (TimeoutError, RuntimeError) as error:
                if (self.root/'STOP').exists() or time.time() >= self.deadline:
                    raise
                self.recover_saved_merge(candidate)
                if self.saved_training(candidate) is None:
                    raise
                self.update(training_note=f'trainer stopped after weights were saved: {error}')
            accepted = self.finish_round(pending)
            if self.state.get('stop_reason') == 'round completed no optimizer steps':
                break
            if not accepted:
                lr = max(2.5e-5, lr/2)
        self.finalize()

    def finalize(self):
        """Evaluate the saved best model without performing any optimization."""
        self.start_server(self.best)
        if self.best['name'] == 'pilot-v1':
            self.state['final_test'] = str(STORE/'evals/pilot-v1-test/report.json')
        else:
            self.gameplay(PILOT_DATA, 'test', self.root/'final-test', 'final-test')
            self.state['final_test'] = str(self.root/'final-test/report.json')
        self.update(status='complete', stage='finished', best_demo_url=f'http://127.0.0.1:{self.port}/',
                    unique_fresh_rows_used=max([len(self.used_fresh), *[
                        load(p).get('unique_fresh_rows_used_across_rounds', 0)
                        for p in self.root.glob('round-*/data.json')]]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--train-cutoff', required=True)
    parser.add_argument('--port', type=int, default=8789)
    parser.add_argument('--resume-final', action='store_true',
                        help='recover a saved pending candidate and final evaluation; never starts training')
    args = parser.parse_args(argv)
    controller = Controller(args)
    def interrupted(_signum, _frame):
        raise TimeoutError('controller interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.resume_final:
            controller.recover_pending_round()
            controller.finalize()
        else:
            controller.execute()
    except BaseException as error:
        stop_process(controller.child)
        controller.stop_server()
        controller.update(status='deadline_reached' if isinstance(error, TimeoutError) else 'failed',
                          error=f'{type(error).__name__}: {error}', stage='stopped')
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
