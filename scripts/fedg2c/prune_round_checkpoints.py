"""Prune round snapshots, keeping periodic/latest rounds and all JSON records.

Dry-run by default. Only known client/server weight and server probe files are
eligible; source data, private final states and target models are never removed.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat

ROUND = re.compile(r'round_(\d+)$')
CLIENT = re.compile(r'client_\d+$')
SKIP = {'code_snapshot', 'cache_data', 'smoke_inputs', '.git', '__pycache__',
        'stage2', 'target_none', 'target_lora', 'target_adapter', 'target_prompt'}


def round_files(folder):
    result = []
    with os.scandir(folder) as children:
        for child in children:
            if not child.is_dir(follow_symlinks=False):
                continue
            names = (('shared_backbone.pt', 'server_probes.pt') if child.name == 'server'
                     else ('shared_backbone.pt',) if CLIENT.fullmatch(child.name) else ())
            if not names:
                continue
            # Enumerate actual entries rather than stat potentially missing
            # probe files: negative lookups can stall on shared filesystems.
            with os.scandir(child.path) as entries:
                for entry in entries:
                    if entry.name not in names:
                        continue
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode):
                        result.append((Path(entry.path), info.st_size))
    return result


def inventory(root):
    groups, json_paths = defaultdict(list), []
    rounds = []
    pending = [Path(root)]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as children:
            entries = sorted(children, key=lambda entry: entry.name)
        names = {entry.name for entry in entries}
        if {'source_manifest.json', 'target_manifest.json'} & names:
            continue
        json_paths.extend(folder / name for name in
                          ('resolved_config.json', 'input_config.json', 'execution_plan.json')
                          if name in names)
        descend = []
        for entry in entries:
            if entry.name in SKIP or not entry.is_dir(follow_symlinks=False):
                continue
            path = Path(entry.path)
            match = ROUND.fullmatch(entry.name)
            if match:
                rounds.append((folder, int(match[1]), path))
            else:
                descend.append(path)
        pending.extend(reversed(descend))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (parent, round_id, path), files in zip(
                rounds, pool.map(round_files, (item[2] for item in rounds))):
            if files:
                groups[parent].append((round_id, path, files))
    return groups, json_paths


def referenced_checkpoints(json_paths):
    references = set()

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.endswith('shared_backbone.pt'):
            path = Path(value)
            if path.is_absolute():
                references.add(path.resolve())

    for path in json_paths:
        visit(json.loads(path.read_text(encoding='utf-8')))
    return references


def make_plan(root, interval):
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or interval < 1:
        raise ValueError('Require an existing runs directory and a positive interval')
    groups, json_paths = inventory(root)
    references = referenced_checkpoints(json_paths)
    report = {'root': str(root), 'interval': interval, 'keep_latest': True,
              'round_numbering': 'one-based completed rounds', 'groups': [],
              'delete_files': [], 'delete_bytes': 0, 'checkpoint_bytes_before': 0}
    for parent, rounds in sorted(groups.items()):
        latest = max(round_id for round_id, _, _ in rounds)
        server_rounds = [round_id for round_id, _, files in rounds
                         if any(path.parent.name == 'server' and path.name == 'shared_backbone.pt'
                                for path, _ in files)]
        latest_server = max(server_rounds) if server_rounds else latest
        retained, removed = [], []
        for round_id, folder, files in sorted(rounds):
            sizes = dict(files)
            report['checkpoint_bytes_before'] += sum(sizes.values())
            # inventory/round_files already reject symlinks. Avoid resolving the
            # same long parent chain for every file on shared filesystems.
            pinned = any(path in references for path in sizes)
            keep = (round_id + 1) % interval == 0 or round_id in (latest, latest_server) or pinned
            (retained if keep else removed).append(round_id)
            if not keep:
                for path, size in sizes.items():
                    if not path.is_relative_to(root):
                        raise ValueError(f'Checkpoint escapes runs directory: {path}')
                    report['delete_files'].append({'path': str(path), 'bytes': size})
                    report['delete_bytes'] += size
        report['groups'].append({'path': str(parent), 'kept_round_ids': retained,
                                 'pruned_round_ids': removed})
    report['delete_count'] = len(report['delete_files'])
    report['checkpoint_bytes_after'] = report['checkpoint_bytes_before'] - report['delete_bytes']
    return report


def apply_plan(report):
    root = Path(report['root']).resolve(strict=True)
    # Validate every candidate before unlinking any of them.
    paths = []
    parents = set()
    for item in report['delete_files']:
        path = Path(item['path'])
        allowed = (path.name == 'shared_backbone.pt' and
                   (path.parent.name == 'server' or CLIENT.fullmatch(path.parent.name)))
        allowed = allowed or (path.name == 'server_probes.pt' and path.parent.name == 'server')
        if ('..' in path.parts or not path.is_relative_to(root) or not allowed
                or not ROUND.fullmatch(path.parent.parent.name)):
            raise ValueError(f'Unsafe checkpoint path: {path}')
        for parent in path.parents:
            if parent == root:
                break
            parents.add(parent)
        paths.append(path)

    def validate_parent(parent):
        if parent.is_symlink():
            raise ValueError(f'Symlink in checkpoint path: {parent}')

    def validate_file(item):
        info = Path(item['path']).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != item['bytes']:
            raise ValueError(f'Checkpoint changed after inventory: {item["path"]}')

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(validate_parent, parents))
        list(pool.map(validate_file, report['delete_files']))
        list(pool.map(Path.unlink, paths))
    removed_bytes = sum(item['bytes'] for item in report['delete_files'])
    report.update(applied=True, deleted_count=len(paths), deleted_bytes=removed_bytes,
                  applied_at=datetime.now(timezone.utc).isoformat())


def summary(report):
    return {key: value for key, value in report.items() if key != 'delete_files'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs_root', type=Path)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--audit', type=Path,
                        help='Required with --apply; writes the deletion manifest before removing files')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be positive')
    if args.apply and args.audit is None:
        parser.error('--apply requires --audit')
    if args.audit is not None and args.audit.exists():
        parser.error('Use a new audit path')
    report = make_plan(args.runs_root, args.interval)
    report['applied'] = False
    if args.audit is not None:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.apply:
        apply_plan(report)
        args.audit.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary(report), indent=2))


if __name__ == '__main__':
    main()
