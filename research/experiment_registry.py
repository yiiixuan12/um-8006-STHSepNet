"""Append-only invocation provenance; no training dependencies."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

FIELDS = ('run_id', 'timestamp', 'git_commit', 'config', 'data_version', 'seed',
          'hardware', 'runtime', 'cost', 'status', 'metrics', 'artifact_path', 'notes')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *arguments):
    try:
        return subprocess.check_output(
            ['git', '-c', f'safe.directory={Path(root).as_posix()}', *arguments],
            cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def append_event(path, row):
    """Serialize concurrent writers with an OS advisory lock on a sidecar file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        if platform.system() == 'Windows':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with path.open('a+', newline='', encoding='utf-8') as stream:
                stream.seek(0)
                header = next(csv.reader(stream), None)
                if header is not None and header != list(FIELDS):
                    raise ValueError('Unexpected experiment registry columns')
                stream.seek(0, 2)
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                if header is None:
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
        finally:
            lock.seek(0)
            if platform.system() == 'Windows':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def registered_run(args, root, spec, sources, hardware, callback):
    """Append one final record per CLI invocation; keep configuration evidence."""
    root = Path(root)
    run_id = uuid.uuid4().hex
    started = time.perf_counter()
    artifacts = root / 'experiments' / 'runs' / run_id
    artifacts.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config['source_sha256'] = {name: sha256(root / name) for name in sources}
    config['git_status'] = git(root, 'status', '--porcelain')
    (artifacts / 'invocation.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    row = dict.fromkeys(FIELDS, '')
    row.update(run_id=run_id, git_commit=git(root, 'rev-parse', 'HEAD'),
               config=str(artifacts / 'invocation.json'), seed=args.seed,
               hardware=json.dumps(hardware), cost='not separately metered',
               artifact_path=str(args.output.resolve()),
               notes=json.dumps({'dataset': args.dataset, 'stage': args.stage,
                                 'smoke': args.smoke, 'resume': args.resume}))

    def event(status, notes=None):
        row.update(timestamp=datetime.now(timezone.utc).isoformat(),
                   runtime=round(time.perf_counter() - started, 3), status=status)
        if notes is not None:
            row['notes'] = json.dumps(notes)
        append_event(root / 'experiments' / 'registry.csv', row)

    try:
        row['data_version'] = json.dumps({name: sha256(root / 'dataset' / spec['folder'] / spec[name])
                                          for name in ('csv', 'adj')})
        callback(args)
        report_path = args.output / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        # Keep per-invocation evidence even when the shared report later changes.
        (artifacts / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        stages = ('baseline', 'improved') if args.stage == 'pair' else (args.stage,)
        row['metrics'] = json.dumps({s: report['stages'][s]['test'] for s in stages})
        event('completed')
    except BaseException as exc:
        event('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
              {'dataset': args.dataset, 'stage': args.stage, 'error': repr(exc)})
        raise
