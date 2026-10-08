"""Create and verify a private rebuild snapshot; never publish its data to GitHub."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def backup(destination):
    manifest = json.loads((ROOT/'artifacts/cycle5_recovered/manifest.json').read_text())
    paths = set(manifest['files'])
    paths.update(['artifacts/cycle2/freeze_receipt.json',
                  'artifacts/cycle3/manifest.json', 'artifacts/cycle3/start_receipt.json',
                  'scripts/backup_cycle5.py'])
    for name in subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines():
        if name.endswith(('.py', '.md', '.toml', '.json', '.yml', '.yaml')) or name == 'LICENSE':
            paths.add(name)
    # Completed fold files are atomically renamed and never changed in place.
    for path in (ROOT/'artifacts/cycle5_recovered').rglob('*'):
        if path.is_file() and path.suffix in {'.joblib', '.json'} and path.name != 'progress.json':
            paths.add(str(path.relative_to(ROOT)))
    hashes = {}
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.pending')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in sorted(paths):
            content = (ROOT/name).read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        receipt = {'schema_version': 1, 'kind': 'newly_rebuilt_cycle5_snapshot',
                   'fold_checkpoints': sum(name.endswith('.joblib') for name in paths),
                   'private_data_included': True, 'sha256': hashes}
        archive.writestr('BACKUP_MANIFEST.json', json.dumps(receipt, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        assert archive.testzip() is None
    temporary.replace(destination)
    result = {'path': str(destination), 'bytes': destination.stat().st_size,
              'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'fold_checkpoints': receipt['fold_checkpoints'], 'all_members_verified': True}
    destination.with_suffix('.receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    backup(parser.parse_args().destination)
