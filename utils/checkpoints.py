import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def checkpoint_metadata(root, model, dataset):
    folder = Path(root) / model / dataset
    metadata = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    path = folder / metadata['file']
    if sha256(path) != metadata['sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {path}')
    return (path, metadata)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
