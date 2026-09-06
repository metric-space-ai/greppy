"""Bind a dispatched agent's public turn identity and start the live oracle watcher."""
import argparse
import json
from pathlib import Path
import subprocess

from verify_on_completion import watch


def start(series, position, agent_path, session_dir):
    series = Path(series)
    found = subprocess.run(['greppy', 'rg', '-l', '--fixed-strings',
        '"agent_path":' + json.dumps(agent_path), str(session_dir)],
        capture_output=True, text=True, check=True)
    paths = found.stdout.splitlines()
    if len(paths) != 1:
        raise ValueError(f'Expected one participant rollout, found {len(paths)}')
    source = Path(paths[0])
    contexts = []
    for line in source.open():
        record = json.loads(line)
        if record.get('type') == 'turn_context':
            contexts.append(record['payload'])
    if len(contexts) != 1:
        raise ValueError('Require exactly one participant turn')
    binding = dict(agent_path=agent_path, source=str(source),
                   turn_id=contexts[0]['turn_id'], position=position,
                   scope='Returned spawn path and public turn context; not opaque assignment readback')
    folder = series / 'live'
    folder.mkdir(exist_ok=True)
    with (folder / f'{position:02d}-binding.json').open('x') as output:
        json.dump(binding, output, indent=2)
    result = watch(series, position, source, binding['turn_id'], folder / f'{position:02d}')
    print(json.dumps({k: result.get(k) for k in (
        'timing_valid', 'oracle', 'error', 'end_to_end_verified_seconds')}))
    return 0 if result['timing_valid'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('series', type=Path)
    p.add_argument('position', type=int)
    p.add_argument('--agent-path', required=True)
    p.add_argument('--session-dir', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(start(a.series, a.position, a.agent_path, a.session_dir))
