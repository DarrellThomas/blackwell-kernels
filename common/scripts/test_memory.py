#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / 'common/tests'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the common/memory pytest suite')
    parser.add_argument('-k', dest='keyword', default='', help='Optional pytest -k expression')
    parser.add_argument('--real-embedding', action='store_true', help='Include the real embedding smoke test')
    parser.add_argument('--verbose', action='store_true', help='Use verbose pytest output')
    args = parser.parse_args(argv)

    env = os.environ.copy()
    current = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(ROOT) if not current else f"{ROOT}{os.pathsep}{current}"
    if args.real_embedding:
        env['FB_RUN_REAL_EMBEDDING'] = '1'

    cmd = [sys.executable, '-m', 'pytest']
    if args.verbose:
        cmd.append('-vv')
    else:
        cmd.append('-q')
    if not args.real_embedding:
        cmd.extend(['-m', 'not integration'])
    if args.keyword:
        cmd.extend(['-k', args.keyword])
    cmd.append(str(TEST_ROOT))

    print('$ ' + ' '.join(cmd))
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
