#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('engine')
    p.add_argument('library')
    args = p.parse_args()
    process = subprocess.Popen([args.engine, '--offline', '3000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = process.stdout.readline().strip()
        if ready != 'offline-ready':
            raise RuntimeError(f'engine did not become ready: {ready} {process.stderr.read()}')
        subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'vision_mock_sender.py'),
                        '--library', args.library, '--seconds', '1'], check=True, timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
        stdout, stderr = process.communicate(timeout=5)
    if process.returncode != 0:
        raise RuntimeError(f'engine failed: {process.returncode} {stdout} {stderr}')
    print(stdout.strip())

if __name__ == '__main__':
    main()
