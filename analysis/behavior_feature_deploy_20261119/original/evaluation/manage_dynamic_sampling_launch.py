"""Own a local BM25 child only when needed, check real retrieval, then launch.

--check-only validates data, model, config and retrieval with no GPU or training.
"""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

import requests


def healthy(session, url):
    try:
        r = session.post(url, json={'queries': ['Paris France'], 'topk': 3,
                                   'return_scores': True}, timeout=3)
        r.raise_for_status()
        rows = r.json()['result']
        return (isinstance(rows, list) and len(rows) == 1 and bool(rows[0])
                and all(isinstance(x.get('document', {}).get('contents'), str) for x in rows[0]))
    except (requests.RequestException, ValueError, KeyError, AttributeError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=[1, 2], required=True)
    parser.add_argument('--check-only', action='store_true')
    args, overrides = parser.parse_known_args()
    if args.check_only and overrides:
        parser.error('--check-only accepts no training overrides')
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.setdefault('RETRIEVER_URL', 'http://127.0.0.1:8000/retrieve')
    for override in overrides:
        if override.startswith('retriever.url='):
            env['RETRIEVER_URL'] = override.split('=', 1)[1]
    url = env['RETRIEVER_URL']
    parsed = urlparse(url)
    session = requests.Session()
    session.trust_env = False
    child = None
    log = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        if not healthy(session, url):
            if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.path != '/retrieve':
                raise RuntimeError('Retriever is unhealthy; auto-start supports local HTTP /retrieve only: '+url)
            port = parsed.port or 80
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Port {port} is occupied but /retrieve failed; refusing to start a conflicting server')
            env['BM25_PORT'] = str(port)
            logs = root / 'logs'
            logs.mkdir(exist_ok=True)
            log_path = logs / ('bm25_launch_%s_%s.log' % (time.strftime('%Y%m%d_%H%M%S'), os.getpid()))
            log = log_path.open('w')
            child = subprocess.Popen(['bash', str(root/'start_bm25_local.sh')],
                                     cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            print('Starting BM25, log: '+str(log_path), flush=True)
            deadline = time.monotonic() + float(env.get('BM25_START_TIMEOUT', '180'))
            while not healthy(session, url):
                if child.poll() is not None or time.monotonic() >= deadline:
                    log.flush()
                    print(log_path.read_text()[-6000:], file=sys.stderr)
                    raise RuntimeError('BM25 failed readiness check; training was not started')
                time.sleep(1)
        print('BM25 POST /retrieve: PASS ('+url+')', flush=True)
        trainer = root / f'train_grpo_dynamic_sampling_{args.gpus}gpu.sh'
        if args.check_only:
            subprocess.run(['bash', str(trainer), '--print-config'], cwd=root, env=env,
                           stdout=subprocess.DEVNULL, check=True)
            base = Path(env.get('BASE_ROOT', str(root)))
            evidence = env.get('EVIDENCE_DATA_DIR', str(root/'algorithm_a_data/evidence'))
            model = env.get('BASE_MODEL', str(base/'models/Qwen2.5-3B'))
            subprocess.run([sys.executable, str(root/'evaluation/check_evidence_training_inputs.py'),
                            '--base-root', str(base), '--evidence-dir', evidence,
                            '--retriever-url', url, '--model', model,
                            '--gpu-ids', env['CUDA_VISIBLE_DEVICES'], '--gpus', str(args.gpus),
                            '--alpha', env.get('EVIDENCE_ALPHA', '0.1'), '--skip-gpu-checks'],
                           cwd=root, env=env, check=True)
            print('CHECK_ONLY PASS: paths, model, dataset hashes, annotations, config and BM25; no training/GPU use', flush=True)
        else:
            subprocess.run(['bash', str(trainer), *overrides], cwd=root, env=env, check=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if log is not None:
            log.close()
        session.close()


if __name__ == '__main__':
    main()
