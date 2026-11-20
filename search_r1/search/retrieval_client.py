"""Offline BM25 execution: exact-query cache, batch coalescing and bounded retries.

Logical model searches are never removed. Only backend work is coalesced.
Failed or malformed results are never persisted as successful observations.
"""
from collections import Counter, OrderedDict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import requests


def index_fingerprint(paths):
    entries = []
    for name in paths:
        path = Path(name).expanduser().resolve()
        files = sorted(p for p in path.rglob('*') if p.is_file() and not p.name.endswith('.lock')) if path.is_dir() else [path]
        for p in files:
            stat = p.stat()
            entries.append((str(p), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def valid_documents(rows):
    return isinstance(rows, list) and all(
        isinstance(entry, dict) and isinstance(entry.get('document'), dict)
        and isinstance(entry['document'].get('contents'), str)
        and ('score' not in entry or (isinstance(entry['score'], (int, float)) and math.isfinite(entry['score'])))
        for entry in rows)


class RetrievalClient:
    def __init__(self, url, timeout=30., cache_path=None, index_version='default',
                 max_retries=2, retry_backoff=.1, memory_items=1024, session=None):
        if timeout <= 0 or max_retries < 0 or retry_backoff < 0 or memory_items < 0:
            raise ValueError('Invalid retrieval timeout/retry/cache limits')
        self.url, self.timeout = url, float(timeout)
        self.max_retries, self.backoff = int(max_retries), float(retry_backoff)
        self.namespace, self.memory_items = str(index_version), int(memory_items)
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.memory = OrderedDict()
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.last_stats, self.last_outcomes = {}, []
        self._cache_init_error = False
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as db:
                    db.execute('CREATE TABLE IF NOT EXISTS retrieval_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            except (OSError, sqlite3.Error):
                # Persistence failures must not become model/tool failures.
                self.cache_path = None
                self._cache_init_error = True

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.cache_path), timeout=5)
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _key(self, query, topk):
        return hashlib.sha256(json.dumps([self.url, self.namespace, int(topk), query], ensure_ascii=False).encode()).hexdigest()

    def _remember(self, key, rows):
        if self.memory_items:
            self.memory[key] = rows; self.memory.move_to_end(key)
            while len(self.memory) > self.memory_items: self.memory.popitem(last=False)

    def _get(self, key, stats):
        if key in self.memory:
            self.memory.move_to_end(key); return self.memory[key]
        if not self.cache_path: return None
        try:
            with self._connect() as db:
                row = db.execute('SELECT payload FROM retrieval_cache WHERE cache_key=?', (key,)).fetchone()
            if row:
                data = json.loads(row[0])
                if not valid_documents(data): raise ValueError('Malformed cached documents')
                self._remember(key, data); return data
        except (sqlite3.Error, OSError, ValueError, TypeError):
            stats['cache_errors'] += 1
        return None

    def _put(self, key, rows, stats):
        self._remember(key, rows)
        if not self.cache_path: return
        try:
            payload = json.dumps(rows, ensure_ascii=False, allow_nan=False)
            with self._connect() as db:
                db.execute('INSERT OR REPLACE INTO retrieval_cache VALUES (?,?)', (key, payload))
            stats['cache_writes'] += 1
        except (sqlite3.Error, OSError, ValueError, TypeError):
            stats['cache_errors'] += 1

    def retrieve(self, queries, topk=3):
        if topk < 1: raise ValueError('topk must be positive')
        started = time.perf_counter()
        stats = Counter(logical_queries=len(queries), unique_queries=0, batch_duplicates=0,
                        cache_hits=0, cache_writes=0, cache_errors=int(self._cache_init_error), backend_http_requests=0,
                        backend_query_attempts=0, retries=0, failures=0)
        self._cache_init_error = False
        groups = OrderedDict()
        for i, query in enumerate(queries):
            if not isinstance(query, str): raise TypeError('Retrieval queries must be strings')
            groups.setdefault(self._key(query, topk), []).append(i)
        stats['unique_queries'] = len(groups)
        stats['batch_duplicates'] = len(queries)-len(groups)
        results = [None] * len(queries)
        outcomes = [dict(cache_hit=False, batch_coalesced=False, backend_query_attempts=0,
                         error_category=None, error=None) for _ in queries]
        pending = []
        for key, positions in groups.items():
            cached = self._get(key, stats)
            for offset, i in enumerate(positions): outcomes[i]['batch_coalesced'] = offset > 0
            if cached is not None:
                for i in positions: results[i] = cached; outcomes[i]['cache_hit'] = True
                stats['cache_hits'] += len(positions)
            else: pending.append((key, positions))
        if pending:
            error, category, payload = None, None, None
            for attempt in range(self.max_retries+1):
                stats['backend_http_requests'] += 1
                stats['backend_query_attempts'] += len(pending)
                for _, positions in pending: outcomes[positions[0]]['backend_query_attempts'] += 1
                retry = False
                try:
                    response = self.session.post(self.url, json=dict(queries=[queries[p[0]] for _,p in pending],
                        topk=int(topk), return_scores=True), timeout=self.timeout)
                    response.raise_for_status()
                    payload = response.json()['result']
                    if not isinstance(payload, list) or len(payload) != len(pending) or not all(valid_documents(rows) for rows in payload):
                        raise ValueError('Malformed retrieval response')
                    error, category = None, None
                    break
                except requests.Timeout as exc:
                    error, category, retry = str(exc), 'timeout', True
                except requests.ConnectionError as exc:
                    error, category, retry = str(exc), 'connection', True
                except requests.HTTPError as exc:
                    status = getattr(getattr(exc, 'response', None), 'status_code', 0)
                    error, category = str(exc), 'http_'+str(status)
                    retry = status in (408, 429) or status >= 500
                except (ValueError, KeyError, TypeError) as exc:
                    error, category = str(exc), 'malformed_response'
                except requests.RequestException as exc:
                    error, category = str(exc), 'request_error'
                if not retry or attempt == self.max_retries: break
                stats['retries'] += 1
                if self.backoff: time.sleep(self.backoff * (2**attempt))
            if category is None:
                for (key, positions), rows in zip(pending, payload):
                    self._put(key, rows, stats)
                    for i in positions: results[i] = rows
            else:
                stats['failures_'+category] += sum(len(positions) for _,positions in pending)
                for _, positions in pending:
                    for i in positions:
                        outcomes[i]['error_category'], outcomes[i]['error'] = category, error
                        stats['failures'] += 1
        stats['wall_seconds'] = time.perf_counter()-started
        self.last_stats, self.last_outcomes = dict(stats), outcomes
        return results


def runtime_options(config):
    """Resolve a local index namespace once per generation manager."""
    import os
    options = dict(config.get('engineering', {}).get('retrieval', {}))
    if not options.pop('enabled', False):
        return None
    root = Path(__file__).resolve().parents[2]
    paths = [os.environ.get('BM25_INDEX_PATH', str(root/'data/index/bm25')),
             os.environ.get('BM25_CORPUS_PATH', str(root/'data/corpus/hotpot_contexts.jsonl'))]
    if options.get('index_version', 'auto') == 'auto':
        options['index_version'] = index_fingerprint(paths)
    cache = Path(options.get('cache_path') or root/'data/cache/retrieval_runtime.sqlite3')
    options['cache_path'] = str(cache if cache.is_absolute() else root/cache)
    return options
