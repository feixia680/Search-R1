"""Compare raw/cold/memory/disk BM25 on repeated historical TRAIN queries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import time
import requests
from search_r1.search.retrieval_client import RetrievalClient, index_fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8000/retrieve')
    parser.add_argument('--unique-queries',type=int,default=32)
    args=parser.parse_args()
    unique=[]
    with Path(args.trace).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('split') != 'train':continue
            for event in row.get('events',[]):
                query=event.get('query')
                if event.get('attempted') and isinstance(query,str) and query not in unique:
                    unique.append(query)
                    if len(unique)>=args.unique_queries:break
            if len(unique)>=args.unique_queries:break
    if not unique:raise ValueError('No attempted TRAIN search queries found')
    queries=[q for q in unique for _ in range(2)]
    chunks=[queries[i:i+16] for i in range(0,len(queries),16)]
    session=requests.Session();session.trust_env=False
    def raw():
        rows=[];started=time.perf_counter()
        for chunk in chunks:
            response=session.post(args.url,json=dict(queries=chunk,topk=3,return_scores=True),timeout=30)
            response.raise_for_status();rows.extend(response.json()['result'])
        return rows,dict(logical_queries=len(queries),backend_query_attempts=len(queries),
                        backend_http_requests=len(chunks),wall_seconds=time.perf_counter()-started)
    # Warm BM25 itself before comparing client layers.
    expected,_=raw()
    baseline,baseline_stats=raw()
    if baseline!=expected:raise ValueError('Raw BM25 is nondeterministic')
    root=Path(__file__).resolve().parents[1]
    version=index_fingerprint([root/'data/index/bm25',root/'data/corpus/hotpot_contexts.jsonl'])
    stages={'raw':baseline_stats}
    with tempfile.TemporaryDirectory(prefix='bm25_runtime_bench_') as temporary:
        cache=Path(temporary)/'cache.sqlite3'
        client=RetrievalClient(args.url,cache_path=cache,index_version=version)
        for name in ('cold','memory_warm','disk_warm'):
            if name=='disk_warm':client=RetrievalClient(args.url,cache_path=cache,index_version=version)
            rows=[];stats=Counter();started=time.perf_counter()
            for chunk in chunks:
                rows.extend(client.retrieve(chunk,topk=3));stats.update(client.last_stats)
            stats['wall_seconds']=time.perf_counter()-started
            if rows!=expected:raise ValueError('Optimized results differ from raw BM25: '+name)
            stages[name]=dict(stats)
    result=dict(kind='offline_exact_query_replay',trace=args.trace,unique_queries=len(unique),
                logical_queries=len(queries),workload_sha256=hashlib.sha256(json.dumps(queries).encode()).hexdigest(),
                identical_results=True,stages=stages,
                limitation='Repeated-query benchmark; cache hit rate is workload dependent, not full training speedup.')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
