import requests
import pytest
from search_r1.search.retrieval_client import RetrievalClient


def docs(text):return [{'document':{'id':text,'contents':text+'\nbody'},'score':1.0}]
class Response:
    def __init__(self,payload,status=200):self.payload,self.status_code=payload,status
    def raise_for_status(self):
        if self.status_code>=400:raise requests.HTTPError(str(self.status_code),response=self)
    def json(self):return self.payload
class Session:
    def __init__(self,items):self.items=list(items);self.calls=[]
    def post(self,url,**kwargs):
        self.calls.append(kwargs)
        result=self.items.pop(0)
        if isinstance(result,Exception):raise result
        return result

def client(tmp_path,items,**kwargs):
    session=Session(items)
    return RetrievalClient('http://127.0.0.1/retrieve',cache_path=tmp_path/'cache.sqlite3',
                           session=session,retry_backoff=0,**kwargs),session

def test_exact_dedup_and_disk(tmp_path):
    c,s=client(tmp_path,[Response({'result':[docs('Q'),docs('q')]})])
    assert c.retrieve(['Q','Q','q'])==[docs('Q'),docs('Q'),docs('q')]
    assert s.calls[0]['json']['queries']==['Q','q']
    assert c.last_stats['logical_queries']==3 and c.last_stats['backend_query_attempts']==2
    other,backend=client(tmp_path,[])
    assert other.retrieve(['Q','q','Q'])==[docs('Q'),docs('q'),docs('Q')]
    assert not backend.calls and other.last_stats['cache_hits']==3

@pytest.mark.parametrize('error,status,expected,retries',[(requests.Timeout('late'),200,'timeout',2),
    (None,503,'http_503',2),(None,400,'http_400',0)])
def test_retry_bounds(tmp_path,error,status,expected,retries):
    c,s=client(tmp_path,[error or Response({},status)]*3)
    assert c.retrieve(['x','x'])==[None,None]
    assert len(s.calls)==retries+1 and c.last_stats['retries']==retries
    assert c.last_outcomes[0]['error_category']==expected
    assert c.last_stats['failures']==2

def test_partial_cache_failure(tmp_path):
    c,s=client(tmp_path,[Response({'result':[docs('x')]}),requests.ConnectionError('down')],max_retries=0)
    c.retrieve(['x'])
    assert c.retrieve(['x','new'])==[docs('x'),None]
    assert c.last_outcomes[0]['cache_hit']
    assert c.last_stats['backend_query_attempts']==1

def test_malformed_never_cached(tmp_path):
    c,s=client(tmp_path,[Response({'result':[[{'wrong':1}]]}),Response({'result':[[]]})])
    assert c.retrieve(['x'])==[None]
    assert c.retrieve(['x'])==[[]]
    assert c.retrieve(['x'])==[[]] and len(s.calls)==2

@pytest.mark.parametrize('change',[{'index_version':'next'}, {'topk':2}])
def test_namespace_and_topk(tmp_path,change):
    c,_=client(tmp_path,[Response({'result':[docs('old')]})]);c.retrieve(['x'])
    topk=change.pop('topk',3)
    other,s=client(tmp_path,[Response({'result':[docs('new')]})],**change)
    assert other.retrieve(['x'],topk)==[docs('new')] and len(s.calls)==1

def test_corrupt_disk_falls_back(tmp_path):
    (tmp_path/'cache.sqlite3').write_text('broken sqlite')
    c,s=client(tmp_path,[Response({'result':[docs('ok')]})])
    assert c.retrieve(['x'])==[docs('ok')] and len(s.calls)==1
