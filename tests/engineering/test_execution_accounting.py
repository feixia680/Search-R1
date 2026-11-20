from collections import defaultdict
from types import SimpleNamespace
from search_r1.llm_agent.generation import LLMGenerationManager
from search_r1.llm_agent.trajectory import new_trajectory,reward_components
from search_r1.search.retrieval_client import RetrievalClient


def test_cache_keeps_per_trajectory_search_cost(tmp_path):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'result':[[{'document':{'id':'1','contents':'Paris\nCapital'}}]]}
    class Session:
        def __init__(self):self.calls=[]
        def post(self,url,**kwargs):self.calls.append(kwargs);return Response()
    backend=Session()
    manager=LLMGenerationManager.__new__(LLMGenerationManager)
    manager.config=SimpleNamespace(topk=3,search_timeout=1,search_url='http://localhost/retrieve')
    manager._retrieval_client=RetrievalClient(manager.config.search_url,session=backend,cache_path=tmp_path/'cache.sqlite3')
    manager.retrieval_metrics=defaultdict(float)
    manager._trajectories=[new_trajectory(),new_trajectory()]
    for _ in range(2):
        manager.execute_predictions(['<search>Paris</search>']*2,'',[True,True])
    assert len(backend.calls)==1
    assert manager.retrieval_metrics['logical_queries']==4
    assert manager.retrieval_metrics['backend_query_attempts']==1
    for trace in manager._trajectories:
        assert trace['search_actions']==trace['retrieval_attempts']==trace['retrieval_successes']==2
        assert trace['duplicate_queries']==1
        assert trace['events'][1]['cache_hit'] is True
        assert abs(reward_components(0,trace,beta=.02,gamma=.05)['total_reward']+.09)<1e-9
