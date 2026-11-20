import argparse, json, math, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.trainer.evidence_reward_20261112 import facts_from_row, coverage
from verl.utils.reward_score.qa_em import em_check

ROOT = Path('/g22538014wej/11x22f/Search-R1')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--label',required=True); p.add_argument('--out',required=True); p.add_argument('--n',type=int,default=200)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-3B',local_files_only=True)
    llm=LLM(model=a.model,tokenizer=str(ROOT/'models/Qwen2.5-3B'),dtype='bfloat16',gpu_memory_utilization=.45,max_model_len=4096,enforce_eager=True,max_num_seqs=16,seed=42)
    def generate(prompts,stop,limit,probs=False):
        ids=[tok.encode(x,add_special_tokens=False)[-3200:] for x in prompts]
        params=SamplingParams(temperature=0,max_tokens=limit,stop=[stop],logprobs=1 if probs else None)
        return [x.outputs[0] for x in llm.generate(prompt_token_ids=ids,sampling_params=params,use_tqdm=False)]
    def retrieve(queries):
        res=requests.post('http://127.0.0.1:8013/retrieve',json={'queries':queries,'topk':3,'return_scores':True},timeout=120);res.raise_for_status()
        observations=[]
        for docs in res.json()['result']:
            text=''
            for i,doc in enumerate(docs):
                content=doc['document']['contents']; title,_,body=content.partition('\n')
                text+=f'Doc {i+1}(Title: {title}) {body}\n'
            observation='\n\n<information>'+text.strip()+'</information>\n\n'
            observations.append(tok.decode(tok.encode(observation,add_special_tokens=False)[:1024],skip_special_tokens=True))
        return observations
    data=pd.read_parquet(ROOT/'data/processed/hotpotqa/test.parquet').sample(a.n,random_state=42)
    raw=pd.read_parquet(ROOT/'data/raw/hotpotqa_validation.parquet')
    canon=lambda q: q.strip().rstrip('?').strip()
    rawmap={canon(r['question']):r for _,r in raw.iterrows()}
    rows=[]; started=time.time()
    stream=(out/(a.label+'.jsonl')).open('w')
    for start in range(0,len(data),8):
        batch=data.iloc[start:start+8]
        original=[r['prompt'][0]['content'] for _,r in batch.iterrows()]
        questions=[s.rsplit('Question:',1)[-1].strip() for s in original]
        qprefix=[s+'<think>I should retrieve evidence for the question.</think>\n<search>' for s in original]
        q1=[x.text.strip().split('<')[0].strip() or questions[i] for i,x in enumerate(generate(qprefix,'</search>',96))]
        obs1=retrieve(q1)
        state1=[p+q+'</search>'+o for p,q,o in zip(qprefix,q1,obs1)]
        aprefix=[s+'<think>I will answer from the available evidence.</think>\n<answer>' for s in state1]
        ans1=generate(aprefix,'</answer>',64,True)
        confidence=[math.exp(x.cumulative_logprob/max(1,len(x.token_ids))) for x in ans1]
        # Fixed threshold, set before observing any evaluation labels.
        demand=[c<.8 or not x.text.strip() for c,x in zip(confidence,ans1)]
        qprefix2=[s+'<think>I should search for additional evidence or the missing link using a different query.</think>\n<search>' for s in state1]
        q2=[x.text.strip().split('<')[0].strip() or questions[i] for i,x in enumerate(generate(qprefix2,'</search>',96))]
        obs2=retrieve(q2)
        state2=[p+q+'</search>'+o for p,q,o in zip(qprefix2,q2,obs2)]
        ans2=generate([s+'<think>I will answer from the available evidence.</think>\n<answer>' for s in state2],'</answer>',64,True)
        for i,(_,sample) in enumerate(batch.iterrows()):
            gold=list(sample['reward_model']['ground_truth']['target'])
            facts=facts_from_row(rawmap[canon(questions[i])])
            c1,_=coverage([obs1[i]],facts); c2,gains=coverage([obs1[i],obs2[i]],facts)
            r=dict(order=start+i,index=int(sample['extra_info']['index']),question=questions[i],gold=gold,query1=q1[i],query2=q2[i],observation1=obs1[i],observation2=obs2[i],answer1=ans1[i].text.strip(),answer2=ans2[i].text.strip(),em1=int(em_check(ans1[i].text.strip(),gold)),em2=int(em_check(ans2[i].text.strip(),gold)),confidence=confidence[i],adaptive_second=demand[i],coverage1=c1,coverage2=c2,gain2=gains[1],fact_count=len(facts))
            stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();rows.append(r)
        print(json.dumps(dict(label=a.label,n=len(rows),elapsed=time.time()-started)),flush=True)
    stream.close()
    policies={}
    for mode in ['fixed_one','fixed_two','adaptive']:
        second=np.array([mode=='fixed_two' or (mode=='adaptive' and r['adaptive_second']) for r in rows])
        em=np.array([r['em2'] if s else r['em1'] for r,s in zip(rows,second)])
        policies[mode]=dict(n=len(rows),em=float(em.mean()),correct=int(em.sum()),mean_calls=float(1+second.mean()),second_calls=int(second.sum()),coverage=float(np.mean([r['coverage2'] if s else r['coverage1'] for r,s in zip(rows,second)])),mean_added_coverage=float(np.mean([r['gain2'] if s else 0 for r,s in zip(rows,second)])))
    summary=dict(label=a.label,model=a.model,policies=policies,seconds=time.time()-started,settings=dict(seed=42,adaptive_confidence_threshold=.8,adaptive_uses_gold=False,greedy=True,topk=3,max_obs_tokens=1024,forced_query_and_answer_prefixes=True,paired_shared_first_hop=True,adaptive_counterfactual_branch_reuse=True))
    (out/(a.label+'_summary.json')).write_text(json.dumps(summary,indent=2))
    print('POLICY_EVAL_COMPLETE '+json.dumps(summary),flush=True)
if __name__=='__main__': main()
