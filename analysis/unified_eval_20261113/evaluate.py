"""Frozen, common evaluation of two checkpoints. No optimizer or training."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.reward_score.qa_em import compute_score_em, em_check
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from search_r1.llm_agent.evidence import question_key, score_evidence

ROOT = Path('/g22538014wej/11x22f/Search-R1')
OUT = ROOT / 'analysis/unified_eval_20261113'

def prepare():
    df = pd.read_parquet(ROOT/'algorithm_a_data/evidence/test.parquet')
    train = pd.read_parquet(ROOT/'data/processed/hotpotqa/train.parquet')
    key = lambda p: question_key(p[0]['content'].rsplit('Question:',1)[-1].strip())
    old = df.sample(200, random_state=42).copy()
    train_q = {key(p) for p in train.prompt}
    old_q = {key(p) for p in old.prompt}
    assert not old_q & train_q
    pool = df[~df.index.isin(old.index)].copy()
    pool = pool[[key(p) not in train_q|old_q for p in pool.prompt]]
    pool = pool[[x['evidence']['status']=='valid' for x in pool.extra_info]]
    pool['_question_key'] = [key(p) for p in pool.prompt]
    pool = pool.drop_duplicates('_question_key').drop(columns='_question_key')
    fresh = pool.sample(500, random_state=20261113).copy()
    old['eval_cohort'] = 'original200'
    fresh['eval_cohort'] = 'heldout500'
    chosen = pd.concat([old,fresh], ignore_index=True)
    assert len({str(x['index']) for x in chosen.extra_info}) == 700
    manifest = [dict(sample_id=str(r.extra_info['index']),cohort=r.eval_cohort,
                     question=r.prompt[0]['content'],gold=list(r.reward_model['ground_truth']['target']))
                for r in chosen.itertuples()]
    OUT.mkdir(parents=True,exist_ok=True)
    text=json.dumps(manifest,ensure_ascii=False,indent=2)
    path=OUT/'manifest.json'
    if path.exists(): assert path.read_text()==text, 'Manifest changed'
    else: path.write_text(text)
    (OUT/'manifest.sha256').write_text(hashlib.sha256(text.encode()).hexdigest()+'\n')
    protocol=dict(seed=42,heldout_sampling_seed=20261113,original_n=200,heldout_n=500,
        heldout_excludes_old_questions=True,heldout_excludes_exact_normalized_training_questions=True,
        heldout_requires_valid_annotation=True,heldout_previously_selected_checkpoints=False,
        batch_size=16,greedy=True,topk=3,max_turns=2,max_start_length=512,
        max_prompt_length=2816,max_response_length=256,max_obs_length=1024,
        native_policy='autonomous',forced_policies=['fixed_one','fixed_two'],
        forced_prefixes=True,forced_first_query_shared_between_fixed_policies=True,
        confidence_threshold_policy=False,gold_labels_used_for_decisions=False,
        forced_action_content_sanitization='trim at first <; empty query falls back to question; close action tag enforced',
        timing_claims=False,training=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return chosen

class Worker:
    def __init__(self,path,tok):
        from vllm import LLM
        self.tok=tok
        self.llm=LLM(model=path,tokenizer=str(ROOT/'models/Qwen2.5-3B'),dtype='bfloat16',
            tensor_parallel_size=1,gpu_memory_utilization=.45,max_model_len=4096,
            enforce_eager=True,max_num_seqs=32,seed=42)
        self.cache={}
        self.empty_query_fallbacks=0

    def generate_sequences(self,batch):
        from vllm import SamplingParams
        assert not batch.non_tensor_batch, 'Reward metadata reached model worker'
        ids=batch.batch['input_ids'];mask=batch.batch['attention_mask']
        prompts=[ids[i][mask[i].bool()].long().tolist() for i in range(len(ids))]
        fixed=self.mode!='autonomous'
        if not fixed:
            params=SamplingParams(temperature=0,top_p=1,top_k=-1,n=1,max_tokens=256,
                                  ignore_eos=False,detokenize=False)
            result=self.llm.generate(prompt_token_ids=prompts,sampling_params=params,use_tqdm=False)
            outputs=[list(x.outputs[0].token_ids) for x in result]
        elif self.stage==0 and self.mode=='fixed_two':
            outputs=[self.cache[tuple(p)] for p in prompts]
        else:
            searching=self.stage < (1 if self.mode=='fixed_one' else 2)
            action='search' if searching else 'answer'
            thought=('I should retrieve evidence for the question.' if self.stage==0
                     else 'I should search for additional evidence or the missing link using a different query.') if searching else 'I will answer from the available evidence.'
            prefix=f'<think>{thought}</think>\n<{action}>'
            closing=f'</{action}>'
            prefix_ids=self.tok.encode(prefix,add_special_tokens=False)
            closing_ids=self.tok.encode(closing,add_special_tokens=False)
            max_new=256-len(prefix_ids)-len(closing_ids)-4
            result=self.llm.generate(prompt_token_ids=[p+prefix_ids for p in prompts],
                sampling_params=SamplingParams(temperature=0,top_p=1,top_k=-1,n=1,
                    max_tokens=max_new,stop=[closing]),use_tqdm=False)
            outputs=[]
            for p,x in zip(prompts,result):
                content=x.outputs[0].text.split('<',1)[0].strip()
                if searching and not content:
                    question=self.tok.decode(p,skip_special_tokens=False).rsplit('Question:',1)[-1].split('<|im_end|>',1)[0].strip()
                    content=question
                    self.empty_query_fallbacks+=1
                seq=self.tok.encode(prefix+content+closing,add_special_tokens=False)
                if len(seq)>256:
                    seq=prefix_ids+self.tok.encode(content,add_special_tokens=False)[:max_new]+closing_ids
                outputs.append(seq)
            if self.stage==0:
                for p,o in zip(prompts,outputs): self.cache[tuple(p)]=o
        self.stage+=1
        responses=torch.full((len(outputs),max(256,max(map(len,outputs)))),self.tok.pad_token_id,dtype=torch.long)
        for i,seq in enumerate(outputs):responses[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        return DataProto.from_dict({'responses':responses},meta_info={})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model');parser.add_argument('--label')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    chosen=prepare()
    if args.prepare_only:
        print('MANIFEST_READY',len(chosen),flush=True);return
    assert args.model and args.label
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-3B',local_files_only=True)
    dataset=RLHFDataset(str(ROOT/'algorithm_a_data/evidence/test.parquet'),tok,
                       max_prompt_length=2816,truncation='error')
    dataset.dataframe=chosen.drop(columns='eval_cohort')
    cohort={str(x['index']):c for x,c in zip(chosen.extra_info,chosen.eval_cohort)}
    worker=Worker(args.model,tok)
    config=GenerationConfig(max_turns=2,max_start_length=512,max_prompt_length=2816,
        max_response_length=256,max_obs_length=1024,num_gpus=1,
        search_url='http://127.0.0.1:8013/retrieve',topk=3)
    for mode in ['autonomous','fixed_one','fixed_two']:
        path=OUT/f'{args.label}_{mode}.jsonl'
        assert not path.exists(),f'Refusing to overwrite {path}'
        worker.mode=mode
        rows=[];start=time.perf_counter()
        with path.open('w') as stream:
            for bi,raw in enumerate(DataLoader(dataset,batch_size=16,shuffle=False,drop_last=False,collate_fn=collate_fn)):
                b=DataProto.from_single_dict(raw)
                g=b.pop(batch_keys=['input_ids','attention_mask','position_ids'])
                assert not g.non_tensor_batch
                g.meta_info={'do_sample':False,'validate':True,'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id}
                worker.stage=0
                manager=LLMGenerationManager(tok,worker,config,is_validation=True)
                result=manager.run_llm_loop(g,g.batch['input_ids'][:,-512:].clone())
                for k in result.batch.keys():result.batch[k]=result.batch[k].long()
                for i in range(len(b)):
                    trace=result.non_tensor_batch['trajectory'][i]
                    ref=b.non_tensor_batch['extra_info'][i]['evidence']
                    evidence=score_evidence(ref,trace)
                    trace.update(evidence)
                    sid=str(b.non_tensor_batch['extra_info'][i]['index'])
                    gold=b.non_tensor_batch['reward_model'][i]['ground_truth']
                    width=result.batch['prompts'].shape[1]
                    p_len=int(result.batch['attention_mask'][i,:width].sum())
                    r_len=int(result.batch['attention_mask'][i,width:].sum())
                    seq=tok.decode(torch.cat([result.batch['prompts'][i,-p_len:],result.batch['responses'][i,:r_len]]))
                    trace.update(label=args.label,policy=mode,cohort=cohort[sid],sample_id=sid,
                        gold=list(gold['target']),em=float(compute_score_em(seq,gold)),
                        final_answer_em=int(em_check(trace['final_answer'],list(gold['target']))) if trace['final_answer'] else 0,
                        model=args.model)
                    if mode.startswith('fixed'):
                        assert trace['retrieval_attempts']==(1 if mode=='fixed_one' else 2)
                    assert trace['retrieval_failures']==0, 'Retrieval failure invalidates comparison'
                    assert trace['retrieval_attempts']<=2
                    stream.write(json.dumps(trace,ensure_ascii=False)+'\n')
                    rows.append(trace)
                stream.flush()
                print('PROGRESS',json.dumps(dict(label=args.label,policy=mode,n=len(rows),
                      correct=sum(x['em'] for x in rows),seconds=time.perf_counter()-start)),flush=True)
        print('POLICY_COMPLETE',args.label,mode,len(rows),flush=True)
    (OUT/f'{args.label}_COMPLETE.json').write_text(json.dumps(dict(model=args.model,
        label=args.label,n=700,policies=3,empty_query_fallbacks=worker.empty_query_fallbacks),indent=2))

if __name__=='__main__':main()
