import os,sys,json,time,re,argparse,hashlib
from pathlib import Path
ROOT=Path("/g22538014wej/11x22f/Search-R1")
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
from search_r1.llm_agent.generation import GenerationConfig,LLMGenerationManager
from verl.utils.reward_score.qa_em import compute_score_em,extract_solution,em_check
from torch.utils.data import DataLoader
from vllm import LLM,SamplingParams
class Worker:
 def __init__(self,path,tok):
  self.tok=tok
  self.llm=LLM(model=path,tokenizer=str(ROOT/"models/Qwen2.5-3B"),dtype="bfloat16",tensor_parallel_size=1,gpu_memory_utilization=.30,max_model_len=3072,enforce_eager=True,max_num_seqs=16,seed=42)
  self.params=SamplingParams(temperature=0,top_p=1,top_k=-1,n=1,max_tokens=256,ignore_eos=False,detokenize=False)
 def generate_sequences(self,b):
  ids=b.batch["input_ids"]; mask=b.batch["attention_mask"]
  prompts=[ids[i][mask[i].bool()].long().tolist() for i in range(len(ids))]
  out=self.llm.generate(prompt_token_ids=prompts,sampling_params=self.params,use_tqdm=False)
  responses=torch.full((len(out),256),self.tok.pad_token_id,dtype=torch.long)
  for i,o in enumerate(out):
   seq=list(o.outputs[0].token_ids);responses[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
  return DataProto.from_dict({"responses":responses},meta_info={})
class Manager(LLMGenerationManager):
 def execute_predictions(self,predictions,pad_token,active_mask=None,do_search=True):
  actions,contents=self.postprocess_predictions(predictions)
  obs,dones,valid,search=super().execute_predictions(predictions,pad_token,active_mask,do_search)
  for i,active in enumerate(active_mask):
   if active:
    self.traces[i].append(dict(action=actions[i],content=contents[i],raw=predictions[i],retrieval_executed=bool(do_search and actions[i]=="search"),observation=obs[i]))
  return obs,dones,valid,search
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--label",required=True);p.add_argument("--out",required=True)
 args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 tok=AutoTokenizer.from_pretrained(ROOT/"models/Qwen2.5-3B",local_files_only=True)
 dataset=RLHFDataset(str(ROOT/"data/processed/hotpotqa/test.parquet"),tok,max_prompt_length=2816,truncation="error")
 dataset.dataframe=dataset.dataframe.sample(200,random_state=42)
 data=dataset.dataframe.to_dict("records")
 manifest=[dict(order=i,index=int(row["extra_info"]["index"]),question=row["prompt"][0]["content"],gold=list(row["reward_model"]["ground_truth"]["target"])) for i,row in enumerate(data)]
 (out/"evaluation_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 worker=Worker(args.model,tok)
 config=GenerationConfig(max_turns=2,max_start_length=512,max_prompt_length=2816,max_response_length=256,max_obs_length=1024,num_gpus=1,search_url="http://127.0.0.1:8013/retrieve",topk=3)
 rows=[];start=time.perf_counter()
 with (out/(args.label+".jsonl")).open("w") as stream:
  for bi,batch in enumerate(DataLoader(dataset,batch_size=8,shuffle=False,drop_last=True,collate_fn=collate_fn)):
   b=DataProto.from_single_dict(batch);g=b.pop(batch_keys=["input_ids","attention_mask","position_ids"])
   g.meta_info={"do_sample":False,"validate":True,"eos_token_id":tok.eos_token_id,"pad_token_id":tok.pad_token_id}
   manager=Manager(tok,worker,config,is_validation=True);manager.traces=[[] for _ in range(8)]
   started=time.perf_counter();result=manager.run_llm_loop(g,g.batch["input_ids"][:,-512:].clone());seconds=time.perf_counter()-started
   for key in result.batch.keys(): result.batch[key]=result.batch[key].long()
   for i in range(8):
    promptlen=int(result.batch["attention_mask"][i,:result.batch["prompts"].shape[1]].sum())
    response_len=int(result.batch["attention_mask"][i,result.batch["prompts"].shape[1]:].sum())
    prompt_ids=result.batch["prompts"][i,-promptlen:]
    response_ids=result.batch["responses"][i,:response_len]
    text=tok.decode(torch.cat([prompt_ids,response_ids]))
    gold=b.non_tensor_batch["reward_model"][i]["ground_truth"]
    score=compute_score_em(text,gold)
    actions=manager.traces[i]
    finals=[e["content"] for e in actions if e["action"]=="answer"]
    strict=em_check(finals[-1],list(gold["target"])) if finals else 0
    r=dict(label=args.label,order=bi*8+i,sample_id=int(b.non_tensor_batch["index"][i]),gold=list(gold["target"]),legacy_em=score,action_em=strict,extracted_answer=extract_solution(text),final_action_answer=finals[-1] if finals else None,retrieval_calls=sum(e["retrieval_executed"] for e in actions),invalid_actions=sum(e["action"] is None for e in actions),batch_seconds=seconds,events=actions,sequence=text)
    stream.write(json.dumps(r,ensure_ascii=False)+"\n");stream.flush();rows.append(r)
   print(json.dumps(dict(label=args.label,batch=bi+1,n=len(rows),correct=sum(r["legacy_em"] for r in rows),elapsed=time.perf_counter()-start)),flush=True)
 summary=dict(label=args.label,n=len(rows),correct=sum(r["legacy_em"] for r in rows),em=np.mean([r["legacy_em"] for r in rows]),action_correct=sum(r["action_em"] for r in rows),mean_search=np.mean([r["retrieval_calls"] for r in rows]),missing_final=sum(r["final_action_answer"] is None for r in rows),seconds=time.perf_counter()-start,model=args.model,settings=dict(backend="vllm",dtype="bfloat16",batch_size=8,seed=42,greedy=True,max_turns=2,max_response_length=256,max_start_length=512,max_prompt_length=2816,max_obs_length=1024,topk=3,dataset_sample_seed=42))
 (out/(args.label+"_summary.json")).write_text(json.dumps(summary,indent=2))
 print("EVALUATION_COMPLETE",json.dumps(summary),flush=True)
if __name__=="__main__":main()

