import json,re,time
import torch,pandas as pd,requests
from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList
ROOT="/g22538014wej/11x22f/Search-R1"; MODEL=ROOT+"/models/Qwen2.5-3B"; OUT=ROOT+"/logs/hotpot_inference_5.jsonl"
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda",low_cpu_mem_usage=True)
df=pd.read_parquet(ROOT+"/data/processed/hotpotqa/test.parquet").head(5)
class StopAtAction(StoppingCriteria):
 def __init__(self,ids): self.ids=ids
 def __call__(self,input_ids,scores,**kw):
  tail=input_ids[0].tolist()
  return any(len(tail)>=len(s) and tail[-len(s):]==s for s in self.ids)
stopids=[tok.encode(x,add_special_tokens=False) for x in ["</search>","</answer>"]]
def answer(q,prompt):
 ids=tok(prompt,return_tensors="pt").input_ids.to("cuda"); transcript=[]; queries=[]; docs=[]
 for turn in range(3):
  out=model.generate(ids,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id,stopping_criteria=StoppingCriteriaList([StopAtAction(stopids)]))
  text=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True); transcript.append(text)
  m=re.search(r"<search>(.*?)</search>",text,re.S)
  if m and turn<2:
   query=m.group(1).strip(); queries.append(query)
   result=requests.post("http://127.0.0.1:8000/retrieve",json={"queries":[query],"topk":3,"return_scores":True},timeout=30).json()["result"][0]
   passages=[d["document"]["contents"] for d in result]; docs.extend(passages)
   obs="\n".join(f"Doc {i+1}: {d}" for i,d in enumerate(passages))
   ids=torch.cat([ids,tok(f"{text}<information>{obs}</information>\n",return_tensors="pt").input_ids.to("cuda")],dim=1); continue
  break
 return {"question":q,"transcript":transcript,"queries":queries,"retrieved_documents":docs,"answer":(re.findall(r"<answer>(.*?)</answer>","".join(transcript),re.S) or [""])[-1].strip()}
with open(OUT,"w") as f:
 for i,row in df.iterrows():
  raw=row["prompt"]; prompt=raw[0]["content"] if hasattr(raw,"__len__") and len(raw)>0 else str(raw)
  qm=re.search(r"Question:\s*(.*)",prompt,re.S); q=qm.group(1).strip() if qm else prompt
  if tok.chat_template: prompt=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
  t=time.time(); rec=answer(q,prompt); rec["seconds"]=round(time.time()-t,2); f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
  print(f"{i+1}/5 done; searches={len(rec['queries'])}; answer={rec['answer'][:100]}",flush=True)
print("WROTE",OUT)
