"""Trajectory reward from trusted, model-visible observation tokens only."""
import json, os, re
from pathlib import Path
import pandas as pd
import torch
from verl.utils.reward_score import qa_em

def norm(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()

def facts_from_row(row):
    ctx = dict(zip(row['context']['title'], row['context']['sentences']))
    return {f'{t}:{int(i)}': str(ctx[t][int(i)]) for t, i in zip(row['supporting_facts']['title'], row['supporting_facts']['sent_id'])}

def coverage(observations, facts):
    seen, gains = set(), []
    for obs in observations:
        text = norm(obs)
        found = {key for key, sentence in facts.items() if norm(sentence) and norm(sentence) in text}
        gains.append(len(found - seen) / max(1, len(facts)))
        seen |= found
    return sum(gains), gains

def trusted_observations(tokenizer, responses, attention, generated):
    # info_mask distinguishes externally inserted tokens from model text.
    mask = (attention.bool() & ~generated.bool()).tolist()
    spans, start = [], None
    for i, selected in enumerate(mask + [False]):
        if selected and start is None: start = i
        if not selected and start is not None:
            text = tokenizer.decode(responses[start:i], skip_special_tokens=True)
            if '<information>' in text: spans.append(text)
            start = None
    return spans

class EvidenceRewardManager:
    def __init__(self, tokenizer, num_examine, format_score=0.):
        self.tokenizer, self.num_examine, self.format_score = tokenizer, num_examine, format_score
        self.variant = os.environ.get('EVIDENCE_VARIANT', 'em') if num_examine == 0 else 'em'
        self.alpha = float(os.environ.get('EVIDENCE_ALPHA', '0.5'))
        self.beta = float(os.environ.get('SEARCH_BETA', '0.05'))
        self.facts = None
        if self.variant != 'em':
            raw = pd.read_parquet('data/raw/hotpotqa_train.parquet')
            self.facts = {}
            for i, row in raw.iterrows():
                try: self.facts[i] = facts_from_row(row)
                except (KeyError, IndexError): pass

    def __call__(self, data):
        reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        components = []
        for i in range(len(data)):
            item = data[i]
            plen = item.batch['prompts'].shape[-1]
            pmask = item.batch['attention_mask'][:plen].bool()
            response = item.batch['responses']
            attention = item.batch['attention_mask'][plen:]
            n = int(attention.sum())
            text = self.tokenizer.decode(torch.cat([item.batch['prompts'][pmask], response[:n]]))
            em = qa_em.compute_score_em(text, item.non_tensor_batch['reward_model']['ground_truth'])
            observations = trusted_observations(self.tokenizer, response, attention, item.batch['info_mask'][plen:])
            cov, gains = (0., [])
            if self.facts is not None:
                index = int(item.non_tensor_batch['index'])
                if index not in self.facts: raise ValueError(f'Unknown training index {index}')
                cov, gains = coverage(observations, self.facts[index])
            total = em + (self.alpha * cov if self.variant != 'em' else 0.)
            if self.variant == 'evidence_cost': total -= self.beta * len(observations)
            if n: reward[i, n - 1] = total
            components.append(dict(em=float(em), coverage=cov, gains=gains, calls=len(observations), total=float(total)))
        print('EVIDENCE_REWARD_METRICS ' + json.dumps(dict(variant=self.variant, rows=components)), flush=True)
        return reward
