"""CPU replay of saved model outputs; no model inference, HTTP or training."""
import argparse
import json
import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.trainer.main_ppo import RewardManager
from search_r1.llm_agent.evidence import score_evidence
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from search_r1.llm_agent.trajectory import reward_components


def documents_from_observation(observation):
    prefix, suffix = '\n\n<information>', '</information>\n\n'
    if not observation.startswith(prefix) or not observation.endswith(suffix):
        raise ValueError('Historical observation has unexpected wrappers')
    content = observation[len(prefix):-len(suffix)]
    matches = re.findall(r'Doc \d+\(Title: (.*?)\) (.*?)(?=\nDoc \d+\(Title:|\Z)', content, re.S)
    docs = [dict(document=dict(id=f'replay-{i}', contents=title + '\n' + body))
            for i, (title, body) in enumerate(matches)]
    rebuilt = ''.join(f'Doc {i+1}(Title: {title}) {body}\n' for i, (title, body) in enumerate(matches)).strip()
    if rebuilt != content:
        raise ValueError('Cannot reconstruct historical document formatting exactly')
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-root', required=True)
    parser.add_argument('--annotated-data', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    base, out = Path(args.base_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / '.gitignore').write_text('*\n')
    tok = AutoTokenizer.from_pretrained(base / 'models/Qwen2.5-3B', local_files_only=True)
    dataset = pd.read_parquet(args.annotated_data)
    lookup = {int(row['extra_info']['index']): row for row in dataset.to_dict('records')}
    saved = [json.loads(s) for s in (base / 'analysis/grpo_20261112/step500.jsonl').read_text().splitlines()]
    results, review = [], []
    for record in saved:
        row = lookup[int(record['sample_id'])]
        reference = row['extra_info']['evidence']
        if reference['status'] != 'valid':
            results.append(dict(sample_id=record['sample_id'], status=reference['status']))
            continue
        events = record['events']
        class ReplayWorker:
            calls = 0
            def generate_sequences(worker, batch):
                if batch.non_tensor_batch:
                    raise ValueError('Reward metadata reached policy generation')
                text = events[worker.calls]['raw']
                worker.calls += 1
                ids = tok([text], return_tensors='pt', add_special_tokens=False).input_ids
                return DataProto.from_dict({'responses': ids})
        worker = ReplayWorker()
        manager = LLMGenerationManager(tok, worker, GenerationConfig(
            max_turns=2, max_start_length=512, max_prompt_length=2816,
            max_response_length=256, max_obs_length=1024, num_gpus=1))
        encoded = tok([row['prompt'][0]['content']], return_tensors='pt', add_special_tokens=False)
        batch = DataProto.from_dict(dict(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
            position_ids=torch.arange(encoded.input_ids.shape[1]).unsqueeze(0)))
        batches = [{'result': [documents_from_observation(event['observation'])]}
                   for event in events if event['retrieval_executed']]
        with patch.object(manager, '_batch_search', side_effect=batches):
            output = manager.run_llm_loop(batch, encoded.input_ids[:, -512:])
        trace = output.non_tensor_batch['trajectory'][0]
        evidence = score_evidence(reference, trace)
        components = reward_components(record['action_em'], trace, alpha=.5, beta=.02,
                                       require_final_answer=True, max_search_calls=2,
                                       evidence_gain=evidence['evidence_gain'])
        result = dict(sample_id=record['sample_id'], raw_id=reference['raw_id'],
                      question=row['prompt'][0]['content'].split('Question:', 1)[-1].strip(),
                      answer=trace['final_answer'], answer_em=record['action_em'],
                      supporting_facts=list(reference['supporting_facts']), trajectory=trace)
        result.update(evidence)
        result.update(components)
        # Match ray_trainer's post-generation integer conversion contract.
        reward_batch = DataProto.from_dict({key: value.long() for key, value in output.batch.items()},
            non_tensors=dict(data_source=[row['data_source']], index=[record['sample_id']],
                reward_model=[row['reward_model']], extra_info=[row['extra_info']], trajectory=[trace]))
        reward_manager = RewardManager(tok, 0, alpha=.5, beta=.02,
                                      require_final_answer=True, max_search_calls=2)
        actual_reward = float(reward_manager(reward_batch).sum())
        if abs(actual_reward - result['total_reward']) > 1e-6:
            raise ValueError('RewardManager disagrees with independently computed reward components')
        result['reward_manager_verified'] = True
        results.append(result)
        cov = evidence['evidence_coverage']
        category = ('full' if cov == 1 else 'none' if cov == 0 else 'partial') + ('_em1' if record['action_em'] else '_em0')
        if category not in [r['category'] for r in review]:
            review.append(dict(category=category, result=result))
    with (out / 'replayed_trajectories.jsonl').open('w') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    scored = [r for r in results if 'evidence_coverage' in r]
    summary = dict(replay_count=len(results), scored_count=len(scored),
        annotation_excluded=len(results)-len(scored),
        reward_manager_verified=sum(r.get('reward_manager_verified', False) for r in scored),
        supporting_fact_count=sum(r['evidence_fact_count'] for r in scored),
        visible_fact_hits=sum(r['evidence_hit_count'] for r in scored),
        fully_covered_questions=sum(r['evidence_coverage'] == 1 for r in scored),
        review_categories=[r['category'] for r in review],
        reference_labels_reached_policy=False, gpu_inference=False, optimizer_updates=0,
        interpretation='Saved text replay with exact observation-format reconstruction; not fresh inference or an accuracy benchmark.')
    (out / 'replay_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ['# Evidence reward case review', '', summary['interpretation'],
             '', 'Coefficients below (.5/.02) are arithmetic fixtures, not tuned training recommendations.',
             'em0/em1 refer only to the existing exact-match labels; they are not a new semantic correctness judgment.']
    for item in review:
        r = item['result']
        lines.extend(['', f"## {item['category']} — sample {r['sample_id']}", '', r['question'],
                      '', f"Answer: {r['answer']} / EM={r['answer_em']}",
                      f"Coverage={r['evidence_coverage']:.4f}; reward={r['total_reward']:.4f}", '', 'Reference facts:'])
        hits = {x for e in r['evidence_events'] for x in e['matched_fact_ids']}
        for fact in r['supporting_facts']:
            lines.append(f"- {'HIT' if fact['fact_id'] in hits else 'MISS'} [{fact['fact_id']}] {fact['sentence']}")
        for event in r['trajectory']['events']:
            if event['action'] != 'search':
                continue
            lines.extend(['', 'Query: ' + event['query'], '', 'Actually visible document bodies:'])
            for view in event.get('visible_views', []):
                for chunk in view['chunks']:
                    for body in chunk['document_bodies']:
                        lines.extend(['', f"[{body['title']}] {body['text']}"])
    (out / 'CASE_REVIEW.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
