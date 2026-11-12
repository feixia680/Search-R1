"""Paired initial/final EM on a fixed held-out set; exploratory only."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    records = [json.loads(s) for s in (run/'traces/validation.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for row in records:
        groups[row['global_step']].append(row)
    assert len(groups) == 2 and 0 in groups, sorted(groups)
    final_step = max(groups)
    before = {r['sample_id']: r for r in groups[0]}
    after = {r['sample_id']: r for r in groups[final_step]}
    assert len(before) == len(groups[0]) == len(after) == len(groups[final_step]) == 32
    assert before.keys() == after.keys()
    manifest = json.loads((run/'data/manifest.json').read_text())
    assert set(before) == set(map(str, manifest['val32']['sample_ids']))
    train = [json.loads(s) for s in (run/'traces/train.jsonl').read_text().splitlines()]
    assert len(train) == 80 and len({r['global_step'] for r in train}) == 10
    paired = []
    for sample_id, initial in before.items():
        final = after[sample_id]
        assert initial['ground_truth'] == final['ground_truth']
        assert initial['em'] == initial['total_reward']
        assert final['em'] == final['total_reward']
        paired.append(dict(sample_id=sample_id, ground_truth=initial['ground_truth'],
                           before_answer=initial['final_answer'], after_answer=final['final_answer'],
                           before_em=initial['em'], after_em=final['em'],
                           before_calls=initial['retrieval_attempts'], after_calls=final['retrieval_attempts']))
    count = len(paired)
    initial_correct = sum(r['before_em'] for r in paired)
    final_correct = sum(r['after_em'] for r in paired)
    summary = dict(validation_questions=count, training_questions=40, optimizer_updates=10,
                   initial_correct=initial_correct, final_correct=final_correct,
                   initial_em=initial_correct/count, final_em=final_correct/count,
                   em_change_percentage_points=100*(final_correct-initial_correct)/count,
                   wrong_to_right=sum(r['after_em']>r['before_em'] for r in paired),
                   right_to_wrong=sum(r['after_em']<r['before_em'] for r in paired),
                   initial_mean_calls=sum(r['before_calls'] for r in paired)/count,
                   final_mean_calls=sum(r['after_calls'] for r in paired)/count,
                   initial_missing_answers=sum(r['before_answer'] is None for r in paired),
                   final_missing_answers=sum(r['after_answer'] is None for r in paired),
                   legacy_em_mismatches=sum(r['em'] != r['final_answer_em'] for r in records),
                   evaluation='greedy, same 32 questions, unchanged retriever; exploratory, one seed')
    (run/'em_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (run/'paired_answers.jsonl').open('w') as out:
        for row in paired:
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
