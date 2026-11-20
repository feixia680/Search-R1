"""CPU-only diagnosis of existing GRPO logs and trajectories. No inference/training."""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path

RUNS = {
    'em': ('hotpotqa-qwen25-3b-grpo-2gpu.log', None),
    'evidence': ('algorithm_a_runs/evidence_2gpu_a0.1_s42_20261112_174015/train.log',
                 'algorithm_a_runs/evidence_2gpu_a0.1_s42_20261112_174015/traces/train.jsonl'),
    'cost': ('algorithm_a_runs/cost_state_2gpu_a0.1_b0.005_l0.01_s42_20261113_095134/train.log',
             'algorithm_a_runs/cost_state_2gpu_a0.1_b0.005_l0.01_s42_20261113_095134/traces/train.jsonl'),
}
TOL = 1e-8


def f32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]


def spread(xs):
    return max(xs) - min(xs)


def avg(xs):
    return statistics.mean(xs) if xs else None


def clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    return obj


def dump(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(clean(rows))


def parse_log(path):
    result = []
    for number, raw in enumerate(path.open(errors='replace'), 1):
        line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        m = re.search(r'\bstep:(\d+) - ', line)
        if not m:
            continue
        row = {'step': int(m[1]), 'line': number}
        for part in line[m.start():].strip().split(' - '):
            key, sep, value = part.rpartition(':')
            if sep:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        result.append(row)
    return result


def log_summary(rows):
    train = [r for r in rows if 'actor/pg_loss' in r]
    bad = [r for r in train if any(not math.isfinite(v) for k, v in r.items() if k.startswith('actor/'))]
    first = int(bad[0]['step']) if bad else None
    by_field = {}
    for r in train:
        for k, v in r.items():
            if not math.isfinite(v) and k not in by_field:
                by_field[k] = {'step': int(r['step']), 'line': r['line'], 'value': str(v)}
    zero = [r for r in train if r.get('critic/advantages/max') == 0 and r.get('critic/advantages/min') == 0]
    return dict(train_steps=len(train), first_actor_nonfinite_step=first,
                nonfinite_actor_steps=len(bad), first_nonfinite_by_field=by_field,
                zero_advantage_batch_count=len(zero), zero_advantage_batch_rate=len(zero) / len(train),
                validation=[dict(step=int(r['step']), em=r['val/test_score/hotpotqa'])
                            for r in rows if 'val/test_score/hotpotqa' in r],
                nan_context=[r for r in train if first and first - 6 <= r['step'] <= first + 3],
                first100=train[:100], last100=train[-100:])


def groups_from_trace(path, name):
    # Keep compact fields only, not document bodies or generated responses.
    groups = collections.defaultdict(list)
    row_count = 0
    first_bad_logged_reward = None
    for raw in path.open():
        x = json.loads(raw)
        assert x.get('split') == 'train'
        key = (int(x['global_step']), int(x['reward_call']), str(x['sample_id']))
        em = f32(x['em'])
        gain = float(x.get('evidence_gain', 0))
        logged = f32(x['total_reward'])
        no_cost = f32(em + .1 * gain)
        expected = (em + float(x.get('evidence_reward', 0))
                    - sum(float(x.get(k, 0)) for k in
                          ('search_cost', 'repetition_cost', 'invalid_cost', 'after_full_cost', 'failure_guard_cost')))
        assert math.isclose(expected, x['total_reward'], rel_tol=1e-7, abs_tol=1e-7)
        if not math.isfinite(logged) and first_bad_logged_reward is None:
            first_bad_logged_reward = key[0]
        if name == 'evidence':
            assert math.isclose(logged, no_cost, abs_tol=1e-7)
        events = x.get('events', [])
        generated = [e.get('generated_text', '') for e in events]
        groups[key].append(dict(em=em, coverage=float(x.get('evidence_coverage', 0)),
            reward=logged, evidence_reward=no_cost,
            final_present=bool(str(x.get('final_answer') or '').strip()),
            failures=x.get('retrieval_failures', 0), invalid=x.get('invalid_actions', 0),
            calls=x.get('retrieval_attempts', 0), length=x.get('response_tokens', 0),
            guard=x.get('failure_guard_cost', 0),
            after_full=x.get('after_full_search_calls'),
            bang_only=bool(generated) and all(t.strip() and set(t.strip()) == {'!'} for t in generated)))
        row_count += 1
    records = []
    for (step, call, sid), xs in sorted(groups.items()):
        assert len(xs) == 2, (name, step, sid, len(xs))
        ems = [x['em'] for x in xs]
        rewards = [x['reward'] for x in xs]
        evidence = [x['evidence_reward'] for x in xs]
        valid = all(x['final_present'] for x in xs)
        failures = any(x['failures'] for x in xs)
        reward_signal = spread(rewards) > TOL
        em_signal = spread(ems) > TOL
        evidence_signal = spread(evidence) > TOL
        all_wrong = all(x == 0 for x in ems)
        if failures:
            category = 'retrieval_failure'
        elif not valid:
            category = 'missing_answer'
        elif not reward_signal:
            category = 'zero_reward_difference'
        elif em_signal:
            category = 'answer_difference'
        elif all_wrong:
            category = 'all_wrong_evidence_difference'
        else:
            category = 'all_correct_evidence_difference'
        weight = 3 if valid and not failures and evidence_signal else 1
        std = statistics.stdev(rewards)
        records.append(dict(run=name, step=step, reward_call=call, sample_id=sid, group_size=len(xs),
            reward_std=std, reward_range=spread(rewards), zero_reward=not reward_signal,
            zero_em=not em_signal, zero_evidence_counterfactual=not evidence_signal,
            all_wrong=all_wrong, all_correct=all(x == 1 for x in ems),
            both_final_present=valid, any_retrieval_failure=failures,
            rescued_by_evidence=(not em_signal and evidence_signal),
            valid_all_wrong_rescued=(all_wrong and valid and not failures and evidence_signal),
            category=category, candidate_weight=weight,
            mean_em=avg(ems), mean_coverage=avg([x['coverage'] for x in xs]),
            mean_calls=avg([x['calls'] for x in xs]), mean_length=avg([x['length'] for x in xs]),
            missing_answers=sum(not x['final_present'] for x in xs),
            invalid_actions=sum(x['invalid'] for x in xs),
            after_full_calls=(sum(x['after_full'] for x in xs)
                              if all(x['after_full'] is not None for x in xs) else None),
            bang_only_trajectories=sum(x['bang_only'] for x in xs),
            guard_count=sum(x['guard'] > 0 for x in xs),
            mean_abs_advantage=avg([abs((r - avg(rewards)) / (std + 1e-6)) for r in rewards])))
    return records, dict(rows=row_count, groups=len(records),
        unique_questions=len({r['sample_id'] for r in records}),
        group_size_histogram=dict(collections.Counter(r['group_size'] for r in records)),
        first_nonfinite_logged_reward_step=first_bad_logged_reward)


def group_stats(xs):
    n = len(xs)
    return dict(groups=n, trajectories=2 * n,
        zero_reward_group_rate=avg([int(x['zero_reward']) for x in xs]),
        zero_em_group_rate=avg([int(x['zero_em']) for x in xs]),
        zero_evidence_counterfactual_rate=avg([int(x['zero_evidence_counterfactual']) for x in xs]),
        rescued_by_evidence=sum(x['rescued_by_evidence'] for x in xs),
        valid_all_wrong_rescued=sum(x['valid_all_wrong_rescued'] for x in xs),
        categories=dict(collections.Counter(x['category'] for x in xs)),
        candidate_high_priority_groups=sum(x['candidate_weight'] > 1 for x in xs),
        mean_em=avg([x['mean_em'] for x in xs]),
        mean_coverage=avg([x['mean_coverage'] for x in xs]),
        mean_calls=avg([x['mean_calls'] for x in xs]),
        missing_answer_rate=sum(x['missing_answers'] for x in xs) / (2 * n),
        invalid_actions_per_trajectory=sum(x['invalid_actions'] for x in xs) / (2 * n),
        after_full_calls=(sum(x['after_full_calls'] for x in xs)
                          if all(x['after_full_calls'] is not None for x in xs) else None),
        bang_only_rate=sum(x['bang_only_trajectories'] for x in xs) / (2 * n),
        guarded_trajectories=sum(x['guard_count'] for x in xs))


LOG_FIELDS = ['actor/entropy_loss', 'actor/kl_loss', 'actor/grad_norm', 'actor/pg_loss',
              'actor/pg_clipfrac', 'actor/ppo_kl', 'response_length/mean',
              'env/finish_ratio', 'env/ratio_of_valid_action', 'critic/score/mean',
              'algorithm_a/invalid_actions', 'algorithm_a/after_full_cost', 'state_tokens/coverage']


def window_stats(xs):
    out = {}
    for key in LOG_FIELDS:
        values = [r[key] for r in xs if key in r and math.isfinite(r[key])]
        out[key] = avg(values)
        out[key + '_finite_n'] = len(values)
    return out


def write_report(out, summaries, group_rows, windows):
    e = summaries['evidence']
    all_e, early, late = e['trace_all'], e['trace_first100'], e['trace_last100']
    eg = [g for g in group_rows if g['run'] == 'evidence']
    missing_rescued = sum(g['rescued_by_evidence'] and not g['both_final_present'] for g in eg)
    assertions = []
    for name in ('evidence', 'cost'):
        gs = [g for g in group_rows if g['run'] == name]
        by_step = collections.defaultdict(list)
        for g in gs:
            by_step[g['step']].append(g)
        assert len(by_step) == 624 and all(len(v) == 8 for v in by_step.values())
        trace_zero_batches = sum(all(g['zero_reward'] for g in v) for v in by_step.values())
        assert trace_zero_batches == summaries[name]['zero_advantage_batch_count']
        assertions.append(dict(run=name, groups_per_step=8, rows_per_group=2,
                               trace_zero_batches=trace_zero_batches,
                               logged_zero_batches=summaries[name]['zero_advantage_batch_count']))
    dump(out / 'consistency_checks.json', assertions)
    cost = summaries['cost']
    lines = ['# GRPO 训练诊断与自适应采样依据', '',
        '分析对象：已有纯 EM、evidence、成本扩展三次训练。只读重放日志和训练轨迹；未启动训练、模型推理或新增题目评测。', '',
        '## 结论', '',
        f"1. evidence 末 100 步仍有 {late['zero_reward_group_rate']:.2%} 的题目组没有 outcome 奖励差异，自适应采样有明确诊断依据，但收益尚未验证。",
        f"2. 在 evidence 的同一批固定轨迹上，纯 EM 的零方差组比例为 {all_e['zero_em_group_rate']:.2%}；加入证据项后为 {all_e['zero_reward_group_rate']:.2%}，减少 {(all_e['zero_em_group_rate']-all_e['zero_reward_group_rate'])*100:.2f} 个百分点，相对减少 {(all_e['zero_em_group_rate']-all_e['zero_reward_group_rate'])/all_e['zero_em_group_rate']:.2%}。此为奖励重算，不是两次训练的因果对照。",
        f"3. {all_e['valid_all_wrong_rescued']} 个组的两条回答均答错、均有非空最终答案，但证据奖励不同；直接过滤全错组会丢失这些信号。",
        '4. 纯 EM 和 evidence 的策略熵下降，但未记录到非有限 actor 指标，且原训练验证 EM 上升。不能仅据熵下降写成“熵坍缩”或声称 Clip-Higher 必需。',
        '5. 成本扩展在 step 588 首次记录非有限 actor 指标，此时奖励和优势有限且非零；589 起生成退化。未保存微批前向/梯度张量，根因尚未确定。', '',
        '## 1. 数据口径与完整性', '',
        '- 每次训练记录 624 次更新（1–624）；625 是最后验证标签，不是保存的 step-625 checkpoint。最后定期保存的 checkpoint 为 step 500。',
        '- evidence、成本组各有 9,984 条训练轨迹、4,992 个题目组，每组两条 rollout、每步八组。按 (global_step, reward_call, sample_id) 分组，不跨更新混组。',
        '- 纯 EM 未找到逐题训练轨迹；只报告日志里的整批零优势代理指标，不伪造逐组比例。',
        '- reward 先转 float32，以匹配实际奖励 tensor；零差异阈值为 1e-8；组内标准差采用 sample std（ddof=1）。',
        '- 非有限值在 JSON 中显式记为字符串 nan/inf，窗口均值只使用有限值，并单独记录有限样本数。',
        '- 所有轨迹的奖励分量相加核验通过；由逐组轨迹算出的整批零优势数量与日志一致。输入路径、大小与 SHA256 在 source_manifest.json。',
        '- 零 outcome 优势表示该组没有这部分策略梯度，KL/熵正则仍可能产生梯度；不等于整次更新没有梯度。', '',
        '## 2. 训练日志对照', '',
        '窗口为 step 1–100 与 525–624；成本组末窗口的 actor 指标仅 63 个有限记录，不能与健康末窗口当作等价比较。', '',
        '|训练组|熵 前100→末100|KL loss 前100→末100|actor 非有限更新数|整批零优势/624|',
        '|---|---:|---:|---:|---:|']
    for name, s in summaries.items():
        a, b = s['log_first100'], s['log_last100']
        lines.append(f"|{name}|{a['actor/entropy_loss']:.4f} → {b['actor/entropy_loss']:.4f}|{a['actor/kl_loss']:.4f} → {b['actor/kl_loss']:.4f}|{s['nonfinite_actor_steps']}|{s['zero_advantage_batch_count']}/624 ({s['zero_advantage_batch_rate']:.2%})|")
    lines += ['', '这些训练日志来自不同策略与奖励配置，不能把所有差异归因于证据项。actor/kl_loss 是实现中 low_var_kl 的统计量，不是精确全词表 KL；日志保留三位小数，actor/lr 显示 0.000 不代表实际学习率为零。', '',
        '## 3. evidence 的逐组学习信号', '',
        '|窗口|题目组数|纯 EM 重算零差异|实际证据奖励零差异|双答错且答案完整但有证据信号|缺答案轨迹率|',
        '|---|---:|---:|---:|---:|---:|']
    for label, r in [('全程', all_e), ('前100步', early), ('末100步', late)]:
        lines.append(f"|{label}|{r['groups']}|{r['zero_em_group_rate']:.2%}|{r['zero_reward_group_rate']:.2%}|{r['valid_all_wrong_rescued']}|{r['missing_answer_rate']:.2%}|")
    lines += ['', f"证据项共使 {all_e['rescued_by_evidence']} 个原本 EM 无差异的组获得奖励差异；其中 {missing_rescued} 个组含缺答案轨迹。因此不能把全部新增差异都等同于高质量学习信号。", '',
        '|完整轨迹组分类|组数|', '|---|---:|']
    for key, count in all_e['categories'].items():
        lines.append(f'|{key}|{count}|')
    lines += ['', '分类按检索故障、缺答案、零奖励差异、答案差异、全错证据差异、全对证据差异依次互斥。invalid_actions 可能包含最终被纠正的早期动作，不能等同于缺答案。', '',
        '## 4. 长度与熵的解释边界', '',
        f"- evidence 平均 response 长度从 {e['log_first100']['response_length/mean']:.1f} 增至 {e['log_last100']['response_length/mean']:.1f} token；检索调用从 {early['mean_calls']:.3f} 增至 {late['mean_calls']:.3f}。该长度包含插入的检索 observation，不能直接写成推理文本变长或长度导致的梯度失衡。",
        f"- evidence 的无效动作从每轨迹 {early['invalid_actions_per_trajectory']:.3f} 降至 {late['invalid_actions_per_trajectory']:.3f}，缺答案率从 {early['missing_answer_rate']:.2%} 降至 {late['missing_answer_rate']:.2%}；这是格式行为改善，同时仍留有未作答问题。",
        '- 条件上下文、问题和生成轨迹在变化，观测熵均值不等于在固定提示分布上的因果熵比较。', '',
        '## 5. 成本扩展的 NaN 时间线（独立诊断，不纳入新方法）', '',
        '|step|总奖励均值|优势最大/最小|熵|梯度范数|全局 loss-mask 覆盖率|',
        '|---|---:|---:|---:|---:|---:|']
    ctx = json.loads((out / 'cost_nonfinite_context.json').read_text())
    for r in ctx:
        lines.append(f"|{int(r['step'])}|{r.get('algorithm_a/total_reward')}|{r.get('critic/advantages/max')} / {r.get('critic/advantages/min')}|{r.get('actor/entropy_loss')}|{r.get('actor/grad_norm')}|{r.get('state_tokens/coverage')}|")
    lines += ['',
        '- step 588 的 actor KL、熵、PG loss、梯度范数首次同时记录为 NaN；所有保存的轨迹标量奖励仍有限。',
        '- 记录粒度是整次更新，无法知道最先异常的微批、rank、前向算子或参数；不能断言是梯度爆炸、成本惩罚或零方差导致。',
        '- step 588 的全局 mask 覆盖率非零，仍不能排除某个微批的 mask 分母为零。',
        '- 静态代码风险：masked_mean 直接除 mask.sum；low_var_kl 先 exp 再 clamp；optimizer.step 前未拦截非有限梯度。它们是下一次加观测和保护的位置，不是已确认根因。',
        '- 589–624 的轨迹全部无有效最终答案、无实际检索，并重复输出叹号；最终 625 标签验证 200 题全部缺答案、EM 为 0。',
        f"- 更正先前粗读：全程“证据充分后继续搜索”实际为 {cost['trace_all']['after_full_calls']} 次，均出现在前100步；不是全程从未触发。", '',
        '## 6. 原训练验证结果（只复用已有日志）', '',
        '|训练组|125|250|375|500|625标签|', '|---|---:|---:|---:|---:|---:|']
    for name, s in summaries.items():
        by_step = {r['step']: r['em'] for r in s['validation']}
        lines.append('|' + name + '|' + '|'.join(f'{by_step[k]:.1%}' for k in [125, 250, 375, 500, 625]) + '|')
    lines += ['', '上表为各自训练内验证协议；不与 unified_eval_20261113 的统一执行器复测数字混算。原200题已用于验证和 checkpoint 选择，结果不代表独立测试集泛化。', '',
        '## 7. 下一步采样规则', '',
        '详见 SAMPLING_SPEC.md。建议只新增有上限、保留均匀探索的题目优先采样；保持现有 GRPO 与证据奖励公式。候选采样权重只源于训练集，没有用验证题决定难度。',
        f"离线候选中 {all_e['candidate_high_priority_groups']}/{all_e['groups']} 组满足两条最终答案均非空、无检索故障、证据总奖励有差异。candidate_training_priorities.json 只是历史诊断候选，未接入训练，也不证明节省训练量。", '',
        '## 8. 产物与复现', '',
        '- summary.json：精确计数与窗口统计；log_metrics.csv：全部日志指标。',
        '- group_diagnostics.csv：逐题逐步组内标准差、奖励差异、缺答案与分类。',
        '- step_windows.csv：每50步窗口；cost_nonfinite_context.json：NaN 前后完整指标。',
        '- consistency_checks.json：组数、每组样本数及日志/轨迹交叉核验。',
        '- candidate_training_priorities.json：训练题候选权重，未应用。',
        '', '```bash',
        'cd /g22538014wej/cityins1122/cityins',
        'python3 evaluation/diagnose_grpo_training.py', '```', '',
        '限制：每种方法只有一次训练；每组仅两条轨迹；无纯 EM 原始训练组；未保存异常微批张量。未改变训练奖励、启动脚本、checkpoint 或采样行为。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (out / 'SAMPLING_SPEC.md').write_text('''# 证据感知自适应采样：建议规格（尚未实现）

目标：提高固定 rollout 预算下可用于 outcome 策略梯度的问题比例；不预设 EM 或训练耗时提升。

1. 保持奖励 R=EM+0.1*新增可见支持事实覆盖率，成本系数为0。缺答案保护沿用原 evidence 设置，不能在同一对比中额外打开保护造成混杂。
2. 完整问题组是统计单位；保留每题两条 rollout，不把单条正确/错误轨迹拆开优先抽取。反馈只来自训练集。
3. 两条最终答案均非空、没有检索故障且总奖励有差异的组，作为高优先候选；双答错但证据不同仍保留。缺答案/故障组保持均匀探索，不用其奖励差异当成优先信号；全对组也不直接删除。
4. 最小候选：普通权重1，高优先权重3，提议概率 p_i=0.5/N+0.5*w_i/sum(w)。提议分布一半来自均匀抽样，权重始终有限；未见/过期题回到普通权重。批内去重会改变实际入选概率，需记录提议与接受数量。数值只是待验证起点。
5. 现有训练是一轮打乱、每题只出现一次。仅在 epoch 末调整权重对该轮题目频率没有效果；要用按步抽样且允许跨更新重复的题目索引，更新权重时避免 DataLoader 预取导致权重延迟。每个 batch 内题目去重，防止同题重复导致 uid 合并成四条/更多 rollout。
6. 每次观察后更新优先分数，并对重复观察做平滑/老化。历史轨迹每题只有两条，且来自不同训练阶段，不能当作固定难度真值。优先选从均匀权重起步并在线更新；历史候选只用于诊断或明确标注的可选初始化。
7. 公平对照：均匀组也使用相同的按步带放回抽样方式，并共享 batch内去重规则；保持模型初始化、优化器、seed、全局batch、group size、reward、学习率调度和最大搜索次数一致。旧的单轮无放回 evidence 训练仅作历史参照。
8. 指标：总 rollout 数、生成token数（与检索token分开）、有效组比例、零优势组比例、唯一题覆盖/重复次数、墙钟耗时，以及原200题既有评测协议下的EM、证据覆盖、调用数、缺答案率。不能把更少更新步数当作更少计算。
9. 数值安全与采样效果分开：loss/梯度非有限时，多rank统一阻止更新并保存诊断。不能只在一个rank跳过，也不能认为采样自动解决NaN。若加入保护，对照组同样加入。
10. 该方案是有界优先题目采样，不是完整DAPO Dynamic Sampling。后者通常对当前采样组筛选并补充rollout，计算预算和采样分布不同，不应混称。

验收：在同等资源预算下，观察有效组比例是否提高，并确认EM、证据覆盖、格式有效率和题目覆盖没有明显恶化。当前尚无新策略训练结果。
''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--out-dir', type=Path, default=Path('analysis/training_diagnostics_20261117'))
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summaries, log_rows, group_rows, windows, sources = {}, [], [], [], []
    for name, (log_name, trace_name) in RUNS.items():
        path = a.root / log_name
        rows = parse_log(path)
        assert len([r for r in rows if 'actor/pg_loss' in r]) == 624, name
        ls = log_summary(rows)
        logs = [r for r in rows if 'actor/pg_loss' in r]
        summaries[name] = {k: v for k, v in ls.items() if k not in ('first100', 'last100', 'nan_context')}
        summaries[name]['log_first100'] = window_stats(ls['first100'])
        summaries[name]['log_last100'] = window_stats(ls['last100'])
        groups = []
        if trace_name:
            groups, info = groups_from_trace(a.root / trace_name, name)
            summaries[name]['trace_inventory'] = info
            summaries[name]['trace_all'] = group_stats(groups)
            summaries[name]['trace_first100'] = group_stats([g for g in groups if g['step'] <= 100])
            summaries[name]['trace_last100'] = group_stats([g for g in groups if g['step'] >= 525])
            if name == 'cost':
                first = ls['first_actor_nonfinite_step']
                summaries[name]['trace_pre_nonfinite'] = group_stats([g for g in groups if g['step'] < first])
                summaries[name]['trace_post_nonfinite'] = group_stats([g for g in groups if g['step'] > first])
            group_rows += groups
        for start in range(1, 625, 50):
            stop = min(start + 49, 624)
            win = dict(run=name, first_step=start, last_step=stop,
                       **window_stats([r for r in logs if start <= r['step'] <= stop]))
            gs = [g for g in groups if start <= g['step'] <= stop]
            if gs:
                win.update({k: v for k, v in group_stats(gs).items() if k != 'categories'})
            windows.append(win)
        for r in rows:
            log_rows.append(dict(run=name, **r))
        for p in (path, a.root / trace_name if trace_name else None):
            if p:
                sources.append(dict(run=name, path=str(p), bytes=p.stat().st_size,
                                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        dump(out / f'{name}_nonfinite_context.json', ls['nan_context'])
    csv_write(out / 'log_metrics.csv', log_rows)
    csv_write(out / 'group_diagnostics.csv', group_rows)
    csv_write(out / 'step_windows.csv', windows)
    dump(out / 'summary.json', summaries)
    dump(out / 'source_manifest.json', sources)
    # Candidate priorities use training trajectories only; no validation/test IDs.
    candidates = [r for r in group_rows if r['run'] == 'evidence']
    assert len({r['sample_id'] for r in candidates}) == len(candidates)
    dump(out / 'candidate_training_priorities.json', dict(
        status='proposal_only_not_applied', alpha=.1, mixed_uniform_fraction=.5,
        high_weight=3, default_weight=1, source_run='evidence',
        warning='One group of two rollouts per question at changing policy steps; noisy/stale, not fixed difficulty labels.',
        rows=[{k: r[k] for k in ('sample_id', 'step', 'candidate_weight', 'category')} for r in candidates]))
    write_report(out, summaries, group_rows, windows)
    print(json.dumps(clean({name: dict(
        steps=s['train_steps'], first_nonfinite=s['first_actor_nonfinite_step'],
        zero_advantage_batches=s['zero_advantage_batch_count'],
        entropy_first100=s['log_first100']['actor/entropy_loss'],
        entropy_last100=s['log_last100']['actor/entropy_loss'],
        trace_all=s.get('trace_all'), trace_last100=s.get('trace_last100'),
        validation=s['validation']) for name, s in summaries.items()}), indent=2))


if __name__ == '__main__':
    main()
