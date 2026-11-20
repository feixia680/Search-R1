"""Compact candidate/selected behavior diagnostics; never modifies rewards."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import warnings
import math


def validate_monitor_options(window_size=10, warmup_steps=100,
                 min_second_search_rate=.05, min_incomplete_evidence_rate=.5,
                 max_missing_answer_rate=.2, max_budget_blocked_rate=.1,
                 min_target_trajectories=16, stop_on_alert=False):
    for name, value, minimum in [('window_size', window_size, 1), ('warmup_steps', warmup_steps, 0),
                                  ('min_target_trajectories', min_target_trajectories, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('Invalid behavior monitor '+name)
    for value in (min_second_search_rate, min_incomplete_evidence_rate,
                  max_missing_answer_rate, max_budget_blocked_rate):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Behavior monitor thresholds must be finite and in [0,1]')
    if not isinstance(stop_on_alert, bool): raise ValueError('stop_on_alert must be boolean')


def compact_trace(trace, group_id, stratum='unobserved'):
    events = trace.get('events', [])
    calls = [event for event in events if event.get('attempted')]
    second_turn = calls[1].get('turn') if len(calls) > 1 else None
    second_gain = sum(float(e.get('evidence_gain', 0)) for e in trace.get('evidence_events', [])
                      if second_turn is not None and e.get('turn') == second_turn)
    return dict(group_id=str(group_id), stratum=str(stratum),
                calls=int(trace.get('retrieval_attempts', len(calls))),
                second_new=bool(second_gain > 0),
                missing=not str(trace.get('final_answer') or '').strip(),
                blocked=any(e.get('error') == 'search_budget_exhausted' for e in events),
                reward=float(trace.get('total_reward', 0)),
                coverage=float(trace.get('evidence_coverage', 0)),
                eligible=bool(trace.get('evidence_eligible', False)),
                duplicates=int(trace.get('duplicate_queries', 0)),
                actions=tuple(str(e.get('action', 'invalid')) for e in events))


def summarize(rows, epsilon=1e-8):
    if not rows: return dict(trajectories=0, groups=0)
    groups = defaultdict(list)
    for row in rows: groups[row['group_id']].append(row)
    comparable = [group for group in groups.values() if len(group) >= 2]
    valid = [r for r in rows if r['eligible']]
    n, second = len(rows), sum(r['calls'] >= 2 for r in rows)
    return dict(trajectories=n, groups=len(groups),
                mean_search_calls=sum(r['calls'] for r in rows)/n,
                second_search_rate=second/n,
                second_new_evidence_rate=sum(r['second_new'] for r in rows)/max(1, second),
                missing_answer_rate=sum(r['missing'] for r in rows)/n,
                budget_blocked_rate=sum(r['blocked'] for r in rows)/n,
                evidence_eligible_count=len(valid),
                evidence_coverage=sum(r['coverage'] for r in valid)/max(1, len(valid)),
                incomplete_evidence_rate=sum(r['coverage'] < 1. - epsilon for r in valid)/max(1, len(valid)),
                duplicate_queries=sum(r['duplicates'] for r in rows),
                reward_mean=sum(r['reward'] for r in rows)/n,
                zero_reward_group_rate=sum(max(r['reward'] for r in g)-min(r['reward'] for r in g) <= epsilon
                                           for g in comparable)/max(1, len(comparable)),
                within_group_reward_std=sum(statistics.stdev(r['reward'] for r in g) for g in comparable)/max(1, len(comparable)),
                identical_tool_sequence_group_rate=sum(len({r['actions'] for r in g}) == 1 for g in comparable)/max(1, len(comparable)))


def summarize_scopes(candidate_rows, selected_rows):
    out = {}
    for name, rows in [('candidate', candidate_rows), ('selected', selected_rows)]:
        roles = defaultdict(list)
        for row in rows: roles[row['stratum']].append(row)
        out[name] = dict(overall=summarize(rows), by_stratum={role: summarize(rs) for role, rs in sorted(roles.items())})
    return out


def flatten(report):
    out = {}
    for scope in ('candidate', 'selected'):
        for key, value in report[scope]['overall'].items(): out[f'behavior/{scope}/{key}'] = value
        for role, metrics in report[scope]['by_stratum'].items():
            for key, value in metrics.items(): out[f'behavior/{scope}/{role}/{key}'] = value
    return out


class BehaviorMonitor:
    """Monitor rolling candidate behavior on historically difficult training strata.

    Low tool use alone is not an alert: incomplete current evidence is also
    required. No automatic stop unless explicitly configured.
    """
    def __init__(self, directory, window_size=10, warmup_steps=100,
                 min_second_search_rate=.05, min_incomplete_evidence_rate=.5,
                 max_missing_answer_rate=.2, max_budget_blocked_rate=.1,
                 min_target_trajectories=16, stop_on_alert=False):
        validate_monitor_options(window_size, warmup_steps, min_second_search_rate,
            min_incomplete_evidence_rate, max_missing_answer_rate, max_budget_blocked_rate,
            min_target_trajectories, stop_on_alert)
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.window_size, self.warmup_steps = window_size, warmup_steps
        self.min_second, self.min_incomplete = min_second_search_rate, min_incomplete_evidence_rate
        self.max_missing, self.max_blocked = max_missing_answer_rate, max_budget_blocked_rate
        self.min_count, self.stop_on_alert = min_target_trajectories, stop_on_alert
        self.history = []

    def record(self, step, report):
        target = report['candidate']['by_stratum'].get('first_insufficient', {})
        self.history.append(dict(step=int(step), target=target))
        self.history = self.history[-self.window_size:]
        count = sum(row['target'].get('trajectories', 0) for row in self.history)
        eligible = sum(row['target'].get('evidence_eligible_count', 0) for row in self.history)
        rolling = {}
        for key in ('second_search_rate', 'missing_answer_rate', 'budget_blocked_rate'):
            rolling[key] = sum(row['target'].get(key, 0)*row['target'].get('trajectories', 0)
                               for row in self.history)/max(1, count)
        rolling['incomplete_evidence_rate'] = sum(row['target'].get('incomplete_evidence_rate', 0)*row['target'].get('evidence_eligible_count', 0)
                                                 for row in self.history)/max(1, eligible)
        alerts = []
        ready = step > self.warmup_steps and len(self.history) == self.window_size and count >= self.min_count
        if ready:
            if eligible >= self.min_count and rolling['second_search_rate'] < self.min_second and rolling['incomplete_evidence_rate'] >= self.min_incomplete:
                alerts.append('SECOND_SEARCH_UNDERUSE_WITH_INCOMPLETE_EVIDENCE')
            if rolling['missing_answer_rate'] > self.max_missing: alerts.append('MISSING_FINAL_ANSWER')
            if rolling['budget_blocked_rate'] > self.max_blocked: alerts.append('SEARCH_BUDGET_TRUNCATION')
        record = dict(global_step=int(step), **report, health=dict(ready=ready, target_trajectories=count,
                       rolling=rolling, alerts=alerts, stop_on_alert=self.stop_on_alert))
        with (self.directory/'behavior_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True)+'\n')
        state = dict(global_step=int(step), history=self.history, health=record['health'])
        temp = self.directory/'behavior_health.tmp'
        temp.write_text(json.dumps(state, allow_nan=False, sort_keys=True)+'\n')
        temp.replace(self.directory/'behavior_health.json')
        for alert in alerts:
            warnings.warn(f'BEHAVIOR_WARNING step={step}: {alert}', RuntimeWarning)
        metrics = {'behavior/health/alert_count': len(alerts), 'behavior/health/ready': int(ready)}
        metrics.update({f'behavior/health/rolling_{k}': v for k, v in rolling.items()})
        return metrics, alerts
