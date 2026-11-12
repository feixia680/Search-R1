"""Per-trajectory search accounting; no GPU dependencies."""
import math
import unicodedata
import uuid


def normalize_query(query):
    # Deliberately conservative: preserve punctuation and word order.
    return ' '.join(unicodedata.normalize('NFKC', query).casefold().split())


def new_trajectory():
    return dict(trajectory_id=uuid.uuid4().hex, events=[], final_answer=None,
                search_actions=0, retrieval_attempts=0, retrieval_successes=0,
                retrieval_failures=0, duplicate_queries=0, invalid_actions=0, termination_reason=None)


def record_action(trace, action, content, generated_text, outcome=None, do_search=True):
    event = dict(turn=len(trace['events']), action=action or 'invalid',
                 generated_text=generated_text)
    if action not in ('search', 'answer') or not str(content or '').strip():
        trace['invalid_actions'] = trace.get('invalid_actions', 0) + 1
    if action == 'answer':
        trace['final_answer'] = content
        trace['termination_reason'] = 'answer'
    elif action == 'search':
        trace['search_actions'] += 1
        event.update(query=content, normalized_query=normalize_query(content),
                     attempted=bool(do_search), success=False, duplicate=False)
        if do_search:
            previous = {e['normalized_query'] for e in trace['events']
                        if e.get('attempted')}
            event['duplicate'] = event['normalized_query'] in previous
            trace['duplicate_queries'] += int(event['duplicate'])
            trace['retrieval_attempts'] += 1
            if outcome is None:
                raise ValueError('An attempted search must have a retrieval outcome')
            event.update(outcome)
            trace['retrieval_successes'] += int(outcome['success'])
            trace['retrieval_failures'] += int(not outcome['success'])
        else:
            event['error'] = 'search_budget_exhausted'
            trace['termination_reason'] = 'search_budget_exhausted'
    trace['events'].append(event)


def reward_components(em, trace, beta=0., gamma=0., alpha=0., delta=0.,
                      evidence_gain=0., require_final_answer=False, max_search_calls=None):
    for value in (beta, gamma, alpha, delta):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Search reward coefficients must be finite and nonnegative')
    if trace is None and (beta or gamma or alpha or delta or require_final_answer):
        raise ValueError('Nonzero search penalties require trajectory accounting')
    trace = trace or {}
    # Charge dispatched calls, including failures. A failed call is not free.
    search_cost = beta * trace.get('retrieval_attempts', 0)
    repetition_cost = gamma * trace.get('duplicate_queries', 0)
    if not math.isfinite(evidence_gain) or not 0 <= evidence_gain <= 1:
        raise ValueError('Evidence gain must be a finite coverage fraction')
    invalid_cost = delta * trace.get('invalid_actions', 0)
    evidence_reward = alpha * evidence_gain
    total = float(em) + evidence_reward - search_cost - repetition_cost - invalid_cost
    failure_guard_cost = 0.
    if require_final_answer:
        if max_search_calls is None or max_search_calls < 0:
            raise ValueError('Final-answer guard requires an explicit search budget')
        if trace.get('retrieval_attempts', 0) > max_search_calls:
            raise ValueError('Trajectory exceeds the configured search budget')
        if trace.get('invalid_actions', 0) > max_search_calls + 1:
            raise ValueError('Trajectory exceeds the configured action budget')
        if not str(trace.get('final_answer') or '').strip():
            # Below the worst valid-answer trajectory at this bounded budget.
            floor = -1. - beta * max_search_calls - gamma * max_search_calls - delta * (max_search_calls + 1)
            failure_guard_cost = max(0., total - floor)
            total = min(total, floor)
    return dict(em=float(em), search_cost=search_cost, repetition_cost=repetition_cost,
                evidence_gain=evidence_gain, evidence_reward=evidence_reward,
                invalid_cost=invalid_cost, failure_guard_cost=failure_guard_cost,
                total_reward=total)
