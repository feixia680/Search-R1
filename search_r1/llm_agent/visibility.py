"""Track external-token provenance through the existing rolling-context operations.

No policy token IDs, ordering or budgets are changed. Reference labels are not used.
Views are captured immediately before an active row is passed to generation.
"""
import torch


class VisibilityTracker:
    def __init__(self, tokenizer, input_ids, trajectories):
        if not getattr(tokenizer, 'is_fast', False):
            raise ValueError('Visibility tracing requires a fast tokenizer with offsets')
        self.tokenizer, self.trajectories = tokenizer, trajectories
        self.refs = torch.zeros_like(input_ids, dtype=torch.long)
        self.maps = [{} for _ in trajectories]
        self.counters = [0] * len(trajectories)
        self.pending_refs = None
        self.generation = 0
        for trace in trajectories:
            trace['visibility_schema_version'] = 1

    def observations(self, texts, ids, offsets, limit):
        pending = torch.zeros_like(ids, dtype=torch.long)
        for row, (text, trace) in enumerate(zip(texts, self.trajectories)):
            if not trace['events']:
                continue
            event = trace['events'][-1]
            if event['action'] != 'search' or not event.get('attempted') or not text:
                continue
            event['observation_text'] = text
            event['visible_views'] = []
            event['document_body_spans'] = []
            cursor = 0
            for docno, doc in enumerate(event.get('documents', []), 1):
                header = f"Doc {docno}(Title: {doc['title']})"
                pos = text.find(header, cursor)
                if pos < 0:
                    raise ValueError('Retrieved document formatting and body offsets disagree')
                start = pos + len(header)
                if text[start:start + 1] == ' ':
                    start += 1
                end = min(start + len(doc['body']), len(text) - len('</information>\n\n'))
                event['document_body_spans'].append(dict(title=doc['title'], start=start, end=end))
                cursor = end
            count, retained = 0, 0
            for col in range(ids.shape[1]):
                if int(ids[row, col]) == self.tokenizer.pad_token_id:
                    continue
                count += 1
                if col >= limit:
                    continue
                retained += 1
                self.counters[row] += 1
                ref = self.counters[row]
                pending[row, col] = ref
                a, b = map(int, offsets[row, col])
                self.maps[row][ref] = (len(trace['events']) - 1, col, a, b, int(ids[row, col]))
            event['observation_tokens_before_truncation'] = count
            event['observation_tokens_after_truncation'] = retained
            event['observation_truncated'] = retained < count
        self.pending_refs = pending[:, :limit]

    def merge(self, old_ids, responses, observations, new_width):
        # Mirror TensorHelper's stable padding permutation and left context crop.
        self.refs = self.refs[:, -old_ids.shape[1]:]
        combined_ids = torch.cat([old_ids, responses, observations], dim=1)
        refs = torch.cat([self.refs, torch.zeros_like(responses), self.pending_refs], dim=1)
        order = (combined_ids != self.tokenizer.pad_token_id).long().argsort(dim=1, stable=True)
        self.refs = refs.gather(1, order)[:, -new_width:]

    def capture(self, input_ids, active_mask):
        self.refs = self.refs[:, -input_ids.shape[1]:]
        if self.refs.shape != input_ids.shape:
            raise ValueError('Visibility provenance lost alignment with rolling inputs')
        for row, active in enumerate(active_mask):
            if not active:
                continue
            grouped = {}
            for col, ref in enumerate(self.refs[row].tolist()):
                if not ref:
                    continue
                item = self.maps[row][ref]
                if item[4] != int(input_ids[row, col]):
                    raise ValueError('Visible token IDs do not match their external provenance')
                grouped.setdefault(item[0], []).append(item)
            for event_index, items in grouped.items():
                event = self.trajectories[row]['events'][event_index]
                runs = []
                for item in items:
                    if not runs or item[1] != runs[-1][-1][1] + 1:
                        runs.append([])
                    runs[-1].append(item)
                chunks = []
                for run in runs:
                    start, end = min(x[2] for x in run), max(x[3] for x in run)
                    bodies = []
                    for span in event.get('document_body_spans', []):
                        a, b = max(start, span['start']), min(end, span['end'])
                        if a < b:
                            bodies.append(dict(title=span['title'], text=event['observation_text'][a:b]))
                    token_ids = [x[4] for x in run]
                    chunks.append(dict(token_ids=token_ids,
                                       decoded_text=self.tokenizer.decode(token_ids, skip_special_tokens=True),
                                       source_char_start=start, source_char_end=end,
                                       document_bodies=bodies))
                event['visible_views'].append(dict(generation=self.generation,
                    context_tokens=int((input_ids[row] != self.tokenizer.pad_token_id).sum()),
                    visible_observation_tokens=len(items), chunks=chunks))
        self.generation += 1
