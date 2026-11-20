"""Reference evidence stays in reward metadata, never in policy prompts."""
import re
import unicodedata

SCHEMA_VERSION = 1


def normalize_text(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def question_key(text):
    return normalize_text(text).rstrip('?').strip()


def title_key(text):
    return normalize_text(str(text).strip().strip('"'))


def contains_sentence(text, sentence):
    text, sentence = normalize_text(text), normalize_text(sentence)
    if not sentence:
        return False
    before = r'(?<!\w)' if sentence[0].isalnum() else ''
    after = r'(?!\w)' if sentence[-1].isalnum() else ''
    return re.search(before + re.escape(sentence) + after, text) is not None


def reference_from_raw(row):
    """Reject incomplete/ambiguous annotations instead of silently skipping facts."""
    errors, facts, seen = [], [], set()
    context = {}
    for title, sentences in zip(row['context']['title'], row['context']['sentences']):
        key = title_key(title)
        if key in context:
            errors.append('duplicate_context_title:' + str(title))
        context[key] = list(sentences)
    titles, indices = row['supporting_facts']['title'], row['supporting_facts']['sent_id']
    if len(titles) != len(indices):
        errors.append('supporting_fact_array_length_mismatch')
    for title, index in zip(titles, indices):
        index = int(index)
        identity = (title_key(title), index)
        if identity in seen:
            continue
        seen.add(identity)
        sentences = context.get(identity[0])
        if sentences is None or index < 0 or index >= len(sentences):
            errors.append(f'missing_sentence:{title}:{index}')
            continue
        sentence = str(sentences[index])
        if not normalize_text(sentence):
            errors.append(f'empty_sentence:{title}:{index}')
            continue
        facts.append(dict(fact_id=f'{title}:{index}', title=str(title),
                          sentence_id=index, sentence=sentence))
    if not facts:
        errors.append('no_supporting_facts')
    return dict(schema_version=SCHEMA_VERSION, raw_id=str(row['id']),
                status='invalid_annotation' if errors else 'valid',
                supporting_facts=facts, errors=errors)


def score_evidence(reference, trajectory):
    """Credit only trusted document-body spans present in an actual model input."""
    if not reference or reference.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Evidence reward requires versioned supporting-fact metadata')
    if reference.get('status') != 'valid':
        raise ValueError('Evidence reward refuses invalid supporting-fact annotations')
    facts = list(reference.get('supporting_facts', []))
    if not facts or len({f['fact_id'] for f in facts}) != len(facts):
        raise ValueError('Evidence facts must be nonempty and deduplicated')
    if not trajectory or trajectory.get('visibility_schema_version') != SCHEMA_VERSION:
        raise ValueError('Evidence reward requires token-provenance visibility traces')
    covered, events = set(), []
    for event in trajectory.get('events', []):
        if event.get('action') != 'search' or not event.get('attempted'):
            continue
        hits = set()
        if event.get('success'):
            for view in event.get('visible_views', []):
                for chunk in view.get('chunks', []):
                    decoded = chunk['decoded_text']
                    for body in chunk.get('document_bodies', []):
                        for fact in facts:
                            if (title_key(body['title']) == title_key(fact['title'])
                                    and contains_sentence(body['text'], fact['sentence'])
                                    and contains_sentence(decoded, fact['sentence'])):
                                hits.add(fact['fact_id'])
        added = hits - covered
        covered |= hits
        events.append(dict(turn=event['turn'], matched_fact_ids=sorted(hits),
                           new_fact_ids=sorted(added), evidence_gain=len(added) / len(facts),
                           cumulative_coverage=len(covered) / len(facts)))
    return dict(evidence_eligible=True, evidence_gain=len(covered) / len(facts), evidence_coverage=len(covered) / len(facts),
                evidence_fact_count=len(facts), evidence_hit_count=len(covered),
                evidence_events=events)
