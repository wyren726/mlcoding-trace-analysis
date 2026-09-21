"""Mechanical reference repairs; never infer semantic support or rewrite judgments."""
from copy import deepcopy
import re

from .selection import content_hash, unique_index

VERSION = 'verified-reference-repairs-v1'


def requirement_paths(value):
    """Resolve an explicit requirement ID only when its claim target agrees."""
    value = deepcopy(value)
    changes = []
    requirements = value.get('case_requirements', [])
    if not isinstance(requirements, list):
        return value, changes
    indices = {}
    for i, requirement in enumerate(requirements):
        if isinstance(requirement, dict):
            indices.setdefault(requirement.get('case_requirement_id'), []).append(i)
    for claim in value.get('claims', []):
        if not isinstance(claim, dict) or not isinstance(claim.get('field_path'), str):
            continue
        path = claim['field_path']
        match = re.fullmatch(r'case_requirements\[([^\[\]]+)\](\.[A-Za-z_][A-Za-z_0-9]*)?', path)
        if not match or match[1].isdigit():
            continue
        req_id = match[1]
        candidates = indices.get(req_id, [])
        if len(candidates) != 1 or claim.get('target_id') != req_id:
            continue
        i = candidates[0]
        suffix = match[2] or ''
        if suffix and suffix[1:] not in requirements[i]:
            continue
        resolved = f'case_requirements[{i}]' + suffix
        claim['field_path'] = resolved
        changes.append(dict(operation='requirement_id_to_index', claim_id=claim.get('claim_id'),
                            previous=path, resolved=resolved, requirement_id=req_id))
    return value, changes


def review_references(response, model, bundle, catalog):
    """Register explicit exact citations and resolve only unique existing identities."""
    response = deepcopy(response)
    changes = []
    refs = unique_index(model['evidence_refs'], 'evidence_id')
    extra = []
    requests = response.pop('evidence_requests', [])
    if not isinstance(requests, list):
        raise ValueError('evidence_requests must be a list')
    events = unique_index(bundle['events'], 'event_id') if bundle else {}
    for request in requests:
        if not isinstance(request, dict) or set(request) != {'event_id', 'quote'}:
            raise ValueError('Evidence request requires exactly event_id and quote')
        eid, quote = request['event_id'], request['quote']
        if not isinstance(eid, str) or not isinstance(quote, str) or not quote:
            raise ValueError('Evidence request needs a nonempty exact quote and event_id')
        event = events.get(eid)
        if (not event or not event['body_provided'] or eid not in bundle['scope']['provided_event_ids']
                or quote not in event['content']):
            raise ValueError(f'Evidence request is not exact provided text: {eid}')
        if not any(eid in check.get('evidence_ids', []) for check in response['rule_checks'].values()):
            raise ValueError(f'Unused evidence request: {eid}')
        matches = [ref for ref in refs.values() if ref['event_id'] == eid and ref['quote'] == quote]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous existing evidence: {eid}; select an evidence_id')
        if matches:
            ref = matches[0]
        else:
            ref = dict(event_id=eid, quote=quote, turn_id=event['turn_id'], source_role=event['role'],
                       locator=deepcopy(event['locator']), episode_membership=event['episode_membership'])
            ref['evidence_id'] = 'ev_registered_' + content_hash([model['case_id'], ref])[:24]
            if ref['evidence_id'] in refs:
                raise ValueError('Generated evidence ID collision')
            refs[ref['evidence_id']] = ref
            extra.append(ref)
        # Multiple quotes for the same event cannot select a unique evidence ID.
        previous = [c for c in changes if c['operation'] == 'register_exact_evidence' and c['previous'] == eid]
        if previous and previous[0]['resolved'] != ref['evidence_id']:
            raise ValueError(f'Ambiguous evidence requests for event: {eid}')
        if eid in refs and eid != ref['evidence_id']:
            raise ValueError(f'Event ID collides with evidence ID: {eid}')
        changes.append(dict(operation='register_exact_evidence', previous=eid, resolved=ref['evidence_id']))
    requested = {c['previous']: c['resolved'] for c in changes}
    claim_ids = {c['claim_id'] for c in model['claims']}
    for rule, check in response['rule_checks'].items():
        for i, identifier in enumerate(check.get('claim_ids', [])):
            if identifier in claim_ids:
                continue
            matches = [c for c in catalog if c['target_id'] == identifier and c['usable_for_pass_fail']]
            if len(matches) == 1:
                replacement = matches[0]['claim_id']
                check['claim_ids'][i] = replacement
                changes.append(dict(operation='requirement_id_to_claim', rule=rule,
                                    previous=identifier, resolved=replacement))
        for i, identifier in enumerate(check.get('evidence_ids', [])):
            if identifier in refs:
                continue
            replacement = requested.get(identifier)
            if replacement is None:
                # Only exact event IDs; do not guess prefixes or search similar text.
                matches = [r for r in refs.values() if r['event_id'] == identifier]
                replacement = matches[0]['evidence_id'] if len(matches) == 1 else None
            if replacement is not None:
                check['evidence_ids'][i] = replacement
                changes.append(dict(operation='event_id_to_evidence', rule=rule,
                                    previous=identifier, resolved=replacement))
    return response, extra, changes
