"""Model-assisted task and completion-criteria review of existing labels."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from jsonschema.exceptions import ValidationError

from .selection import BASE, content_hash
from .reference_repairs import review_references, VERSION as REPAIR_VERSION


def reference_catalog(model, baseline, reviews=()):
    """Expose distinct requirement and claim identities without changing judgments."""
    requirements = {r['case_requirement_id']: r['text'] for r in model['case_requirements']}
    targets = ({baseline['assessment_id']} if baseline['mode'] == 'case_requirement'
               else set(requirements) | {model['assessment_target']['target_id']})
    evidence = {r['evidence_id']: r for r in model['evidence_refs']}
    entries = []
    for claim in model['claims']:
        if claim['target_id'] not in targets:
            continue
        status = claim['support_status']
        ids = claim['evidence_ids']
        relevant = [(v, r['status']) for r in reviews if r.get('case_id') == baseline['case_id']
                    for v in r.get('claim_reviews', []) if v['claim_id'] == claim['claim_id']]
        if relevant:
            statuses = {v['support_status'] for v, _ in relevant}
            status = statuses.pop() if len(statuses) == 1 and all(s == 'accepted' for _, s in relevant) else 'unknown'
            ids = sorted({x for v, _ in relevant for x in v['evidence_ids']})
        entries.append(dict(claim_id=claim['claim_id'], target_id=claim['target_id'],
            requirement_text=requirements.get(claim['target_id']), field_path=claim['field_path'],
            support_status=status, reason=claim['reason'],
            usable_for_pass_fail=status == 'supported' and not claim['conflicting_evidence_ids'] and bool(ids),
            evidence=[{k: evidence[eid][k] for k in ('evidence_id', 'event_id', 'source_role', 'quote')}
                      for eid in ids if eid in evidence]))
    return entries


def review_unit(selector, client, label, bundle, baseline, reviews, requested_rules, max_input_chars, repair_context=None):
    model = label.get('model_output', label)
    # Exclude deprecated gate assessments and unrelated historical derived grades.
    fields = ('assessment_target', 'case_requirements', 'task_attributes', 'capability_mapping',
              'feedback_annotations', 'failure_annotations', 'claims', 'evidence_refs', 'limitations')
    # Keep every visible body and the full omitted-ID scope, without repeating empty stubs.
    visible_bundle = ({**bundle, 'events':[e for e in bundle['events'] if e['body_provided']]}
                      if bundle is not None else None)
    payload = dict(case_id=baseline['case_id'], assessment_id=baseline['assessment_id'], mode=baseline['mode'],
                   requested_rules=requested_rules, labels={k: model[k] for k in fields}, evidence_bundle=visible_bundle,
                   claim_reviews=[r for r in reviews if r.get('case_id') == baseline['case_id']],
                   policy_rules=[r for r in selector.policy['rules'] if r['rule_id'] in requested_rules])
    payload['claim_reference_catalog'] = reference_catalog(model, baseline, reviews)
    payload['evidence_reference_catalog'] = [
        {k: ref[k] for k in ('evidence_id', 'event_id', 'source_role', 'quote')}
        for ref in model['evidence_refs']]
    payload['unregistered_evidence_instruction'] = (
        '优先使用evidence_reference_catalog中的evidence_id。需要引用未登记但已提供正文的事件时，'
        '在顶层evidence_requests中提交{event_id:完整原事件编号,quote:连续逐字原文}，'
        '该规则的evidence_ids暂填同一个完整event_id；程序核对后生成证据编号。'
        '每个事件本次最多选一段引文，不改前缀或截短ID；未提供正文或证据不足时保留pending。')
    payload['reference_instructions'] = (
        'claim_ids只能从claim_reference_catalog的claim_id列选择；target_id是需求归属，不能代替claim_id。'
        '编号相同也必须确实出现在claim_id列。pass/fail只能引用usable_for_pass_fail=true的判断；'
        '仍须核对判断及原文是否直接支持当前规则，不得为通过校验选择无关判断。'
        'evidence_ids使用已登记evidence_id；只有按evidence_requests明确申请登记时才填完整event_id。'
        '如没有适当依据，保留pending并说明缺少什么。')
    if repair_context is not None:
        payload['repair'] = repair_context
        payload['claim_evidence_links'] = [{k:c[k] for k in (
            'claim_id','target_id','field_path','support_status','evidence_ids','conflicting_evidence_ids')}
            for c in model['claims']]
    system = (BASE / 'config/selection_review_prompt.md').read_text()
    user = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    log = dict(case_id=baseline['case_id'], assessment_id=baseline['assessment_id'],
               requested_rules=requested_rules, request_sha256=content_hash(dict(system=system, user=user)),
               input_chars=len(system) + len(user), model=getattr(client, 'model', type(client).__name__),
               status='pending', call_attempted=False)
    log['prompt_serialization'] = 'compact-json-visible-events-v1'
    log['repair_attempt'] = repair_context is not None
    log['reference_repair_version'] = REPAIR_VERSION
    if log['input_chars'] > max_input_chars:
        log['error'] = 'input_limit_exceeded: 未截断原文，保持待核验'
        return None, log
    response = None
    try:
        log['call_attempted'] = True
        response, metadata = client.complete_json(system, user)
        log.update(metadata=metadata, cache_dir=str(client.cache_dir.resolve()) if getattr(client, "cache_dir", None) else None)
        if (not isinstance(response, dict) or 'rule_checks' not in response
                or set(response) - {'rule_checks', 'evidence_requests'} or not isinstance(response['rule_checks'], dict)):
            raise ValueError('Expected rule_checks and optional evidence_requests')
        if set(response['rule_checks']) != set(requested_rules):
            raise ValueError('Response rule set differs from requested_rules')
        audit = dict(audit_id='model-' + content_hash([baseline['case_id'], baseline['assessment_id'],
                      baseline['label_sha256'], baseline['policy_sha256'], log['request_sha256']])[:24],
                     case_id=baseline['case_id'], assessment_id=baseline['assessment_id'], mode=baseline['mode'],
                     label_sha256=baseline['label_sha256'], policy_sha256=baseline['policy_sha256'],
                     reviewer_type='model', reviewer_id=log['model'], review_time=datetime.now(timezone.utc).isoformat(),
                     rule_checks=response['rule_checks'])
        selector.validate('selection_audit.schema.json', audit)
        normalized, extra, changes = review_references(response, model, bundle, payload['claim_reference_catalog'])
        log['reference_resolutions'] = changes
        audit['rule_checks'] = normalized['rule_checks']
        if extra:
            audit['additional_evidence_refs'] = extra
        selector.validate('selection_audit.schema.json', audit)
        # Keep raw request identities available for the one repair round; generated
        # evidence IDs are not in the original label's catalog.
        log['_repair_response'] = response
        log['status'] = 'generated'
        return audit, log
    except Exception as exc:
        # Provider/schema failures are persisted; do not turn them into exclusions.
        log.update(status='error', error=f'{type(exc).__name__}: {exc}')
        if response is not None and isinstance(exc, (ValueError, ValidationError, TypeError, KeyError, IndexError)):
            # The caller consumes this retry payload; do not duplicate responses in logs.
            log['_invalid_response'] = response
        return None, log
