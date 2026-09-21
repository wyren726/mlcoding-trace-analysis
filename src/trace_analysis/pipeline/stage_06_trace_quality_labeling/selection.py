"""Evidence-linked Stage06 selection. No model calls and no legacy quality gates."""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from os.path import relpath
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource

BASE = Path(__file__).parent
RULES = ('S1', 'S2', 'S3', 'S4')


def content_hash(value: Any) -> str:
    """Record hash, independent of JSONL whitespace (also used by supplied audits)."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    # JSON strings may legally contain U+2028/U+2029. Only LF separates JSONL records.
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').split('\n'), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: expected JSON object')
            rows.append(row)
    return rows


def unique_index(rows: list[dict], key: str) -> dict:
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f'Missing {key}')
        if value in result:
            raise ValueError(f'Duplicate {key}: {value}')
        result[value] = row
    return result


def check(status='pending', reason='尚无实际核验记录', claims=(), evidence=()):
    return dict(status=status, reason=reason, claim_ids=sorted(set(claims)),
                evidence_ids=sorted(set(evidence)))


def validate_citation(ref, events, provided):
    """Use the same exact source checks for label and supplementary audit evidence."""
    event = events.get(ref['event_id'])
    if not event or not event['body_provided'] or ref['event_id'] not in provided:
        raise ValueError('引用了未提供正文的事件')
    for left, right in [('source_role', 'role'), ('turn_id', 'turn_id'),
                        ('locator', 'locator'), ('episode_membership', 'episode_membership')]:
        if ref[left] != event[right]:
            raise ValueError(f'Evidence {left} mismatch')
    if not ref['quote'] or ref['quote'] not in event['content']:
        raise ValueError('引文不是事件原文中的连续文本')


class Selector:
    def __init__(self, policy_path: Path | None = None):
        self.policy_path = policy_path or BASE / 'config/selection_policy.json'
        self.policy = json.loads(self.policy_path.read_text())
        self.policy_sha256 = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        if self.policy.get('empty_filter_semantics') != 'no_restriction':
            raise ValueError('Unsupported empty filter semantics')
        if self.policy['version'] not in ('stage06-selection-v2', 'stage06-selection-v3'):
            raise ValueError('Unsupported selection policy version')
        self.rules = RULES + (('S5',) if self.policy['version'] == 'stage06-selection-v2' else ())
        if [r['rule_id'] for r in self.policy['rules']] != list(self.rules):
            raise ValueError('Rule set does not match selection policy version')
        if self.policy.get('selection_score_enabled') or self.policy.get('capacity') is not None or self.policy.get('coverage_quotas'):
            raise ValueError('Scoring, capacity and coverage quotas are not implemented')
        with (BASE / 'config/requirement_taxonomy.csv').open(encoding='utf-8-sig', newline='') as f:
            self.taxonomy = {int(r['taxonomy_row']): r for r in csv.DictReader(f)}
        self.taxonomy_sha256 = hashlib.sha256((BASE / 'config/requirement_taxonomy.csv').read_bytes()).hexdigest()
        self.parent_names = {r['requirement_id']: r['requirement_name'] for r in self.taxonomy.values()}
        matching_path = self.policy_path.parent / self.policy['requirement_matching_policy']
        if not matching_path.exists():
            matching_path = BASE / 'config' / self.policy['requirement_matching_policy']
        matching = json.loads(matching_path.read_text())
        self.primary = set(matching['primary_eligible_requirement_ids'])
        self.auxiliary = set(matching['auxiliary_only_requirement_ids'])
        self.schemas = {p.name: json.loads(p.read_text()) for p in (BASE / 'schemas').glob('*.json')}
        registry = Registry().with_resources((s['$id'], Resource.from_contents(s)) for s in self.schemas.values())
        self.validators = {n: Draft202012Validator(s, registry=registry, format_checker=FormatChecker())
                           for n, s in self.schemas.items()}
        filters = self.policy['scope_filters']
        if set(filters) != {'domains', 'work_objects', 'taxonomy_rows'}:
            raise ValueError('Unsupported scope filter')
        attrs = self.schemas['common.schema.json']['$defs']['task_attributes']['properties']
        for key, values in filters.items():
            allowed = (set(self.taxonomy) if key == 'taxonomy_rows' else
                       set(attrs['domain_primary' if key == 'domains' else 'work_object_primary']['enum']) - {None, 'unknown'})
            if not isinstance(values, list) or any(v not in allowed for v in values):
                raise ValueError(f'Invalid scope filter: {key}')

    def validate(self, schema, value):
        self.validators[schema].validate(value)

    def mapping_display(self, mapping, requirements):
        row = self.taxonomy.get(mapping['taxonomy_row'])
        return dict(requirement_name=self.parent_names.get(mapping['parent_requirement_id']),
                    subrequirement_id=row['subrequirement_id'] if row else None,
                    subrequirement_name=row['subrequirement_name'] if row else None,
                    case_requirement_text=requirements[mapping['case_requirement_id']]['text'],
                    taxonomy_sha256=self.taxonomy_sha256)

    def enrich_names(self, model):
        """Add readable fields deterministically without changing judgments or IDs."""
        model = deepcopy(model)
        requirements = unique_index(model['case_requirements'], 'case_requirement_id')
        for mapping in model['capability_mapping']:
            mapping.update(self.mapping_display(mapping, requirements))
        return model

    def _links(self, model, bundle):
        """Check IDs, taxonomy hierarchy, quotations and roles, not semantic truth."""
        from .label_view import build_view, without_view
        if 'review_view' in model and model['review_view'] != build_view(model):
            raise ValueError('review_view differs from source labels/claims/evidence; regenerate display')
        model = without_view(model)
        claims = unique_index(model['claims'], 'claim_id')
        refs = unique_index(model['evidence_refs'], 'evidence_id')
        reqs = unique_index(model['case_requirements'], 'case_requirement_id')
        unique_index(model['capability_mapping'], 'mapping_id')
        unique_index(model['feedback_annotations'], 'feedback_id')
        unique_index(model['failure_annotations'], 'failure_id')
        target = model['assessment_target']
        targets = set(reqs) | {target['target_id']}
        if target['target_id'] in reqs:
            raise ValueError('Episode target_id collides with case_requirement_id')
        core, other = set(target['core_requirement_ids']), set(target['noncore_requirement_ids'])
        if core & other or core | other != set(reqs):
            raise ValueError('Core/noncore requirements must partition case requirements')
        if core != {k for k, r in reqs.items() if r['is_core']}:
            raise ValueError('Requirement is_core disagrees with target scope')

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('claim_ids',):
                        if not set(item) <= claims.keys():
                            raise ValueError('Dangling claim reference')
                    if key in ('evidence_ids', 'conflicting_evidence_ids', 'prior_output_evidence_ids', 'requirement_evidence_ids'):
                        if not set(item) <= refs.keys():
                            raise ValueError('Dangling evidence reference')
                    if key == 'evidence_id' and item not in refs:
                        raise ValueError('Dangling evidence_id')
                    if key == 'target_requirement_ids' and not set(item) <= reqs.keys():
                        raise ValueError('Dangling requirement reference')
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(model)
        for c in claims.values():
            if c['target_id'] not in targets:
                raise ValueError('Claim target outside Episode')
            _field_value(model, c['field_path'])
        for index, m in enumerate(model['capability_mapping']):
            if not any((normalize_path(claims[c]['field_path']) == f'capability_mapping.{index}' or normalize_path(claims[c]['field_path']).startswith(f'capability_mapping.{index}.')) for c in m['claim_ids']):
                raise ValueError('Mapping lacks a claim for its own field')
            if m['case_requirement_id'] not in reqs:
                raise ValueError('Mapping outside requirement scope')
            if any(claims[c]['target_id'] != m['case_requirement_id'] for c in m['claim_ids']):
                raise ValueError('Mapping claim belongs to another requirement')
            row = m['taxonomy_row']
            if row is not None and m['parent_requirement_id'] != self.taxonomy[row]['requirement_id']:
                raise ValueError('Parent/subrequirement mismatch')
            for key, expected in self.mapping_display(m, reqs).items():
                if key in m and m[key] != expected:
                    raise ValueError(f'Mapping display field disagrees with taxonomy/requirement: {key}')
        if bundle is None:
            raise ValueError('缺少原文 evidence bundle，不能核验引文')
        self.validate('evidence.schema.json', bundle)
        if bundle['case_id'] != model['case_id']:
            raise ValueError('Evidence bundle case mismatch')
        events = unique_index(bundle['events'], 'event_id')
        scope = bundle['scope']
        provided, omitted = set(scope['provided_event_ids']), set(scope['omitted_body_event_ids'])
        if provided & omitted or not (provided | omitted) <= events.keys():
            raise ValueError('Invalid evidence scope event IDs')
        if scope['full_episode_read'] and any(not e['body_provided'] for e in events.values() if e['episode_membership'] == 'episode'):
            raise ValueError('Full Episode flag contradicts omitted body')
        for ref in refs.values():
            validate_citation(ref, events, provided)
        return claims, refs, reqs

    def select(self, label: dict, bundle: dict | None, *, audits=(), reviews=(),
               mode='episode', requirement_id=None, run_id='selection', evaluated_at=None):
        if mode not in self.policy['allowed_modes']:
            raise ValueError('Unsupported mode')
        model = label.get('model_output', label)
        case_id = label.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ValueError('Input must identify case_id')
        target = model.get('assessment_target', {})
        assessment_id = requirement_id if mode == 'case_requirement' else target.get('target_id', case_id)
        if not assessment_id:
            raise ValueError('Local selection requires requirement_id')
        digest = content_hash(label)
        diagnostics = []
        rules = {k: check() for k in self.rules}
        pre = dict(valid_label_and_references='pending', scope_match='pending',
                   no_selection_critical_unresolved_conflict='pending')
        result = dict(schema_version='stage06-v1-draft', policy_version=self.policy['version'],
                      selection_run_id=run_id, case_id=case_id, assessment_id=assessment_id, mode=mode,
                      label_sha256=digest, policy_sha256=self.policy_sha256, rule_checks=rules,
                      prechecks=pre, primary_requirement_ids=[], auxiliary_requirement_ids=[],
                      duplicate_of=None, source_session_id=None, source_session_path=None,
                      evaluation=dict(reviewer_type='program', reviewer_id=self.policy['version'],
                                      evaluated_at=evaluated_at or datetime.now(timezone.utc).isoformat(), review_ids=[]))
        if 'input_issue' in label:
            issue = label['input_issue']
            diagnostics.append('输入或处理问题待修正，尚未生成语义标签：'+issue['kind']+'；'+issue['error'])
            return self._finish(result, diagnostics)
        try:
            self.validate('selection_input.schema.json', model)
            if case_id != model['case_id']:
                raise ValueError('Label/model case_id mismatch')
            claims, refs, reqs = self._links(model, bundle)
            if mode == 'case_requirement' and requirement_id not in reqs:
                raise ValueError('Unknown local requirement')
        except (ValueError, KeyError, TypeError, IndexError, ValidationError) as exc:
            # Invalid schema/links are repair work, never evidence of a poor Trace.
            diagnostics.append(f'输入待修正：{exc.message if hasattr(exc, "message") else str(exc)}')
            return self._finish(result, diagnostics)
        pre['valid_label_and_references'] = 'pass'
        scope_targets = {assessment_id} if mode == 'case_requirement' else set(reqs) | {target['target_id']}
        active = {k: c for k, c in claims.items() if c['target_id'] in scope_targets}
        support = {k: c['support_status'] for k, c in claims.items()}
        claim_evidence = {k: set(c['evidence_ids']) for k, c in claims.items()}
        reviewed = set()
        review_values = {}
        invalid_review = False
        if label.get('review_record_paths') and not any(r.get('case_id') == case_id for r in reviews):
            invalid_review = True
            diagnostics.append('label 引用复核记录，但未通过 --reviews 提供；不能忽略既有复核')
        if label.get('review_summary') == 'unresolved':
            invalid_review = True
            diagnostics.append('正式 label 仍标为 unresolved，请先提交解决后的版本')
        for review in reviews:
            if review.get('case_id') != case_id:
                continue
            try:
                self.validate('review.schema.json', review)
                if review['label_sha256'] != digest:
                    raise ValueError('复核 label_sha256 过期')
                cr = unique_index(review['claim_reviews'], 'claim_id')
                if set(cr) != set(review['reviewed_claim_ids']) or not set(cr) <= claims.keys():
                    raise ValueError('复核 claim 集合不一致')
                if not set(review['evidence_ids']) <= refs.keys():
                    raise ValueError('复核 evidence 不存在')
                for k, value in cr.items():
                    if not set(value['evidence_ids']) <= refs.keys():
                        raise ValueError('复核 evidence 不存在')
                    review_values.setdefault(k, []).append((value, review['status']))
                result['evaluation']['review_ids'].append(review['review_id'])
                if review['status'] == 'corrected':
                    raise ValueError('存在未应用的标签修正；提交修正后版本再筛选')
            except (ValueError, KeyError, TypeError, IndexError, ValidationError) as exc:
                invalid_review = True
                diagnostics.append(f'复核记录待修正：{exc.message if hasattr(exc, "message") else str(exc)}')
        for k, values in review_values.items():
            statuses = {v['support_status'] for v, _ in values}
            if len(statuses) != 1 or any(state != 'accepted' for _, state in values):
                support[k] = 'unknown'
            else:
                support[k] = statuses.pop()
                reviewed.add(k)
                claim_evidence[k] = set().union(*(set(v['evidence_ids']) for v, _ in values))

        def usable(ids):
            return bool(ids) and all(k in active and support[k] == 'supported'
                                     and not claims[k]['conflicting_evidence_ids'] and claim_evidence[k]
                                     for k in ids)

        def from_claims(status, reason, ids):
            return check(status, reason, ids, set().union(*(claim_evidence[k] for k in ids)))

        mappings = []
        for m in model['capability_mapping']:
            if m['case_requirement_id'] not in scope_targets or not usable(m['claim_ids']):
                continue
            if m['mapping_status'] in ('supported', 'partial'):
                mappings.append(m)
        result['primary_requirement_ids'] = sorted({m['parent_requirement_id'] for m in mappings} & self.primary)
        result['auxiliary_requirement_ids'] = sorted({m['parent_requirement_id'] for m in mappings} & self.auxiliary)
        primary_children = [m for m in mappings if m['parent_requirement_id'] in self.primary and m['mapping_status'] == 'supported']
        if primary_children:
            rules['S1'] = from_claims('pass', '前33类主需求及具体子需求有据',
                                     {c for m in primary_children for c in m['claim_ids']})
        else:
            rules['S1'] = check(reason='主需求/子需求尚未明确；仅有辅助标签不证明已穷尽主需求，需核验')
        difficulty_claims = [k for k, c in active.items()
                             if normalize_path(c['field_path']) == 'task_attributes.difficulty'
                             and c['target_id'] == assessment_id]
        difficulty = model['task_attributes']['difficulty']
        if mode == 'episode' and usable(difficulty_claims):
            if difficulty in ('medium', 'high'):
                rules['S2'] = from_claims('pass', f'难度 {difficulty} 且有支持证据', difficulty_claims)
            elif difficulty == 'low' and set(difficulty_claims) <= reviewed:
                rules['S2'] = from_claims('fail', '复核确认难度 low，保留为简单题备选', difficulty_claims)
            else:
                rules['S2'] = check(reason='难度未知或 low 尚未复核，不据此排除')
        elif mode == 'case_requirement':
            rules['S2'] = check(reason='局部任务须核验自身难度，不能复制 Episode 难度；需 S2 核验记录')

        # Empty filters mean no restriction. Positive matches need supporting claims.
        filter_states = []
        attrs = model['task_attributes']
        for key, selected in self.policy['scope_filters'].items():
            if not selected:
                continue
            if key == 'taxonomy_rows':
                filter_states.append('pass' if any(m['taxonomy_row'] in selected for m in mappings) else 'pending')
                continue
            if mode == 'case_requirement':
                filter_states.append('pending')  # Episode attributes cannot silently define local scope.
                continue
            names = ('domain_primary', 'domain_secondary') if key == 'domains' else ('work_object_primary', 'work_object_secondary')
            states = []
            for name in names:
                ids = [k for k, c in active.items() if normalize_path(c['field_path']) == f'task_attributes.{name}']
                value = attrs[name]; values = value if isinstance(value, list) else [value]
                if not usable(ids) or None in values or 'unknown' in values:
                    states.append('pending')
                else:
                    states.append('pass' if set(values) & set(selected) else 'fail' if set(ids) <= reviewed else 'pending')
            filter_states.append('pass' if 'pass' in states else 'fail' if all(v == 'fail' for v in states) else 'pending')
        pre['scope_match'] = 'fail' if 'fail' in filter_states else 'pending' if 'pending' in filter_states else 'pass'

        matching_audits = [a for a in audits if a.get('case_id') == case_id
                           and a.get('assessment_id') == assessment_id and a.get('mode') == mode]
        seen_rules = set()
        invalid_audit = False
        for audit in matching_audits:
            try:
                self.validate('selection_audit.schema.json', audit)
                if audit['label_sha256'] != digest or audit['policy_sha256'] != self.policy_sha256:
                    raise ValueError('核验 label/policy hash 不匹配')
                extra = unique_index(audit.get('additional_evidence_refs', []), 'evidence_id')
                if extra.keys() & refs.keys():
                    raise ValueError('Supplementary evidence collides with existing evidence ID')
                events = unique_index(bundle['events'], 'event_id')
                for ref in extra.values():
                    validate_citation(ref, events, set(bundle['scope']['provided_event_ids']))
                audit_refs = {**refs, **extra}
                for rule, value in audit['rule_checks'].items():
                    if rule not in self.rules:
                        raise ValueError(f'{rule} 不属于当前筛选规则')
                    if rule in seen_rules:
                        raise ValueError(f'{rule} 有多条核验记录，请先裁决为一条')
                    seen_rules.add(rule)
                    if not set(value['claim_ids']) <= active.keys() or not set(value['evidence_ids']) <= audit_refs.keys():
                        raise ValueError('核验引用超出当前范围或不存在')
                    if value['status'] != 'pending':
                        if not usable(value['claim_ids']):
                            unusable = [k for k in value['claim_ids'] if not usable([k])]
                            raise ValueError(f'{rule} 通过/不满足的核验须引用有支持的现有 claim；不可用 claim_ids={unusable}')
                        linked = set().union(*(claim_evidence[k] for k in value['claim_ids']))
                        if self.policy['version'] == 'stage06-selection-v2' and not set(value['evidence_ids']) <= linked:
                            unlinked = sorted(set(value['evidence_ids']) - linked)
                            raise ValueError(f'{rule} 核验 evidence 未关联所引 claim；未关联 evidence_ids={unlinked}；可关联 evidence_ids={sorted(linked)}')
                    if rule == 'S1' and value['status'] == 'pass' and not primary_children:
                        raise ValueError('S1 核验不能代替有效主子需求映射')
                    if rule == 'S1' and value['status'] == 'fail' and primary_children:
                        raise ValueError('S1 核验与已有有效映射冲突，先修正标签')
                    if rule == 'S2' and value['status'] != 'pending' and mode == 'episode':
                        expected = ('pass' if difficulty in ('medium', 'high') else 'fail' if difficulty == 'low' else 'pending')
                        if value['status'] != expected or not usable(difficulty_claims) or not set(difficulty_claims) <= set(value['claim_ids']):
                            raise ValueError('S2 核验与已有难度或引用冲突，先修正标签')
                    rules[rule] = value.copy()
                result['evaluation']['review_ids'].append(audit['audit_id'])
            except (ValueError, KeyError, TypeError, IndexError, ValidationError) as exc:
                invalid_audit = True
                diagnostics.append(f'筛选核验待修正：{exc.message if hasattr(exc, "message") else str(exc)}')
        critical = {k for k, c in active.items() if c['is_core']}
        critical.update(k for r in rules.values() for k in r['claim_ids'])
        unresolved = any(support[k] != 'supported' or claims[k]['conflicting_evidence_ids'] for k in critical)
        pre['no_selection_critical_unresolved_conflict'] = 'pending' if unresolved else 'pass'
        if invalid_audit or invalid_review:
            pre['valid_label_and_references'] = 'pending'
        if unresolved:
            # Do not let a disputed label drive a negative decision. Independent
            # verified failures may still exclude the unit per the policy.
            for key, value in rules.items():
                if value['status'] == 'fail' and not usable(value['claim_ids']):
                    rules[key] = check(reason='排除依据存在未决冲突，先核验标签')
        return self._finish(result, diagnostics)

    def _finish(self, result, diagnostics):
        pre, rules = result['prechecks'], result['rule_checks']
        if pre['valid_label_and_references'] != 'pass':
            decision = 'review'
        elif pre['scope_match'] == 'fail' or any(r['status'] == 'fail' for r in rules.values()):
            decision = 'not_selected'
        elif all(v == 'pass' for v in pre.values()) and all(r['status'] == 'pass' for r in rules.values()):
            decision = 'priority_candidate'
        else:
            decision = 'review'
        codes = [f'{k}:{v}' for k, v in pre.items() if v != 'pass']
        codes += [f'{k}:{r["status"]}' for k, r in rules.items() if r['status'] != 'pass']
        result.update(decision=decision, reason_codes=codes or ['all_rules_pass'],
                      reason='；'.join(diagnostics + [f'{k}: {r["reason"]}' for k, r in rules.items()]))
        result['evaluation']['review_ids'] = sorted(set(result['evaluation']['review_ids']))
        self.validate('selection.schema.json', result)
        return result


def normalize_path(path):
    """Accept documented dot/index paths and JSON pointers, without executing text."""
    import re
    path = path.lstrip('/').replace('/', '.')
    return re.sub(r'\[(\d+)\]', r'.\1', path)


def _field_value(model, path):
    value = model
    for part in normalize_path(path).split('.'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def run_selection(labels_path: Path, output_dir: Path, *, evidence_path: Path | None = None,
                  audits_path: Path | None = None, reviews_path: Path | None = None,
                  policy_path: Path | None = None, mode: str | None = None,
                  client=None, max_input_chars: int = 180000, workers: int = 1,
                  sources_path: Path | None = None, input_issues_path: Path | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if max_input_chars <= 0:
        raise ValueError('max_input_chars must be positive')
    if workers <= 0:
        raise ValueError('workers must be positive')
    selector = Selector(policy_path)
    if client is not None and selector.policy['version'] == 'stage06-selection-v2':
        raise ValueError('Historical v2 policy supports validation only; use v3 for model review')
    mode = mode or selector.policy['mode']
    labels = read_jsonl(labels_path)
    valid_label_records = len(labels)
    if input_issues_path is None and (labels_path.parent / 'input_issues.jsonl').exists():
        input_issues_path = labels_path.parent / 'input_issues.jsonl'
    input_issues = read_jsonl(input_issues_path) if input_issues_path else []
    if {r['case_id'] for r in labels} & {r['case_id'] for r in input_issues}:
        raise ValueError('Input issue also has a label; resolve its issue record before selection')
    labels.extend(dict(case_id=r['case_id'], assessment_target=dict(target_id=r['episode_id']),
                       input_issue=dict(kind=r['kind'],error=r['error'])) for r in input_issues)
    if not labels:
        raise ValueError('No labels to select')
    # Identical records are collapsed. Competing versions require explicit resolution.
    cases = {}
    duplicates = 0
    for label in labels:
        key = label.get('case_id')
        if key in cases:
            if content_hash(label) != content_hash(cases[key]):
                raise ValueError(f'Conflicting label versions for {key}')
            duplicates += 1
        cases[key] = label
    bundles = unique_index(read_jsonl(evidence_path), 'case_id') if evidence_path else {}
    audits = read_jsonl(audits_path) if audits_path else []
    reviews = read_jsonl(reviews_path) if reviews_path else []
    if sources_path is None and (labels_path.parent / 'sources.jsonl').exists():
        sources_path = labels_path.parent / 'sources.jsonl'
    sources = unique_index(read_jsonl(sources_path), 'case_id') if sources_path else {}
    for issue in input_issues:
        sources.setdefault(issue['case_id'],issue)
    unique_index(audits, 'audit_id'); unique_index(reviews, 'review_id')
    units = []
    for label in cases.values():
        m = label.get('model_output', label)
        ids = ([r['case_requirement_id'] for r in m.get('case_requirements', [])]
               if mode == 'case_requirement' else [None])
        if not ids:
            raise ValueError('Local mode needs existing case_requirements; use Episode mode to inspect invalid inputs')
        for rid in ids:
            aid = rid if mode == 'case_requirement' else m.get('assessment_target', {}).get('target_id', label.get('case_id'))
            units.append((label, rid, aid))
    keys = {(label.get('case_id'), aid, mode) for label, _, aid in units}
    if any((a.get('case_id'), a.get('assessment_id'), a.get('mode')) not in keys for a in audits):
        raise ValueError('Audit does not correspond to a selected unit')
    if any(r.get('case_id') not in cases for r in reviews):
        raise ValueError('Claim review case not in input')
    now = datetime.now(timezone.utc).isoformat()
    run_id = output_dir.name
    results, model_logs, generated_audits = [], [], []
    # Reserve the new run before any paid request; checkpoint every model response.
    output_dir.mkdir(parents=True, exist_ok=False)
    def append_record(name, record):
        with (output_dir / name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    def evaluate(unit):
        label, rid, aid = unit
        accepted_audit, logs = None, []
        kwargs = dict(reviews=reviews, mode=mode, requirement_id=rid, run_id=run_id, evaluated_at=now)
        bundle = bundles.get(label.get('case_id'))
        result = selector.select(label, bundle, audits=audits, **kwargs)
        covered = {k for a in audits if a.get('case_id') == result['case_id']
                   and a.get('assessment_id') == aid and a.get('mode') == mode for k in a.get('rule_checks', {})}
        requested = [k for k in selector.rules if k in ('S3', 'S4', 'S5')
                     and k not in covered and result['rule_checks'][k]['status'] == 'pending']
        if client is not None and requested and result['decision'] != 'not_selected' and result['prechecks']['valid_label_and_references'] == 'pass':
            from .selection_review import review_unit
            repair_context = None
            for attempt in range(2):
                audit, log = review_unit(selector, client, label, bundle, result, reviews, requested, max_input_chars,
                                         repair_context=repair_context)
                invalid_response = log.pop('_invalid_response', None)
                raw_response = log.pop('_repair_response', None)
                retry_format = invalid_response is not None
                if retry_format:
                    repair_context = dict(previous_response=invalid_response, validation_error=log['error'],
                        instruction='只修正返回结构并重新核对指定规则，rule_checks键必须恰好为requested_rules。证据不足仍保留pending，不以通过校验为理由改变判断。')
                if audit is not None:
                    candidate = selector.select(label, bundle, audits=[*audits, audit], **kwargs)
                    if candidate['prechecks']['valid_label_and_references'] == 'pass':
                        result = candidate
                        accepted_audit = audit
                        log['status'] = 'accepted'
                    else:
                        log.update(status='rejected', error=candidate['reason'])
                        # Report every independently invalid rule so one repair does not
                        # merely expose the next error behind the first failing rule.
                        rule_errors = {}
                        for rule, value in audit['rule_checks'].items():
                            single = {**audit, 'rule_checks': {rule: value}}
                            inspected = selector.select(label, bundle, audits=[*audits, single], **kwargs)
                            if inspected['prechecks']['valid_label_and_references'] != 'pass':
                                rule_errors[rule] = inspected['reason'].partition('；S1:')[0]
                        repair_context = dict(previous_rule_checks=audit['rule_checks'],validation_error=candidate['reason'],
                            rule_validation_errors=rule_errors,
                            previous_response=raw_response,
                            instruction='修正引用后重核指定规则。evidence_id须为同一案例中已验证的原文证据；未登记事件须在evidence_requests中重新提交完整event_id与连续原文，勿复制上次程序生成的证据编号。pass/fail所引claim必须有支持且无冲突。证据不必预先挂在所引claim下，但须直接支持本规则结论，并在reason中说明关联。不得添加不相关claim凑ID；证据不足保留pending。')
                logs.append(log)
                if log['status'] != 'rejected' and not retry_format:
                    break
            if log['status'] != 'accepted':
                result['reason_codes'].append('model_audit:' + log['status'])
                result['reason'] += '；模型核验未生效：' + log.get('error', log['status'])
        selector.validate('selection.schema.json', result)
        return result, accepted_audit, logs

    def record(outcome):
        result, audit, logs = outcome
        if audit is not None:
            generated_audits.append(audit)
            append_record('model_selection_audits.jsonl', audit)
        for log in logs:
            model_logs.append(log)
            append_record('model_review_log.jsonl', log)
        results.append(result)
        append_record('selection_checkpoint.jsonl', result)
    if workers == 1:
        for unit in units:
            record(evaluate(unit))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(evaluate,unit) for unit in units]
            for future in as_completed(futures):
                record(future.result())
    results.sort(key=lambda r: (r['case_id'], r['assessment_id']))
    counts = {k: 0 for k in ('priority_candidate', 'review', 'not_selected')}
    counts.update(Counter(r['decision'] for r in results))
    summary = dict(selection_run_id=run_id, mode=mode, policy_version=selector.policy['version'],
                   policy_sha256=selector.policy_sha256, counts=counts, input_records=len(labels),
                   valid_label_records=valid_label_records,input_issue_records=len(input_issues),
                   duplicate_records_collapsed=duplicates, selected_units=len(results),
                   evaluator='program', workers=workers, model_calls=sum(log['call_attempted'] for log in model_logs),
                   model_audits_accepted=len(generated_audits), model_review_enabled=client is not None, created_at=now,
                   contract_sha256={str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [BASE / 'config/requirement_taxonomy.csv',
                                              selector.policy_path.parent / selector.policy['requirement_matching_policy']]
                                    if p.is_relative_to(BASE)},
                   inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (labels_path, evidence_path, audits_path, reviews_path, sources_path, input_issues_path) if p})
    # Never overwrite a prior experiment or a supplied input.
    def jsonl(name, rows):
        (output_dir / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    jsonl('selection.jsonl', results)
    from .session_export import export_priority_sessions
    summary['session_export'] = export_priority_sessions(labels_path.parent, results, sources, overwrite=True)
    from .distribution_report import write_distribution_report
    summary['distribution_report'] = write_distribution_report(
        labels_path, output_dir, results, sources, selector, summary, labels=cases, reviews=reviews)
    # Pending templates deliberately omit reviewer identity/time; not valid signed audits until filled.
    jsonl('selection_audit_templates.jsonl', [dict(case_id=r['case_id'], assessment_id=r['assessment_id'],
          mode=mode, label_sha256=r['label_sha256'], policy_sha256=r['policy_sha256'],
          rule_checks={k: v for k, v in r['rule_checks'].items() if v['status'] == 'pending'})
          for r in results if any(v['status'] == 'pending' for v in r['rule_checks'].values())])
    (output_dir / 'selection_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (output_dir / 'selection_policy.json').write_bytes(selector.policy_path.read_bytes())
    if client is not None:
        (output_dir / 'selection_review_prompt.md').write_bytes((BASE / 'config/selection_review_prompt.md').read_bytes())
    from .report_paths import review_directory
    report_dir = review_directory(output_dir)
    (report_dir / 'selection_report.md').write_text(
        '# Trace 筛选结果\n\n' + '\n'.join(f'- {k}: {v}' for k, v in counts.items()) +
        f'\n\n由程序按已有标签及核验记录分流；模型核验调用 {summary["model_calls"]} 次（含缓存命中），未新增语义标签。'
        '\n未提供任务要求与完成条件核验的案例不会自动入选。逐条理由见 selection.jsonl；待核验模板不能直接作为已完成的核验记录。\n'
        f'\n[优先候选 session 清单]({relpath(summary["session_export"]["output"], report_dir)})：'
        f'{summary["session_export"]["count"]} 个 session；'
        f'{len(summary["session_export"]["missing_session_case_ids"])} 条优先候选缺少 session ID。\n'
        f'\n[标签质量与需求分布审查报告](<{relpath(summary["distribution_report"]["output"], report_dir)}>)\n')
    return summary
