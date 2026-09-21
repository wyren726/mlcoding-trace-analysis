"""Explicit reuse of unchanged v2 rule judgments, with their original provenance."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json

from .selection import RULES, Selector, content_hash, read_jsonl, unique_index


def migrate_v2_audits(labels_path: Path, evidence_path: Path, audit_paths: list[Path],
                     source_policy_path: Path, output_path: Path, *,
                     reviews_path: Path | None = None, policy_path: Path | None = None) -> dict:
    """Reuse only accepted audits for byte-equivalent S1–S4 rule definitions.

    This is a policy projection, not a new model/human review. Original identity,
    review time, statuses, reasons, claims and evidence are preserved. S5 is dropped.
    Rejected audits must instead be reviewed against the new evidence policy.
    """
    if output_path.exists():
        raise FileExistsError(output_path)
    old, new = Selector(source_policy_path), Selector(policy_path)
    if (old.policy['version'], new.policy['version']) != ('stage06-selection-v2', 'stage06-selection-v3'):
        raise ValueError('Only the explicit v2 to v3 migration is supported')
    for key in ('scope_filters', 'prechecks', 'requirement_matching_policy', 'allowed_modes'):
        if old.policy[key] != new.policy[key]:
            raise ValueError(f'Changed policy contract requires new review: {key}')
    old_rules = {r['rule_id']: r for r in old.policy['rules']}
    if any(old_rules[r['rule_id']] != r for r in new.policy['rules']):
        raise ValueError('Changed rule definition requires new review')
    if old.primary != new.primary or old.auxiliary != new.auxiliary:
        raise ValueError('Changed requirement eligibility requires new review')
    labels = unique_index(read_jsonl(labels_path), 'case_id')
    bundles = unique_index(read_jsonl(evidence_path), 'case_id')
    reviews = read_jsonl(reviews_path) if reviews_path else []
    source = [(path, audit) for path in audit_paths for audit in read_jsonl(path)]
    unique_index([audit for _, audit in source], 'audit_id')
    grouped = {}
    for path, audit in source:
        key = (audit['case_id'], audit['assessment_id'], audit['mode'])
        grouped.setdefault(key, []).append((path, audit))
    migrated = []
    now = datetime.now(timezone.utc).isoformat()
    for (case_id, assessment_id, mode), group in grouped.items():
        label, bundle = labels[case_id], bundles[case_id]
        kwargs = dict(mode=mode, requirement_id=assessment_id if mode == 'case_requirement' else None,
                      reviews=reviews)
        before = old.select(label, bundle, audits=[a for _, a in group], **kwargs)
        if before['prechecks']['valid_label_and_references'] != 'pass':
            raise ValueError(f'Cannot migrate invalid source audit: {case_id}: {before["reason"]}')
        projected = []
        for path, audit in group:
            checks = {k: deepcopy(v) for k, v in audit['rule_checks'].items() if k in RULES}
            if not checks:
                continue
            value = deepcopy(audit)
            value.update(audit_id='migrated-' + content_hash([audit, new.policy_sha256])[:24],
                         policy_sha256=new.policy_sha256, rule_checks=checks,
                         policy_migration=dict(method='unchanged_rules_v2_to_v3', actor_type='program',
                             migrated_at=now, source_audit_id=audit['audit_id'],
                             source_audit_sha256=content_hash(audit),
                             source_policy_sha256=old.policy_sha256,
                             source_audit_file=str(path.resolve()), preserved_rule_ids=list(checks)))
            new.validate('selection_audit.schema.json', value)
            projected.append(value)
        after = new.select(label, bundle, audits=projected, **kwargs)
        if after['prechecks']['valid_label_and_references'] != 'pass':
            raise ValueError(f'Migrated audit failed new validation: {case_id}: {after["reason"]}')
        migrated.extend(projected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in migrated))
    return dict(source_audits=len(source), migrated_audits=len(migrated),
                source_policy_sha256=old.policy_sha256, policy_sha256=new.policy_sha256)
