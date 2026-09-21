"""Export priority-candidate session identities from recorded Stage04 provenance."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


def export_priority_sessions(output_dir: Path, results: list[dict], sources: dict[str, dict], *,
                             overwrite: bool = False) -> dict:
    """One row per session, without adding workspace availability to selection gates.

    Only unambiguous recorded paths are exported. Session IDs are never inferred
    from case IDs, path names, or the ordering of unrelated source references.
    """
    from .report_paths import review_directory
    output_dir = review_directory(output_dir)
    rows = {}
    missing = set()
    for result in results:
        if result['decision'] != 'priority_candidate':
            continue
        case_id = result['case_id']
        source = sources.get(case_id, {})
        original = source.get('source_row') or source
        session_ids = source.get('session_ids') or original.get('session_ids') or []
        if not session_ids and result.get('source_session_id'):
            session_ids = [result['source_session_id']]
        if not isinstance(session_ids, list) or any(not isinstance(s, str) or not s.strip() for s in session_ids):
            raise ValueError(f'Invalid session_ids for {case_id}')
        session_ids = sorted(set(session_ids))
        if not session_ids:
            missing.add(case_id)
            continue
        for session_id in session_ids:
            row = rows.setdefault(session_id, dict(case_ids=set(), paths=set()))
            row['case_ids'].add(case_id)
            for ref in original.get('source_references') or []:
                locator = ref.get('source') or {}
                path = locator.get('input_file')
                explicit_id = locator.get('session_id') or ref.get('session_id')
                matches = explicit_id == session_id if explicit_id else len(session_ids) == 1
                if matches and isinstance(path, str) and path.endswith('/_internal/session.jsonl'):
                    row['paths'].add(path)
            if len(session_ids) == 1 and result.get('source_session_path'):
                row['paths'].add(result['source_session_path'])
    exported = []
    unresolved_paths = []
    for session_id, row in sorted(rows.items()):
        path = next(iter(row['paths'])) if len(row['paths']) == 1 else None
        if path is None:
            unresolved_paths.append(session_id)
        exported.append(dict(session_id=session_id, session_jsonl_path=path,
                             workspace_path=str(Path(path).parent.parent)
                             if path and path.endswith('/_internal/session.jsonl') else None,
                             case_ids=sorted(row['case_ids'])))

    # Match the established naming convention when the input is one known batch/run.
    identities = set()
    for case_id in {r['case_id'] for r in results}:
        source = sources.get(case_id, {})
        original = source.get('source_row') or source
        identities.add((original.get('batch_id'), original.get('analysis_run_id')))
    suffix = ''
    if len(identities) == 1:
        identity = next(iter(identities))
        if all(isinstance(v, str) and re.fullmatch(r'[\w.-]+', v) for v in identity):
            suffix = '__' + '__'.join(identity)
    path = output_dir / f'session_paths__priority_candidate{suffix}.jsonl'
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in exported)
    if overwrite:
        # The label-run directory holds the latest completed selection's list.
        # Replace atomically so readers never see a partially written JSONL.
        with NamedTemporaryFile('w', encoding='utf-8', dir=output_dir,
                                prefix='.session-export-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
    return dict(output=str(path), count=len(exported),
                sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(),
                status='incomplete' if missing else 'complete',
                missing_session_case_ids=sorted(missing),
                unresolved_path_session_ids=unresolved_paths)
