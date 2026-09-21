"""Byte-offset Turn index; load only an authoritative Episode and its lookahead."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from ..stage_02_analyze.runner import TraceBundle, _real_user_event
from .runner import _episode_payload


class TurnSourceIndex:
    VERSION = 1

    def __init__(self, source: Path, cache_dir: Path):
        self.source = source.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(str(self.source).encode()).hexdigest()
        self.path = cache_dir / (key + '.sqlite3')
        stat = self.source.stat()
        self.signature = dict(version=self.VERSION, source=str(self.source),
                              size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metadata = self._metadata()
            if metadata.get('signature') != self.signature:
                self._build()
                metadata = self._metadata()
        self.metadata = metadata

    def _metadata(self):
        if not self.path.exists():
            return {}
        with sqlite3.connect(self.path) as db:
            return json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])

    def _build(self):
        temporary = self.path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        count = 0
        seen, current = set(), None
        try:
            with sqlite3.connect(temporary) as db, self.source.open('rb') as stream:
                db.execute('CREATE TABLE turns (turn_id TEXT PRIMARY KEY, trace_id TEXT, '
                           'batch_id TEXT, turn_index INTEGER, physical_line INTEGER, '
                           'record_index INTEGER, offset INTEGER, length INTEGER, real_user INTEGER)')
                physical_line = 0
                while True:
                    offset = stream.tell()
                    raw = stream.readline()
                    if not raw:
                        break
                    physical_line += 1
                    digest.update(raw)
                    if not raw.decode('utf-8-sig').strip():
                        continue
                    turn = json.loads(raw.decode('utf-8-sig'))
                    trace = turn['trace_id']
                    if trace != current:
                        if trace in seen:
                            raise ValueError(f'Noncontiguous trace in source: {trace}')
                        seen.add(trace)
                        current = trace
                    db.execute('INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?)', (
                        turn['turn_id'], trace, turn['batch_id'], int(turn.get('turn_index') or 0),
                        physical_line, count, offset, len(raw), int(_real_user_event(turn) is not None)))
                    count += 1
                stat = self.source.stat()
                if (stat.st_size, stat.st_mtime_ns) != (self.signature['size'], self.signature['mtime_ns']):
                    raise ValueError('Source changed while building index')
                db.execute('CREATE INDEX trace_order ON turns(trace_id, turn_index, turn_id)')
                db.execute('CREATE TABLE metadata (value TEXT)')
                db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(dict(
                    signature=self.signature, source_sha256=digest.hexdigest(),
                    turn_count=count, trace_count=len(seen))),))
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _records(self, row):
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            records = list(db.execute('SELECT * FROM turns WHERE trace_id=? ORDER BY turn_index, turn_id',
                                      (row['trace_id'],)))
        positions = {r['turn_id']: i for i, r in enumerate(records)}
        boundary = row.get('episode_boundary') or {}
        start, end = (positions.get(boundary.get(k)) for k in ('start_turn_id', 'end_turn_id'))
        if start is None or end is None or end < start:
            raise ValueError('Missing or invalid authoritative Episode boundaries')
        if any(r['batch_id'] != row['batch_id'] for r in records):
            raise ValueError('Source batch does not match Stage04')
        query = row.get('query_turn_id')
        if query and (query not in positions or not start <= positions[query] <= end):
            raise ValueError('Stage04 query turn is outside its Episode boundary')
        selected = records[start:end + 1]
        post = next((r for r in records[end + 1:] if r['real_user']), None)
        if post is not None:
            selected.append(post)
        return selected

    def validate_boundary(self, row):
        records = self._records(row)
        return dict(episode_id=row['episode_id'], trace_id=row['trace_id'],
                    start_turn_id=row['episode_boundary']['start_turn_id'],
                    end_turn_id=row['episode_boundary']['end_turn_id'],
                    source_turns_with_lookahead=len(records))

    def load_episode(self, row):
        stat = self.source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.signature['size'], self.signature['mtime_ns']):
            raise ValueError('Source changed after indexing')
        records = self._records(row)
        turns, locations = [], {}
        with self.source.open('rb') as stream:
            for record in records:
                stream.seek(record['offset'])
                turn = json.loads(stream.read(record['length']).decode('utf-8-sig'))
                if turn['turn_id'] != record['turn_id'] or turn['trace_id'] != row['trace_id']:
                    raise ValueError('Source index identity mismatch')
                turns.append(turn)
                locations[(str(self.source), turn['turn_id'])] = dict(
                    source_file=str(self.source), record_index=record['record_index'],
                    physical_line=record['physical_line'])
        bundle = TraceBundle(batch_id=row['batch_id'], trace_id=row['trace_id'], turns=tuple(turns))
        return _episode_payload(bundle, row), locations
