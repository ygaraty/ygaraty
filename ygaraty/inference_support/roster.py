"""Bounded contributor reservations and cumulative, SQLite-backed history."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import threading
import time


class ContributorRoster:
    def __init__(self, assignments, timeout=90, clock=time.monotonic, database=None, preparing_timeout=300):
        self.assignments = assignments
        self.timeout, self.clock, self.preparing_timeout = timeout, clock, preparing_timeout
        self.records = {}
        self.lock = threading.RLock()
        self.database = Path(database).resolve() if database is not None else None
        if self.database:
            self.database.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.database)) as db, db:
                db.execute("""CREATE TABLE IF NOT EXISTS contributor_history (
                    id TEXT PRIMARY KEY, slot INTEGER, state TEXT NOT NULL,
                    reason TEXT NOT NULL, assignment TEXT NOT NULL)""")
                # A restarted coordinator cannot resume the previous browser/GPU session.
                db.execute("UPDATE contributor_history SET state='former', reason='coordinator restarted' "
                           "WHERE state IN ('active', 'preparing')")
                for ident, slot, state, reason, assignment in db.execute(
                        "SELECT id, slot, state, reason, assignment FROM contributor_history ORDER BY rowid"):
                    self.records[ident] = dict(id=ident, slot=slot, state=state,
                                              reason=reason, assignment=json.loads(assignment), last_seen=None)

    def _save(self, record):
        if self.database:
            with closing(sqlite3.connect(self.database)) as db, db:
                db.execute("""INSERT INTO contributor_history VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET state=excluded.state, reason=excluded.reason""",
                    (record['id'], record['slot'], record['state'], record['reason'],
                     json.dumps(record['assignment'])))

    def reserve(self, ident, prefer=None, *, strict=False):
        """A free slot for `ident`: `prefer` if it is free (a browser coming back to the stage whose weights
        it has cached), else the first free one."""
        with self.lock:
            if ident in self.records:
                return dict(self.records[ident])
            occupied = {r['slot'] for r in self.records.values() if r['state'] != 'former'}
            free = [i for i in range(len(self.assignments)) if i not in occupied]
            slot = prefer if prefer in free else (None if strict else next(iter(free), None))
            if slot is None:
                return dict(id=ident, slot=None, state='rejected', reason='capacity reached', assignment=None)
            record = dict(id=ident, slot=slot, state='preparing' if slot is not None else 'former',
                          reason='' if slot is not None else 'capacity reached', last_seen=self.clock(),
                          assignment=self.assignments[slot] if slot is not None else None)
            self._save(record)
            self.records[ident] = record
            return dict(record)

    def touch(self, ident):
        with self.lock:
            record = self.records[ident]
            if record['state'] == 'former':
                raise ValueError('Contributor reservation has ended')
            record['last_seen'] = self.clock()

    def activate(self, ident):
        with self.lock:
            self.touch(ident)
            record = dict(self.records[ident], state='active')
            self._save(record)
            self.records[ident] = record

    def retire(self, ident, reason):
        with self.lock:
            record = self.records.get(ident)
            if record and record['state'] != 'former':
                retired = dict(record, state='former', reason=reason)
                self._save(retired)
                self.records[ident] = retired

    def expire(self):
        with self.lock:
            # A preparing browser downloads its stage, maybe for minutes on a slow link, often in a background
            # tab whose timers the browser slows: it gets longer before it counts as gone.
            expired = [dict(r) for r in self.records.values()
                       if r['state'] != 'former' and self.clock()-r['last_seen'] >
                       (self.preparing_timeout if r['state'] == 'preparing' else self.timeout)]
            for record in expired:
                self.retire(record['id'], 'heartbeat expired')
            return expired

    def revive(self, ident, state='preparing'):
        """A contributor retired for missing heartbeats that turns out to be alive gets its slot back, if no
        one has taken it. Returns whether it did."""
        with self.lock:
            record = self.records.get(ident)
            if not record or record['state'] != 'former' or record.get('reason') != 'heartbeat expired':
                return False
            if any(r['slot'] == record['slot'] and r['state'] != 'former' for r in self.records.values()):
                return False
            revived = dict(record, state=state, reason='', last_seen=self.clock())
            self._save(revived)
            self.records[ident] = revived
            return True

    def snapshot(self):
        with self.lock:
            records = [dict(r) for r in self.records.values()]
            return dict(required=len(self.assignments), active=sum(r['state']=='active' for r in records),
                        preparing=sum(r['state']=='preparing' for r in records),
                        former=sum(r['state']=='former' for r in records), contributors=records)
