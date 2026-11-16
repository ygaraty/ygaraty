"""Small trusted-user host: scoped credentials and bounded volatile records."""

import secrets
import time

from ygaraty.backend_contracts.access import AccessGrant, Action
from ygaraty.backend_contracts.persistence import StoreConflict


class SystemClock:
    def unix_seconds(self):
        return int(time.time())

    def monotonic_seconds(self):
        return time.monotonic()


class LocalAccess:
    """One host owner; contributor credentials never mint generation grants."""

    def __init__(self, spec, owner_key, contributor_key, slots, clock):
        if (
            min(len(owner_key), len(contributor_key)) < 32
            or owner_key == contributor_key
        ):
            raise ValueError(
                "Supply distinct owner/contributor keys of 32+ characters"
            )
        self.spec, self.slots, self.clock = spec, slots, clock
        self._owner, self._contributor = owner_key, contributor_key
        self.version = 1

    def is_owner(self, token):
        return secrets.compare_digest(token.encode(), self._owner.encode())

    def is_contributor(self, token):
        return secrets.compare_digest(
            token.encode(), self._contributor.encode()
        )

    def bind_contributor(self, engine):
        """Install the contributor credential in the compatibility engine."""
        engine.key = self._contributor

    def grant(self, token):
        if not self.is_owner(token):
            raise PermissionError("Invalid owner credential")
        return AccessGrant(
            self.spec.session_id,
            self.spec.epoch,
            "owner",
            "owner",
            self.version,
            self.clock.unix_seconds() + 3600,
            frozenset({Action.GENERATE, Action.CANCEL, Action.MANAGE}),
            self.slots,
            self.slots,
        )

    async def validate(self, grant, *, session_id, epoch, action):
        if (session_id, epoch) != (self.spec.session_id, self.spec.epoch):
            raise PermissionError("Wrong host session")
        grant.require(
            session_id=session_id,
            epoch=epoch,
            action=action,
            now=self.clock.unix_seconds(),
        )
        if (grant.principal_id, grant.grant_id, grant.version) != (
            "owner",
            "owner",
            self.version,
        ):
            raise PermissionError("Grant has been revoked")


class MemoryLedger:
    """Volatile request records with bounded retention; not a billing log."""

    def __init__(self, spec, capacity=4096):
        self.spec, self.capacity = spec, capacity
        self.records = {}

    async def admit(self, record):
        if (
            (record.key.session_id, record.key.epoch)
            != (self.spec.session_id, self.spec.epoch)
            or record.outcome is not None
            or record.key in self.records
        ):
            raise StoreConflict("Stale, terminal or duplicate admission")
        if len(self.records) >= self.capacity:
            finished = next(
                (k for k, r in self.records.items() if r.outcome is not None),
                None,
            )
            if finished is None:
                raise StoreConflict("Request record capacity reached")
            del self.records[finished]
        self.records[record.key] = record

    async def finish(self, key, outcome):
        if key not in self.records:
            raise StoreConflict("Unknown request")
        record = self.records[key].finish(outcome)
        self.records[key] = record
        return record
