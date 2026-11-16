"""Short-lived coturn REST credentials, independent of the cloud application.

Deploy coturn with use-auth-secret and the same static-auth-secret as
YGARATY_TURN_SECRET. Passwords are base64(HMAC-SHA1(secret, username)), where
username is "<expiry>:<session>:<slot>". Keep the shared secret server-side.
This helper does not authorize callers; the session host must do so first.
"""

import base64
import hashlib
import hmac
import os
import re
import time
from dataclasses import dataclass, field

HOST = re.compile(r"^[A-Za-z0-9.-]{1,253}$")


@dataclass(frozen=True)
class TurnConfig:
    host: str
    secret: str = field(repr=False)  # kept out of reprs and logs
    ttl: int = 7200
    port: int = 3478
    tls_port: int | None = 5349

    def __post_init__(self):
        if not HOST.fullmatch(self.host):
            raise ValueError(
                "YGARATY_TURN_HOST must be a host name or address"
            )
        for port in (self.port, self.tls_port):
            if port is not None and (
                type(port) is not int or not 1 <= port <= 65535
            ):
                raise ValueError("TURN ports must be between 1 and 65535")
        if len(self.secret) < 32:
            raise ValueError(
                "YGARATY_TURN_SECRET must be at least 32 characters"
            )
        if not 60 <= self.ttl <= 12 * 3600:
            raise ValueError("YGARATY_TURN_TTL must be between 60 s and 12 h")

    @classmethod
    def from_env(cls, env=None):
        """Load the deployment settings, or None if host/secret are absent."""
        env = os.environ if env is None else env
        host, secret = (
            env.get("YGARATY_TURN_HOST", ""),
            env.get("YGARATY_TURN_SECRET", ""),
        )
        if not host or not secret:
            return None
        tls = env.get("YGARATY_TURN_TLS_PORT", "5349")
        return cls(
            host=host,
            secret=secret,
            ttl=int(env.get("YGARATY_TURN_TTL", "7200")),
            port=int(env.get("YGARATY_TURN_PORT", "3478")),
            tls_port=int(tls) if tls else None,
        )

    def credentials(self, session, slot, now=None):
        """Issue STUN and UDP/TCP/TLS TURN entries for one contributor."""
        expiry = int(now if now is not None else time.time()) + self.ttl
        username = f"{expiry}:{session}:{int(slot)}"
        password = base64.b64encode(
            hmac.new(
                self.secret.encode(), username.encode(), hashlib.sha1
            ).digest()
        ).decode()
        urls = [
            f"turn:{self.host}:{self.port}?transport=udp",
            f"turn:{self.host}:{self.port}?transport=tcp",
        ]
        if self.tls_port:
            urls.append(f"turns:{self.host}:{self.tls_port}?transport=tcp")
        return dict(
            expires=expiry,
            ice=[
                {"urls": f"stun:{self.host}:{self.port}"},
                {"urls": urls, "username": username, "credential": password},
            ],
        )

    def urls(self):
        """The servers without credentials, for reports."""
        return [
            entry["urls"] for entry in self.credentials("x", 0, now=0)["ice"]
        ]
