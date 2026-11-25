"""Launch route selection; TURN secrets stay outside session metadata."""

import os
from pathlib import Path

from ygaraty.backend_contracts.models import Route

from .turn import TurnConfig

ROUTES = {
    "relay": Route.RELAY,
    "webrtc": Route.DIRECT,
    "turn": Route.PRIVATE_TURN,
}


def add_network_arguments(parser):
    parser.add_argument(
        "--route",
        choices=("auto", *ROUTES),
        default="auto",
        help="auto: WebRTC when TURN is configured or links stay on this "
        "computer or network, otherwise coordinator relay",
    )
    parser.add_argument(
        "--turn-host", help="TURN hostname (or YGARATY_TURN_HOST)"
    )
    parser.add_argument(
        "--turn-secret-file",
        type=Path,
        help="Read the coturn shared secret from a local file",
    )


def resolve_network(args, *, local_network=False):
    """Pick the route and TURN configuration.

    WebRTC needs TURN to cross NATs. local_network (launch without public
    sharing): contributors share this machine or network, so direct links
    over host candidates work without TURN, and any pair whose link does not
    open falls back to the coordinator relay.
    """
    route = getattr(args, "route", "auto")
    if isinstance(route, Route):
        return route, getattr(args, "turn_config", None)
    if route == "relay":
        return Route.RELAY, None
    env = dict(os.environ)
    if getattr(args, "turn_host", None):
        env["YGARATY_TURN_HOST"] = args.turn_host
    secret_file = getattr(args, "turn_secret_file", None)
    if secret_file is not None:
        if secret_file.stat().st_size > 4096:
            raise ValueError("TURN secret file is too large")
        env["YGARATY_TURN_SECRET"] = secret_file.read_text().strip()
    host, secret = env.get("YGARATY_TURN_HOST"), env.get("YGARATY_TURN_SECRET")
    if bool(host) != bool(secret):
        raise ValueError(
            "Configure both TURN host and secret, or use --route relay"
        )
    turn = TurnConfig.from_env(env)
    if route == "auto":
        return (Route.DIRECT if turn or local_network else Route.RELAY), turn
    if turn is None and not (route == "webrtc" and local_network):
        raise ValueError(
            (
                "TURN-only links"
                if route == "turn"
                else "WebRTC across networks"
            )
            + " requires --turn-host and --turn-secret-file "
            "or YGARATY_TURN_HOST/YGARATY_TURN_SECRET"
        )
    return ROUTES[route], turn
