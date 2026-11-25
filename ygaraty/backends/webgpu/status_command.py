"""`ygaraty status [URL]`: a running server's state, workers and statistics.

Exit codes: 0 ready, 3 reachable but not ready (waiting for workers,
draining or closed), 1 error (unreachable, key refused), 2 usage error.
"""

import json
import os
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

NOT_READY = 3


def add_parser(commands):
    parser = commands.add_parser(
        "status", help="Show a running server's state, workers and stats"
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="http://127.0.0.1:8765",
        help="Server origin (default: http://127.0.0.1:8765)",
    )
    keys = parser.add_mutually_exclusive_group()
    keys.add_argument("--key", help="Owner or contributor key")
    keys.add_argument(
        "--key-file", type=Path, help="File holding the owner key"
    )
    keys.add_argument(
        "--workspace",
        type=Path,
        help="Read WORKSPACE/keys/owner.key (the launch or serve workspace)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Accept a self-signed certificate (servers started with --lan)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print /v1/status as JSON"
    )
    return parser


def resolve_key(args):
    """--key, --key-file, --workspace, then $YGARATY_KEY; None for an open
    status page."""
    if args.key:
        return args.key
    if args.key_file:
        return args.key_file.read_text().strip()
    if args.workspace:
        return (args.workspace / "keys" / "owner.key").read_text().strip()
    return os.environ.get("YGARATY_KEY") or None


def fetch_status(url, key=None, insecure=False, opener=urlopen):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Give the server origin, e.g. http://127.0.0.1:8765")
    request = Request(
        url.rstrip("/") + "/v1/status",
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    options = {}
    if insecure and parsed.scheme == "https":
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        options["context"] = context
    try:
        with opener(request, timeout=10, **options) as response:
            return json.loads(response.read())
    except HTTPError as error:
        if error.code == 401:
            raise ValueError(
                "The server needs the owner or contributor key: --key, "
                "--key-file, --workspace or YGARATY_KEY"
            ) from None
        raise ValueError(f"HTTP {error.code} from {url}") from None
    except URLError as error:
        raise ValueError(f"Server not reachable: {error.reason}") from None


def run(args, opener=urlopen):
    from .terminal_chat import stats_text

    status = fetch_status(
        args.url, resolve_key(args), args.insecure, opener=opener
    )
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(stats_text(status))
    return 0 if status.get("state") == "ready" else NOT_READY
