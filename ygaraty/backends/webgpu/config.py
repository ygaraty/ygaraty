"""ygaraty.toml: saved launch/serve settings, so a restart is one command.

A config file holds option names without dashes, with either '-' or '_':

    model = "s5-nano"
    num-workers = 2
    slots = 1
    route = "relay"
    lan = true

Flags given on the command line override the file. The file is read from
--config FILE, else WORKSPACE/ygaraty.toml when --workspace is given, else
./ygaraty.toml. Relative paths in it are relative to the file's directory.
"""

import tomllib
from pathlib import Path

NAME = "ygaraty.toml"
# Settings worth repeating on a restart, by argparse destination.
SAVED = {
    "launch": (
        "model",
        "contributors",
        "bundle",
        "bundle_url",
        "checkpoint",
        "gguf",
        "context",
        "memory_gb",
        "slots",
        "host",
        "port",
        "route",
        "turn_host",
        "turn_secret_file",
        "lan",
        "public",
        "public_url",
        "open_status",
        "open_chat",
        "no_terminal_chat",
        "no_status_bar",
        "serve_weights",
    ),
    "serve": (
        "model_id",
        "workers",
        "bundle",
        "bundle_url",
        "context",
        "memory_gb",
        "slots",
        "host",
        "port",
        "route",
        "turn_host",
        "turn_secret_file",
        "public_url",
        "open_status",
    ),
}


def find_config(args):
    """The config file to read, or None."""
    if getattr(args, "config", None) is not None:
        if not args.config.is_file():
            raise ValueError(f"Config file not found: {args.config}")
        return args.config
    workspace = getattr(args, "workspace", None)
    if workspace is not None and (Path(workspace) / NAME).is_file():
        return Path(workspace) / NAME
    if Path(NAME).is_file():
        return Path(NAME)
    return None


def option_map(parser):
    """Config key (option name without dashes, '-' or '_') to action."""
    actions = {}
    for action in parser._actions:  # argparse has no public listing
        for option in action.option_strings:
            if option.startswith("--"):
                name = option[2:]
                actions[name] = action
                actions[name.replace("-", "_")] = action
    return actions


def config_defaults(path, parser):
    """Parser defaults from a config file; unknown keys are errors."""
    try:
        data = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from None
    actions = option_map(parser)
    base = Path(path).resolve().parent
    defaults = {}
    for key, value in data.items():
        action = actions.get(key)
        if action is None or action.dest in {"config", "save_config", "help"}:
            raise ValueError(f"{path}: unknown setting {key!r}")
        if value is True and action.const not in (None, True):
            value = action.const  # e.g. lan = true means --lan (auto)
        elif action.type is Path and isinstance(value, str):
            candidate = Path(value).expanduser()
            value = candidate if candidate.is_absolute() else base / candidate
        elif action.type is int and type(value) is not int:
            raise ValueError(f"{path}: {key} must be a whole number")
        if action.choices is not None and value not in action.choices:
            raise ValueError(
                f"{path}: {key} must be one of {', '.join(action.choices)}"
            )
        defaults[action.dest] = value
    return defaults


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# Destinations whose option name differs from the destination.
OPTION = {
    "contributors": "num-workers",
    "workers": "num-workers",
    "model_id": "model",
}


def save_config(args, command):
    """Write the settings that matter for a restart; returns the path."""
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Saved by 'ygaraty {command} --save-config'. Restart with:",
        f"#   ygaraty {command} --workspace {workspace}",
    ]
    for dest in SAVED[command]:
        value = getattr(args, dest, None)
        if value in (None, False):
            continue
        if hasattr(value, "value"):  # enums such as Route
            value = value.value
        name = OPTION.get(dest, dest.replace("_", "-"))
        lines.append(f"{name} = {_toml_value(value)}")
    path = workspace / NAME
    path.write_text("\n".join(lines) + "\n")
    return path
