#!/usr/bin/env bash
# Public self-host installer. Works from a checkout or via curl | bash.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: install.sh --num-workers N [launch options]
       install.sh --install-only

Install the public server and prepare/launch a WebGPU chat coordinator.
Requires Python 3.12+ with venv/pip, Git for installation, and OpenSSH for sharing.
Public Spaces presets: s5-nano (2 x 3 GiB), qwen3.5-9b (2 x 4 GiB, default).
Prepared artifacts are downloaded and verified; no local checkpoint is needed.
Use --bundle-url for another public bundle or --checkpoint for local preparation.
Links work on this machine by default; add --public for a temporary public link.
Launch options include --workspace, --model, --open-chat, --host and --port.

Environment:
  YGARATY_INSTALL_DIR  Installation directory (default: ~/.local/share/ygaraty)
  YGARATY_REF          Public Git branch/tag/commit (default: main)
  YGARATY_WORKSPACE_ROOT  Model cache parent (default: INSTALL_DIR/workspaces)
  YGARATY_SOURCE_DIR   Install an existing public checkout instead of downloading
  PYTHON              Python executable (default: python3)

Reruns reuse the installed environment. Use a different installation directory
for another version. No sudo, shell profile edits, or background service setup.
HELP
}
fail() { printf 'ygaraty: %s\n' "$*" >&2; exit 1; }

main() {
    case "${1:-}" in
        -h|--help) usage; return ;;
    esac
    local install_only=false
    if [[ "${1:-}" == --install-only ]]; then
        [[ $# == 1 ]] || fail '--install-only takes no other arguments'
        install_only=true
    fi
    if [[ "$install_only" == false ]]; then
        local has_workers=false argument
        for argument in "$@"; do
            case "$argument" in
                --num-workers|--num-workers=*|--workers|--workers=*|--contributors|--contributors=*) has_workers=true ;;
            esac
        done
        [[ "$has_workers" == true ]] || fail '--num-workers is required (for example --num-workers 2)'
    fi
    local root=${YGARATY_INSTALL_DIR:-"$HOME/.local/share/ygaraty"}
    local python=${PYTHON:-python3}
    local ref=${YGARATY_REF:-main}
    [[ "$root" == /* ]] || fail 'YGARATY_INSTALL_DIR must be an absolute path'
    local executable="$root/venv/bin/ygaraty"
    if [[ ! -f "$root/.installed" ]]; then
        command -v "$python" >/dev/null || fail 'Install Python 3.12+ or set PYTHON'
        "$python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12+ required")'
        local source=${YGARATY_SOURCE_DIR:-}
        if [[ -z "$source" ]]; then
            command -v git >/dev/null || fail 'Git is required to download the public source'
        else
            source=$(cd -- "$source" && pwd)
            [[ -f "$source/ygaraty/backends/webgpu/cli.py" && -f "$source/pyproject.toml" ]] || fail 'YGARATY_SOURCE_DIR must be the public repository root'
        fi
        mkdir -p -- "$(dirname -- "$root")"
        mkdir -- "$root" || fail 'Installation directory already exists without a completion marker; inspect it and choose another YGARATY_INSTALL_DIR'
        if [[ -z "$source" ]]; then
            source="$root/source"
            git init -q "$source"
            git -C "$source" remote add origin https://github.com/ygaraty/ygaraty.git
            git -C "$source" fetch --depth 1 origin "$ref"
            git -C "$source" checkout -q --detach FETCH_HEAD
        fi
        "$python" -m venv "$root/venv"
        "$root/venv/bin/python" -m pip install \
            "$source[server,prepare]"
        "$executable" --help >/dev/null
        printf '%s\n' "$ref" > "$root/.installed"
    fi
    [[ -x "$executable" ]] || fail 'Installation is incomplete: missing venv/bin/ygaraty'
    printf 'Ygaraty installed: %s\n' "$executable"
    if [[ "$install_only" == true ]]; then
        printf 'Run: %q launch --num-workers 2 --workspace %q\n' "$executable" "$root/workspaces/qwen3.5-9b"
        return
    fi
    export YGARATY_WORKSPACE_ROOT=${YGARATY_WORKSPACE_ROOT:-"$root/workspaces"}
    exec "$executable" launch "$@"
}

# Keep execution last so a truncated download cannot run a partial installer.
main "$@"
