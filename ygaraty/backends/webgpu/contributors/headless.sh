#!/usr/bin/env bash
# Run Ygaraty contributors as headless Chromium tabs on this machine's GPU.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/headless-contributor.sh CONTRIBUTOR_URL [COUNT]

Starts COUNT (default 1) headless Chromium contributors for the share link
printed by `ygaraty launch` ("Share with contributors: ...") and accepts
automatically. Each contributor uses its own browser profile, as a separate
person would. Runs until Ctrl-C, then stops every browser it started.

Environment:
  CHROMIUM      Browser executable (default: first of chromium,
                chromium-browser, google-chrome-stable, google-chrome)
  PROFILE_ROOT  Parent of per-contributor profiles and logs
                (default: ${XDG_CACHE_HOME:-~/.cache}/ygaraty/contributors)
  YGARATY_BROWSER_DISPLAY
                auto (default), xvfb or headless. auto uses a virtual X
                display (Xvfb) on NVIDIA GPUs when there is no display, and
                headless mode otherwise.

WebGPU uses the real GPU through Vulkan. Without these flags headless Chromium
has no adapter or falls back to SwiftShader on the CPU. Headless mode needs
the Vulkan extension VK_EXT_headless_surface, which Mesa drivers (AMD, Intel)
have and NVIDIA's driver does not: there Chromium runs on an Xvfb display
instead (DO-111: same speed as the native worker; install the xvfb package).
EOF
}

[[ ${1:-} == -h || ${1:-} == --help ]] && { usage; exit 0; }
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
url=$1
count=${2:-1}
[[ $url == http://* || $url == https://* ]] && [[ $url == *'#key='* ]] ||
    { echo "Expected the contributor link with #key=..." >&2; exit 2; }
[[ $count =~ ^[1-9][0-9]?$ ]] || { echo "COUNT must be 1-99" >&2; exit 2; }
[[ $url == *autostart=1* ]] || url+='&autostart=1'
# `ygaraty launch --lan` serves a self-signed certificate on a private
# address: accept it there (and only there) instead of a warning page.
insecure=()
if [[ $url =~ ^https://(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.|localhost[:/]) ]]; then
    insecure=(--ignore-certificate-errors)
fi

browser=${CHROMIUM:-}
if [[ -z $browser ]]; then
    for candidate in chromium chromium-browser google-chrome-stable google-chrome; do
        command -v "$candidate" >/dev/null && { browser=$candidate; break; }
    done
fi
[[ -n $browser ]] || { echo "Chromium not found; set CHROMIUM" >&2; exit 1; }

# Headless mode or a virtual X display (see the usage text).
display_mode=${YGARATY_BROWSER_DISPLAY:-auto}
case $display_mode in
auto)
    display_mode=headless
    if [[ -z ${DISPLAY:-} ]] && command -v nvidia-smi >/dev/null &&
        nvidia-smi -L >/dev/null 2>&1; then
        if command -v Xvfb >/dev/null; then
            display_mode=xvfb
        else
            echo "NVIDIA GPU without a display: headless Chromium would run" \
                "WebGPU on the CPU. Install Xvfb (apt install xvfb)." >&2
        fi
    fi;;
xvfb | headless) ;;
*) echo "YGARATY_BROWSER_DISPLAY must be auto, xvfb or headless" >&2; exit 2;;
esac
mode_flags=(--headless=new)
xvfb_pid=''
if [[ $display_mode == xvfb ]]; then
    command -v Xvfb >/dev/null || { echo "Xvfb not found" >&2; exit 1; }
    number=99
    while [[ -e /tmp/.X$number-lock ]]; do number=$((number + 1)); done
    Xvfb ":$number" -screen 0 1280x800x24 -nolisten tcp >/dev/null 2>&1 &
    xvfb_pid=$!
    export DISPLAY=":$number"
    sleep 1
    mode_flags=(--ozone-platform=x11)
    echo "Using a virtual display ($DISPLAY) for WebGPU on the NVIDIA GPU."
fi

# Chromium refuses to run as root with its sandbox (containers often run as
# root); there the only way is without it.
if [[ $EUID -eq 0 ]]; then
    mode_flags+=(--no-sandbox)
    echo "Running as root: Chromium starts without its sandbox." >&2
fi

root=${PROFILE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/ygaraty/contributors}
mkdir -p "$root"
pids=()
stop() {
    trap - INT TERM EXIT
    [[ ${#pids[@]} -gt 0 ]] && kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    [[ -n $xvfb_pid ]] && kill "$xvfb_pid" 2>/dev/null || true
    echo "Stopped ${#pids[@]} contributor(s)."
}
trap stop EXIT
trap 'stop; exit 0' INT TERM

# Chromium allows one browser per profile: a second one hands off and exits.
# Take the first profile no running browser holds, so several runs can share
# a machine while each profile keeps its verified weight cache for reuse.
in_use() {
    local lock pid
    lock=$(readlink "$1/SingletonLock" 2>/dev/null) || return 1
    pid=${lock##*-}
    [[ $pid =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}
slot=0
for ((i = 1; i <= count; i++)); do
    while :; do
        slot=$((slot + 1))
        profile=$root/worker$slot
        in_use "$profile" || [[ " ${taken:-} " == *" $slot "* ]] || break
    done
    taken="${taken:-} $slot"
    "$browser" "${mode_flags[@]}" \
        --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan \
        --disable-background-timer-throttling --no-first-run \
        --no-default-browser-check \
        --user-data-dir="$profile" "${insecure[@]}" \
        --enable-logging=stderr --v=0 \
        "$url" 2>"$profile.log" &
    pids+=("$!")
    echo "Contributor $i: pid $!, profile worker$slot, log $profile.log"
done
echo "Accepting automatically. Press Ctrl-C to stop."

# Progress comes from the server's status endpoint (contributor scope); it
# covers every contributor of the session, not only the ones started here.
key=${url#*#key=}
key=${key%%&*}
status_url=${url%%#*}
status_url=${status_url%/}/api/status
progress() {
    curl -fsS ${insecure:+-k} --max-time 8 -H "X-Key: $key" "$status_url" 2>/dev/null |
        python3 -c '
import json, sys
try:
    s = json.load(sys.stdin)
except ValueError:
    sys.exit()
roster = s.get("roster", {})
parts = []
for c in roster.get("contributors", []):
    stage = (c.get("assignment") or {}).get("stage", "?")
    p = c.get("progress") or {}
    if c.get("state") == "active":
        parts.append(stage + " ready")
    elif p.get("total"):
        parts.append("%s %.2f/%.2f GB %d%%" % (stage, p["loaded"] / 1e9,
                     p["total"] / 1e9, 100 * p["loaded"] / p["total"]))
    else:
        parts.append(stage + " " + (p.get("phase") or "joining"))
state = s.get("state", "?")
print("Session %s · %s (%d of %s workers joined)" % (
    "ready for chat" if state in ("ready", "running") else state,
    " · ".join(parts) or "no workers yet",
    len(roster.get("contributors", [])), roster.get("required", "?")))
'
}

# Stop as soon as any browser exits; report progress when it changes.
last='' next_report=0 last_shown=0 started=$SECONDS
while :; do
    for index in "${!pids[@]}"; do
        if ! kill -0 "${pids[index]}" 2>/dev/null; then
            echo "Contributor $((index + 1)) (pid ${pids[index]}) exited;" \
                "see its log above." >&2
            exit 1
        fi
    done
    if ((SECONDS >= next_report)); then
        line=$(progress) || line=''
        if [[ -z $line ]]; then
            line="Server not reachable at ${status_url%/api/status}"
        fi
        # Repeat an unchanged line every 30 s so a slow start never looks
        # like a hang; before any worker joins, say what is happening.
        if [[ $line != "$last" ]] || ((SECONDS - last_shown >= 30)); then
            note=''
            if [[ $line == *"no workers yet"* ]]; then
                note=" — browsers still loading the contributor page"
                note+=" ($((SECONDS - started)) s; slow through free tunnels)"
            fi
            echo "$(date +%T) $line$note"
            last_shown=$SECONDS
        fi
        last=$line
        next_report=$((SECONDS + 10))
    fi
    sleep 2
done
