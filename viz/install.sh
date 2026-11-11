#!/usr/bin/env bash
# Install the visualizer as systemd user services (no sudo): the web server, and a timer that rebuilds the data.
#   viz/install.sh [port]        default port 8080
#   viz/install.sh --remove      stop and remove them
# Needs `loginctl enable-linger $USER` (once, may need sudo) so the services run while you are logged out.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
U="$HOME/.config/systemd/user"
if [ "${1:-}" = "--remove" ]; then
    systemctl --user disable --now tribe-viz.service tribe-viz-build.timer 2>/dev/null || true
    rm -f "$U/tribe-viz.service" "$U/tribe-viz-build.service" "$U/tribe-viz-build.timer"
    systemctl --user daemon-reload
    echo "removed"; exit 0
fi
PORT=${1:-8080}
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "no $PY: create the virtualenv first (see viz/README.md)" >&2; exit 1; }
mkdir -p "$U"
cat > "$U/tribe-viz.service" <<UNIT
[Unit]
Description=tribe run visualizer (web server)
[Service]
WorkingDirectory=$ROOT
ExecStart=$PY $ROOT/viz/serve.py --port $PORT
Restart=on-failure
RestartSec=5
[Install]
WantedBy=default.target
UNIT
cat > "$U/tribe-viz-build.service" <<UNIT
[Unit]
Description=tribe run visualizer (record new models from the results repo)
[Service]
Type=oneshot
WorkingDirectory=$ROOT
ExecStart=$PY $ROOT/viz/build.py
Nice=10
UNIT
cat > "$U/tribe-viz-build.timer" <<UNIT
[Unit]
Description=tribe run visualizer: look for new models every 30 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=30min
[Install]
WantedBy=timers.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now tribe-viz.service tribe-viz-build.timer
systemctl --user restart tribe-viz.service
sleep 1
echo "serving on port $PORT: http://$(curl -s -m 3 ifconfig.me || hostname -I | cut -d' ' -f1):$PORT/"
echo "login (user:password): $(cat "$HOME/.config/tribe-viz/auth" 2>/dev/null || echo 'created on first start, in ~/.config/tribe-viz/auth')"
