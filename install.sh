#!/usr/bin/env bash
# Install on a Raspberry Pi (Raspberry Pi OS / Debian). Run from the repo folder:
#   ./install.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="$(id -un)"
CONFIG="$HOME/.config/immich-video-faces.env"

# Block accidental commits of keys/config from this clone
git -C "$DIR" config core.hooksPath .githooks 2>/dev/null || true

echo "==> Installing ffmpeg, python3-requests and python3-opencv"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ffmpeg python3-requests python3-opencv python3-numpy

if [ ! -f "$CONFIG" ]; then
  mkdir -p "$(dirname "$CONFIG")"
  cp "$DIR/config.example.env" "$CONFIG"
  chmod 600 "$CONFIG"
  echo "==> Created $CONFIG - edit IMMICH_URL, IMMICH_API_KEY and the Telegram settings"
fi

echo "==> Installing systemd timers (scan 02:30, Telegram report 08:00, as $RUN_USER)"
for unit in immich-video-faces immich-video-faces-report; do
  sed -e "s|__USER__|$RUN_USER|" -e "s|__DIR__|$DIR|" "$DIR/systemd/$unit.service" \
    | sudo tee "/etc/systemd/system/$unit.service" >/dev/null
  sudo cp "$DIR/systemd/$unit.timer" /etc/systemd/system/
done
sudo systemctl daemon-reload
sudo systemctl enable --now immich-video-faces.timer immich-video-faces-report.timer

echo
echo "Done. Next steps:"
echo "  1. nano $CONFIG"
echo "  2. python3 $DIR/immich_video_faces.py --dry-run          # which videos tonight's run will scan"
echo "  3. python3 $DIR/immich_video_faces.py --max-videos 1 -v  # try one now"
echo "  4. python3 $DIR/immich_video_faces.py --report           # send the Telegram report now"
echo "  Logs: journalctl -u immich-video-faces"
