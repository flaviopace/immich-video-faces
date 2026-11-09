#!/usr/bin/env bash
# Install on a Raspberry Pi (Raspberry Pi OS / Debian). Run from the repo folder:
#   ./install.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="$(id -un)"
CONFIG="$HOME/.config/immich-video-faces.env"

echo "==> Installing ffmpeg and python3-requests"
sudo apt-get update -qq
sudo apt-get install -y ffmpeg python3-requests

if [ ! -f "$CONFIG" ]; then
  mkdir -p "$(dirname "$CONFIG")"
  cp "$DIR/config.example.env" "$CONFIG"
  chmod 600 "$CONFIG"
  echo "==> Created $CONFIG - edit IMMICH_URL and IMMICH_API_KEY"
fi

echo "==> Installing systemd timer (runs nightly at 02:30 as $RUN_USER)"
sed -e "s|__USER__|$RUN_USER|" -e "s|__DIR__|$DIR|" "$DIR/systemd/immich-video-faces.service" \
  | sudo tee /etc/systemd/system/immich-video-faces.service >/dev/null
sudo cp "$DIR/systemd/immich-video-faces.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now immich-video-faces.timer

echo
echo "Done. Next steps:"
echo "  1. nano $CONFIG"
echo "  2. python3 $DIR/immich_video_faces.py --dry-run     # which videos tonight's run will scan"
echo "  3. python3 $DIR/immich_video_faces.py --max-videos 1 # try one now"
echo "  Logs: journalctl -u immich-video-faces"
