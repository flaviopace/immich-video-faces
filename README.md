# immich-video-faces

Tags the people who appear in your [Immich](https://immich.app) videos.

Immich only looks for faces in a video's **thumbnail**, so most people in a
video are never recognised. This script runs at night on a Raspberry Pi and,
for a few videos per night:

1. grabs 24 frames from the video with ffmpeg (streamed from Immich; the full video isn't downloaded)
2. **checks them for faces on the Pi** (OpenCV, about 0.15 s per frame) and keeps up to 6 frames that contain a face, spread across the video
3. uploads only those frames to Immich as **archived** photos in the album *Video face scan (temp)*
4. waits for Immich to detect and recognise the faces in those frames
5. adds every recognised person to the video (the video then shows up on that person's page)
6. deletes the frames permanently, skipping the trash, then deletes the temporary album at the end of the run

A video with nobody in it causes **no writes at all** on the Immich server.
Frames are kept in the Pi's RAM (`/dev/shm`), so the SD card isn't worn either.

At 08:00 a **Telegram report** arrives: videos processed, frames checked and uploaded,
faces detected, people added (with names), failures, and how many videos are left.

Each scanned video is recorded in a small SQLite database, so a video is never
scanned twice, and new videos are picked up on the next night. New videos
go first (newest first). Videos that failed are retried later, up to 3 times.

## What gets tagged

* Only people **already known** in Immich are added to a video. Because the
  frames are archived, Immich matches them to existing people and never creates
  new people from them. This keeps your People page clean.
* Each person is added once per video. People Immich already found in the thumbnail aren't duplicated.
* Faces Immich can't match to a known person are ignored.
* Only your own videos are scanned. Videos shared by a partner and the video part of live/motion photos are skipped.

## Requirements

* Immich **v1.135 or newer** (v2.x / v3.x recommended), with facial recognition enabled
* Raspberry Pi OS / Debian 12 or newer, Python 3.9+, `ffmpeg`, `python3-requests`, `python3-opencv`

### API key

In Immich go to *Account Settings → API Keys → New API Key*. Give the key all
permissions, or at least these:

`asset.read`, `asset.view`, `asset.download`, `asset.upload`, `asset.update`,
`asset.delete`, `album.read`, `album.create`, `album.delete`, `albumAsset.create`,
`face.read`, `face.create`, `person.read`, `user.read`, and `job.read` (recommended).

`job.read` only works for an **admin** user. It lets the script see when
Immich has finished processing the frames. Without it, the script waits until
the results stop changing for 3 minutes, which is slower but still works.

### Telegram

Create a bot with [@BotFather](https://t.me/BotFather) (or reuse one you already have) and
put its token and your chat id in the config file:

```
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789
```

To find your chat id, send your bot a message, then open
`https://api.telegram.org/bot<TOKEN>/getUpdates` and look for `"chat":{"id":...}`.

## Install on the Raspberry Pi

```bash
git clone https://github.com/flaviopace/immich-video-faces.git
cd immich-video-faces
./install.sh
nano ~/.config/immich-video-faces.env   # Immich URL + API key, Telegram token + chat id
```

`install.sh` installs `ffmpeg`, `python3-requests` and `python3-opencv`, and creates the config
file. It also enables two systemd timers, both running as the user who ran the installer:
the scan **every night at 02:30**, and the Telegram report **every morning at 08:00**.

Try it before the first night:

```bash
python3 immich_video_faces.py --dry-run          # which videos the next run will scan
python3 immich_video_faces.py --max-videos 1 -v  # scan one video now, verbose
python3 immich_video_faces.py --report --dry-run # print the report instead of sending it
python3 immich_video_faces.py --status           # progress so far
```

Logs: `journalctl -u immich-video-faces`. Run now via systemd: `sudo systemctl start immich-video-faces`.

To change the times, edit `OnCalendar=` in `/etc/systemd/system/immich-video-faces.timer`
or `immich-video-faces-report.timer`, then run `sudo systemctl daemon-reload`.

## Settings

All settings go in `~/.config/immich-video-faces.env` (see [`config.example.env`](config.example.env)):

| Setting | Default | |
|---|---|---|
| `IMMICH_URL` | – | e.g. `http://your-immich-server:2283` |
| `IMMICH_API_KEY` | – | |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | – | morning report; without them the report is only printed |
| `MAX_VIDEOS` | `3` | videos per night |
| `STOP_AFTER` | – | `HH:MM`; no new video is started after this time |
| `CANDIDATE_FRAMES` | `24` | frames checked on the Pi per video |
| `FRAMES_PER_VIDEO` | `6` | max frames with faces uploaded per video |
| `FACE_FILTER` | `auto` | `auto` checks frames on the Pi; `off` uploads without checking |
| `MIN_FACE_PX` | `48` | smaller faces are ignored (too small to recognise) |
| `FRAME_MAX_SIZE` | `1440` | longest side of uploaded frames |
| `FACE_WAIT_TIMEOUT` | `900` | max seconds to wait for Immich per video |
| `MAX_ATTEMPTS` | `3` | failed videos are retried this many times |
| `ALBUM_NAME` | `Video face scan (temp)` | where frames live while being processed |
| `STATE_DB` | `~/.local/share/immich-video-faces/state.db` | the "already scanned" list |

## Command line

| Option | |
|---|---|
| `--dry-run` | list the videos the next run would scan, change nothing |
| `--status` | scanned / failed / remaining counts |
| `--report` | send the summary since the last report to Telegram (with `--dry-run`: print it) |
| `--video-id ID` | scan one specific video now, even if it was already scanned |
| `--max-videos N`, `--frames N` | override the settings for this run |
| `--retry-empty` | scan again the videos where no face was found (e.g. after naming new people in Immich) |
| `--retry-failed` | scan again the videos that failed `MAX_ATTEMPTS` times |
| `--forget ID` | mark one video as not scanned |
| `--keep-frames` | leave the frames and album in Immich so you can inspect them |
| `--env-file PATH` | use another config file |
| `-v` | verbose logs |

## Notes

* **No new people from videos.** If someone only appears in videos, name them on
  a photo first, or use `--retry-empty` later.
* **Safe if interrupted.** Every uploaded frame is recorded before anything
  else happens. If a run dies or the network drops, the next run deletes the
  leftover frames first. The album is deleted only once all its frames are gone.
  Only one run can happen at a time.
* **Light on the Immich server's disk.** Reading video doesn't wear an SSD; the only writes are
  the few frames with faces (≈ 150–300 KB each, plus Immich's thumbnails of them), and they are deleted again.
  ffmpeg reads Immich's transcoded copy when there is one, seeking with HTTP range
  requests, so only a few MB are read per frame. If streaming fails, the original is downloaded
  to a temp folder on the Pi instead.
* **Face check models.** The Pi uses [YuNet](models/README.md) (MIT licence), about 300 KB.
  OpenCV 4.6 (Debian 12) uses the 2022 model, and OpenCV 4.8+ uses the 2023 model; the script picks the right one.
* The API key is passed to ffmpeg on its command line, so other users on the
  Pi could see it with `ps`. This is fine on a single-user Pi.
* If Immich's machine learning is down during a run, a video can be marked as
  "no faces". Run `--retry-empty` to scan those again.

## Development

Secrets never belong in the repo: your URL, API key and Telegram token live only in
`~/.config/immich-video-faces.env`, outside the repo. A commit hook blocks API
keys, tokens, private keys, LAN IPs, `.env` and `.db` files. Turn it on once per clone
(`install.sh` does this for you):

```bash
git config core.hooksPath .githooks
```

```bash
python3 -m venv .venv && .venv/bin/pip install requests opencv-python-headless numpy
.venv/bin/python -m unittest discover -s tests -v
```

The tests include end-to-end runs against a fake Immich server (needs `ffmpeg`).
