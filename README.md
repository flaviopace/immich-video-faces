# immich-video-faces

Tags the people who appear in your [Immich](https://immich.app) videos.

Immich only looks for faces in a video's **thumbnail**, so most people in a
video are never recognised. This script runs at night on a Raspberry Pi and,
for a few videos per night:

1. grabs 8 frames from the video with ffmpeg (streamed from Immich; the full video isn't downloaded)
2. uploads them to Immich as **archived** photos in the album *Video face scan (temp)*
3. waits for Immich to detect and recognise the faces in those frames
4. adds every recognised person to the video (the video then shows up on that person's page)
5. deletes the frames permanently, skipping the trash

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
* Raspberry Pi OS (Bullseye or newer), Python 3.9+, `ffmpeg`, `python3-requests`

### API key

In Immich go to *Account Settings → API Keys → New API Key*. Give the key all
permissions, or at least these:

`asset.read`, `asset.view`, `asset.download`, `asset.upload`, `asset.update`,
`asset.delete`, `album.read`, `album.create`, `albumAsset.create`, `face.read`,
`face.create`, `person.read`, `user.read`, and `job.read` (recommended).

`job.read` only works for an **admin** user. It lets the script see when
Immich has finished processing the frames. Without it, the script waits until
the results stop changing for 3 minutes, which is slower but still works.

## Install on the Raspberry Pi

```bash
git clone https://github.com/flaviopace/immich-video-faces.git
cd immich-video-faces
./install.sh
nano ~/.config/immich-video-faces.env   # set IMMICH_URL and IMMICH_API_KEY
```

`install.sh` installs `ffmpeg` and `python3-requests` and creates the config
file. It also enables a systemd timer that runs the scan **every night at 02:30**,
as the user who ran the installer.

Try it before the first night:

```bash
python3 immich_video_faces.py --dry-run         # which videos the next run will scan
python3 immich_video_faces.py --max-videos 1 -v # scan one video now, verbose
python3 immich_video_faces.py --status          # progress so far
```

Logs: `journalctl -u immich-video-faces`. Run now via systemd: `sudo systemctl start immich-video-faces`.

To change the time, edit `OnCalendar=` in `/etc/systemd/system/immich-video-faces.timer`,
then run `sudo systemctl daemon-reload`.

<details>
<summary>Prefer cron?</summary>

Skip the timer and add this with `crontab -e`:

```
30 2 * * * /usr/bin/python3 /home/pi/immich-video-faces/immich_video_faces.py >> /home/pi/immich-video-faces.log 2>&1
```
</details>

## Settings

All settings go in `~/.config/immich-video-faces.env` (see [`config.example.env`](config.example.env)):

| Setting | Default | |
|---|---|---|
| `IMMICH_URL` | – | e.g. `http://your-immich-server:2283` |
| `IMMICH_API_KEY` | – | |
| `MAX_VIDEOS` | `3` | videos per night |
| `FRAMES_PER_VIDEO` | `8` | more frames = more people found, but slower |
| `FRAME_MAX_WIDTH` | `1920` | frames are scaled down to this width |
| `STOP_AFTER` | – | `HH:MM`; no new video is started after this time |
| `FACE_WAIT_TIMEOUT` | `900` | max seconds to wait for Immich per video |
| `MAX_ATTEMPTS` | `3` | failed videos are retried this many times |
| `ALBUM_NAME` | `Video face scan (temp)` | where frames live while being processed |
| `STATE_DB` | `~/.local/share/immich-video-faces/state.db` | the "already scanned" list |

## Command line

| Option | |
|---|---|
| `--dry-run` | list the videos the next run would scan, change nothing |
| `--status` | scanned / failed / remaining counts |
| `--video-id ID` | scan one specific video now, even if it was already scanned |
| `--max-videos N`, `--frames N` | override the settings for this run |
| `--retry-empty` | scan again the videos where no face was found (e.g. after naming new people in Immich) |
| `--retry-failed` | scan again the videos that failed `MAX_ATTEMPTS` times |
| `--forget ID` | mark one video as not scanned |
| `--keep-frames` | leave the frames in Immich so you can inspect them |
| `--env-file PATH` | use another config file |
| `-v` | verbose logs |

## Notes

* **No new people from videos.** If someone only appears in videos, name them on
  a photo first, or use `--retry-empty` later.
* **Safe if interrupted.** Every uploaded frame is recorded before anything
  else happens. If a run dies or the network drops, the next run deletes the
  leftover frames first. Only one run can happen at a time.
* **Video streaming.** ffmpeg reads Immich's transcoded copy of the video when
  there is one, which is smaller and already converted from HDR. It
  seeks with HTTP range requests, so only a few MB are read per frame. If
  streaming fails, the original is downloaded to a temp folder instead.
* The API key is passed to ffmpeg on its command line, so other users on the
  Pi could see it with `ps`. This is fine on a single-user Pi.
* If Immich's machine learning is down during a run, a video can be marked as
  "no faces". Run `--retry-empty` to scan those again.

## Development

Secrets never belong in the repo: your URL and API key live only in
`~/.config/immich-video-faces.env`, outside the repo. A commit hook blocks API
keys, tokens, private keys, LAN IPs, `.env` and `.db` files. Turn it on once per clone
(`install.sh` does this for you):

```bash
git config core.hooksPath .githooks
```

```bash
python3 -m venv .venv && .venv/bin/pip install requests
.venv/bin/python -m unittest discover -s tests -v
```

The tests include an end-to-end run against a fake Immich server (needs `ffmpeg`).
