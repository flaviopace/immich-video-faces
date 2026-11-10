#!/usr/bin/env python3
"""Tag the people that appear in Immich videos.

Immich only looks for faces in a video's thumbnail, so most people in a video
are never recognised. For each video this script:

  1. grabs one frame per second with ffmpeg (streamed from Immich)
  2. checks every frame for faces on the Pi (OpenCV) and keeps the best few frames with faces
  3. uploads only those frames to Immich, archived, in a per-video album named with date and video
  4. waits for Immich to detect and recognise the faces in them
  5. adds every recognised person to the video itself
  6. deletes the frames and the album again

Videos already scanned are remembered in a small SQLite database, so each run
only handles new videos. `--report` sends a summary of the last night to Telegram.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import requests

__version__ = "0.3.0"

log = logging.getLogger("immich-video-faces")

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DEFAULT_ENV_FILE = Path.home() / ".config" / "immich-video-faces.env"
DEFAULT_STATE_DB = Path.home() / ".local" / "share" / "immich-video-faces" / "state.db"
# Every album this script creates has a description starting with this text
ALBUM_DESCRIPTION = "Temporary frames from immich-video-faces"
# Name of every frame this script uploads, e.g. 20261108_180800_t010.0s.jpg
FRAME_NAME_RE = re.compile(r".+_t\d{3,}\.\ds\.jpg")

# Immich's chain of background jobs for an uploaded photo, in order. Face recognition comes last,
# with one job per face (archived photos get a second, "deferred" recognition job).
PIPELINE_QUEUES = ("sidecar", "metadataExtraction", "storageTemplateMigration", "thumbnailGeneration",
                   "faceDetection")
RECOGNITION_QUEUE = "facialRecognition"
PENDING = ("active", "waiting", "delayed", "paused")
ALBUM_DELETE_TRIES = 4
ALBUM_DELETE_PAUSE = 5.0  # seconds, grows with each try


# --------------------------------------------------------------------------- config


@dataclass
class Config:
    url: str
    api_key: str
    state_db: Path = DEFAULT_STATE_DB
    max_videos: int = 3
    candidate_interval: float = 0.5  # seconds between frames checked on the Pi
    candidate_max: int = 120  # cap per video (long videos get a wider interval)
    one_pass_max_seconds: int = 180  # videos up to this length are read in one fast pass
    frames_per_video: int = 60  # max frames uploaded to Immich per video
    frame_max_size: int = 1440  # longest side; Immich analyses a 1440px preview anyway
    face_filter: str = "auto"  # auto: check frames with OpenCV if installed | off: upload without checking
    min_face_px: int = 20  # smaller faces are ignored
    face_score: float = 0.9  # Pi detector confidence needed to count a face
    album_prefix: str = "Video faces"
    debug_dir: Optional[Path] = None  # keep annotated frames here for inspection
    face_wait_timeout: int = 900  # max seconds to wait for Immich per video
    poll_interval: int = 10
    min_wait: int = 20  # never decide "done" before this many seconds
    settle_time: int = 180  # used when queue status can't be read (non-admin key)
    max_attempts: int = 3
    stop_after: Optional[dt.time] = None
    keep_frames: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (existing variables win)."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_hhmm(value: str) -> dt.time:
    hours, minutes = value.strip().split(":")
    return dt.time(int(hours), int(minutes))


def build_config(args: argparse.Namespace) -> Config:
    env = os.environ
    url = env.get("IMMICH_URL", "").strip()
    api_key = env.get("IMMICH_API_KEY", "").strip()
    if not url or not api_key:
        raise SystemExit(
            "IMMICH_URL and IMMICH_API_KEY must be set "
            f"(environment, --env-file, or {DEFAULT_ENV_FILE})"
        )

    cfg = Config(url=url, api_key=api_key)
    if env.get("STATE_DB"):
        cfg.state_db = Path(env["STATE_DB"]).expanduser()
    cfg.max_videos = int(env.get("MAX_VIDEOS", cfg.max_videos))
    cfg.candidate_interval = float(env.get("CANDIDATE_INTERVAL", cfg.candidate_interval))
    cfg.candidate_max = int(env.get("CANDIDATE_MAX", cfg.candidate_max))
    cfg.one_pass_max_seconds = int(env.get("ONE_PASS_MAX_SECONDS", cfg.one_pass_max_seconds))
    cfg.frames_per_video = int(env.get("FRAMES_PER_VIDEO", cfg.frames_per_video))
    cfg.frame_max_size = int(env.get("FRAME_MAX_SIZE", cfg.frame_max_size))
    cfg.face_filter = env.get("FACE_FILTER", cfg.face_filter).strip().lower()
    cfg.min_face_px = int(env.get("MIN_FACE_PX", cfg.min_face_px))
    cfg.face_score = float(env.get("FACE_SCORE", cfg.face_score))
    cfg.album_prefix = env.get("ALBUM_PREFIX", cfg.album_prefix).strip() or cfg.album_prefix
    if env.get("DEBUG_DIR"):
        cfg.debug_dir = Path(env["DEBUG_DIR"]).expanduser()
    cfg.face_wait_timeout = int(env.get("FACE_WAIT_TIMEOUT", cfg.face_wait_timeout))
    cfg.max_attempts = int(env.get("MAX_ATTEMPTS", cfg.max_attempts))
    if env.get("STOP_AFTER"):
        cfg.stop_after = parse_hhmm(env["STOP_AFTER"])
    cfg.keep_frames = env.get("KEEP_FRAMES", "").lower() in ("1", "true", "yes")
    cfg.telegram_token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    cfg.telegram_chat_id = env.get("TELEGRAM_CHAT_ID", "").strip()

    if args.max_videos is not None:
        cfg.max_videos = args.max_videos
    if args.frames is not None:
        cfg.frames_per_video = args.frames
    if args.keep_frames:
        cfg.keep_frames = True
    if args.debug_dir:
        cfg.debug_dir = args.debug_dir.expanduser()
    return cfg


# --------------------------------------------------------------------------- Immich API


class ImmichError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class ImmichClient:
    def __init__(self, url: str, api_key: str, timeout: int = 60):
        base = url.rstrip("/")
        if not base.endswith("/api"):
            base += "/api"
        self.base = base
        self.api_key = api_key
        self.timeout = timeout
        self.major_version = 0
        self.user_id: Optional[str] = None
        self._can_read_queues = True
        self._can_list_jobs = True
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})

    def _call(self, method: str, path: str, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.request(method, self.base + path, **kwargs)
        if resp.status_code >= 400:
            raise ImmichError(
                f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code
            )
        if not resp.content or "json" not in resp.headers.get("Content-Type", ""):
            return None
        return resp.json()

    def server_version(self) -> str:
        v = self._call("GET", "/server/version")
        self.major_version = int(v["major"])
        return f"{v['major']}.{v['minor']}.{v['patch']}"

    def my_user_id(self) -> str:
        self.user_id = self._call("GET", "/users/me")["id"]
        return self.user_id

    def get_asset(self, asset_id: str) -> dict:
        return self._call("GET", f"/assets/{asset_id}")

    def iter_videos(self) -> Iterator[dict]:
        body = {"type": "VIDEO", "size": 1000, "page": 1}
        while True:
            assets = self._call("POST", "/search/metadata", json=body)["assets"]
            yield from assets["items"]
            next_page = assets.get("nextPage")
            if not next_page:
                return
            body["page"] = int(next_page)

    def get_faces(self, asset_id: str) -> List[dict]:
        return self._call("GET", "/faces", params={"id": asset_id}) or []

    def create_face(self, asset_id: str, person_id: str, face: dict) -> None:
        self._call(
            "POST",
            "/faces",
            json={
                "assetId": asset_id,
                "personId": person_id,
                "imageWidth": face["imageWidth"],
                "imageHeight": face["imageHeight"],
                "x": face["boundingBoxX1"],
                "y": face["boundingBoxY1"],
                "width": face["boundingBoxX2"] - face["boundingBoxX1"],
                "height": face["boundingBoxY2"] - face["boundingBoxY1"],
            },
        )

    def upload_frame(self, path: Path, created_at: str):
        data = {
            "fileCreatedAt": created_at,
            "fileModifiedAt": created_at,
            "filename": path.name,
            # Archived: hidden from the timeline, and Immich will only match these
            # faces to people that already exist (it never creates new people from them).
            "visibility": "archive",
        }
        if self.major_version < 2:  # v1.x still requires these two fields
            data["deviceAssetId"] = f"immich-video-faces-{uuid.uuid4()}"
            data["deviceId"] = "immich-video-faces"
        with path.open("rb") as fh:
            res = self._call(
                "POST", "/assets", data=data, files={"assetData": (path.name, fh, "image/jpeg")}, timeout=300
            )
        return res["id"], res.get("status", "created")

    def delete_assets(self, ids: List[str]) -> None:
        # force=True skips the trash, so frames vanish for good
        self._call("DELETE", "/assets", json={"ids": list(ids), "force": True})

    def list_albums(self) -> List[dict]:
        return self._call("GET", "/albums") or []

    def create_album(self, name: str, description: str) -> str:
        return self._call("POST", "/albums", json={"albumName": name, "description": description})["id"]

    def get_album(self, album_id: str) -> dict:
        return self._call("GET", f"/albums/{album_id}")

    def delete_album(self, album_id: str) -> None:
        # removes the album only; assets in it are not touched
        self._call("DELETE", f"/albums/{album_id}")

    def add_to_album(self, album_id: str, ids: List[str]) -> None:
        self._call("PUT", f"/albums/{album_id}/assets", json={"ids": ids})

    def queue_counts(self) -> Optional[Dict[str, int]]:
        """Pending jobs per queue; None if the key can't see Immich's jobs (needs an admin user)."""
        if not self._can_read_queues:
            return None
        try:
            queues = self._call("GET", "/jobs")
        except ImmichError as e:
            if e.status in (401, 403):
                log.info("API key can't read Immich's jobs (needs an admin user + job.read); using timed waits")
                self._can_read_queues = False
                return None
            raise
        out = {}
        for name in PIPELINE_QUEUES + (RECOGNITION_QUEUE,):
            counts = (queues.get(name) or {}).get("jobCounts") or {}
            out[name] = sum(int(counts.get(k) or 0) for k in PENDING)
        return out

    def queue_job_ids(self, queue: str, statuses: Iterable[str]) -> Optional[Set[str]]:
        """The asset/face id of every job in `queue` with one of `statuses`.

        None if this Immich can't list jobs (before v2.4, or missing queueJob.read permission)."""
        if not self._can_list_jobs:
            return None
        params = [("status", s) for s in statuses]
        if len(params) == 1:
            params.append(params[0])  # the API only accepts a list: repeat a single value
        try:
            jobs = self._call("GET", f"/queues/{queue}/jobs", params=params, timeout=120)
        except ImmichError as e:
            if e.status in (400, 401, 403, 404):
                log.info("Immich can't list its queued jobs (HTTP %s); waiting for empty queues instead", e.status)
                self._can_list_jobs = False
                return None
            raise
        return {str(job["data"]["id"]) for job in jobs or [] if (job.get("data") or {}).get("id")}

    def video_url(self, asset_id: str) -> str:
        # playback = Immich's transcoded copy if there is one (smaller, SDR), else the original
        return f"{self.base}/assets/{asset_id}/video/playback"

    def download_original(self, asset_id: str, dest: Path) -> None:
        with self.session.get(f"{self.base}/assets/{asset_id}/original", stream=True, timeout=300) as resp:
            if resp.status_code >= 400:
                raise ImmichError(f"download {asset_id} -> HTTP {resp.status_code}", resp.status_code)
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)


# --------------------------------------------------------------------------- state


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    asset_id     TEXT PRIMARY KEY,
    file_name    TEXT,
    checksum     TEXT,
    status       TEXT NOT NULL,          -- done | failed | waiting (Immich still busy, finish next run)
    attempts     INTEGER NOT NULL DEFAULT 0,
    frames       INTEGER,                -- frames uploaded to Immich
    faces_found  INTEGER,                -- faces Immich detected in those frames
    people_added INTEGER,
    last_error   TEXT,
    updated_at   TEXT NOT NULL
);
-- frames uploaded to Immich that still have to be deleted
CREATE TABLE IF NOT EXISTS pending_deletes (
    asset_id   TEXT PRIMARY KEY,
    video_id   TEXT,
    created_at TEXT NOT NULL
);
-- albums created by this script (the only albums it may delete)
CREATE TABLE IF NOT EXISTS albums (
    album_id   TEXT PRIMARY KEY,
    name       TEXT,
    video_id   TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# columns added after the first release: (table, column, type)
MIGRATIONS = [
    ("videos", "candidates", "INTEGER"),  # frames checked on the Pi
    ("videos", "people_names", "TEXT"),
    ("videos", "faces_recognised", "INTEGER"),
    # fingerprint of every uploaded frame: only a photo matching it is ever deleted
    ("pending_deletes", "file_name", "TEXT"),
    ("pending_deletes", "checksum", "TEXT"),
    ("videos", "people_present", "TEXT"),  # everyone on the video after the scan (incl. found by Immich itself)
]


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        for table, column, kind in MIGRATIONS:
            columns = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        self.db.commit()

    def get(self, asset_id: str) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM videos WHERE asset_id = ?", (asset_id,)).fetchone()

    def record(self, video: dict, status: str, error: Optional[str] = None, **stats) -> None:
        row = self.get(video["id"])
        attempts = (row["attempts"] if row else 0) + 1
        with self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO videos
                   (asset_id, file_name, checksum, status, attempts, candidates, frames, faces_found,
                    faces_recognised, people_added, people_names, people_present, last_error, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video["id"],
                    video.get("originalFileName"),
                    video.get("checksum"),
                    status,
                    attempts,
                    stats.get("candidates"),
                    stats.get("frames"),
                    stats.get("faces_found"),
                    stats.get("faces_recognised"),
                    stats.get("people_added"),
                    ", ".join(stats.get("people_names") or []) or None,
                    ", ".join(stats.get("people_present") or []) or None,
                    error,
                    _now(),
                ),
            )

    def forget(self, asset_id: str) -> int:
        with self.db:
            return self.db.execute("DELETE FROM videos WHERE asset_id = ?", (asset_id,)).rowcount

    def forget_empty(self) -> int:
        with self.db:
            return self.db.execute("DELETE FROM videos WHERE status = 'done' AND faces_found = 0").rowcount

    def forget_failed(self) -> int:
        with self.db:
            return self.db.execute("DELETE FROM videos WHERE status = 'failed'").rowcount

    def add_pending_delete(self, asset_id: str, video_id: str, file_name: Optional[str] = None,
                           checksum: Optional[str] = None) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO pending_deletes (asset_id, video_id, created_at, file_name, checksum)"
                " VALUES (?, ?, ?, ?, ?)",
                (asset_id, video_id, _now(), file_name, checksum),
            )

    def pending_info(self, asset_id: str) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM pending_deletes WHERE asset_id = ?", (asset_id,)).fetchone()

    def add_album(self, album_id: str, name: Optional[str], video_id: Optional[str]) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO albums (album_id, name, video_id, created_at) VALUES (?, ?, ?, ?)",
                            (album_id, name, video_id, _now()))

    def remove_album(self, album_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM albums WHERE album_id = ?", (album_id,))

    def albums(self) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM albums").fetchall()

    def is_pending_delete(self, asset_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM pending_deletes WHERE asset_id = ?", (asset_id,)).fetchone() is not None

    def pending_deletes(self, video_id: Optional[str] = None, skip_videos: Iterable[str] = ()) -> List[str]:
        if video_id:
            rows = self.db.execute("SELECT asset_id, video_id FROM pending_deletes WHERE video_id = ?", (video_id,))
        else:
            rows = self.db.execute("SELECT asset_id, video_id FROM pending_deletes")
        skip = set(skip_videos)
        return [r["asset_id"] for r in rows if r["video_id"] not in skip]

    def waiting_videos(self) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM videos WHERE status = 'waiting'").fetchall()

    def clear_pending_deletes(self, ids: List[str]) -> None:
        with self.db:
            self.db.executemany("DELETE FROM pending_deletes WHERE asset_id = ?", [(i,) for i in ids])

    def start_run(self) -> int:
        with self.db:
            return self.db.execute("INSERT INTO runs (started_at) VALUES (?)", (_now(),)).lastrowid

    def finish_run(self, run_id: int, error: Optional[str] = None) -> None:
        with self.db:
            self.db.execute("UPDATE runs SET finished_at = ?, error = ? WHERE id = ?", (_now(), error, run_id))

    def get_meta(self, key: str) -> Optional[str]:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def summary(self) -> Dict[str, int]:
        rows = self.db.execute("SELECT status, COUNT(*) AS n FROM videos GROUP BY status").fetchall()
        out = {r["status"]: r["n"] for r in rows}
        out["people_added"] = self.db.execute("SELECT COALESCE(SUM(people_added), 0) FROM videos").fetchone()[0]
        out["pending_frame_deletes"] = len(self.pending_deletes())
        return out


def pick_videos(videos: List[dict], state: State, max_attempts: int, owner_id: Optional[str] = None) -> List[dict]:
    """Videos still to scan: never-seen first, then failed ones to retry; newest first."""
    new, retry = [], []
    for v in videos:
        if v.get("visibility") == "hidden" or v.get("isTrashed"):
            continue  # e.g. the video half of a live/motion photo
        if owner_id and v.get("ownerId") and v["ownerId"] != owner_id:
            continue  # partner-shared: we can't tag people on someone else's asset
        row = state.get(v["id"])
        if row is None:
            new.append(v)
        elif row["status"] == "failed" and row["attempts"] < max_attempts:
            retry.append(v)

    def newest_first(v: dict) -> str:
        return v.get("fileCreatedAt") or ""

    new.sort(key=newest_first, reverse=True)
    retry.sort(key=newest_first, reverse=True)
    return new + retry


# --------------------------------------------------------------------------- frames


@dataclass
class Frame:
    at: float  # seconds into the video
    path: Path
    faces: List[float] = field(default_factory=list)  # area (px²) of each usable face found on the Pi
    boxes: List[Tuple] = field(default_factory=list)  # (x, y, w, h, score, usable) for the debug images


def parse_duration(value) -> float:
    """Immich v3 gives milliseconds (int); v1/v2 give text like '0:01:23.456000'."""
    if isinstance(value, (int, float)):
        return value / 1000.0
    try:
        hours, minutes, seconds = (value or "").split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0


def frame_times(duration: float, count: int) -> List[float]:
    """Evenly spaced timestamps, skipping the first and last 5% (often black/blurry)."""
    if duration <= 0:
        return [0.0]
    count = max(1, min(count, int(duration * 2)))  # at most ~2 frames per second
    start, span = duration * 0.05, duration * 0.90
    return sorted({round(start + span * (i + 0.5) / count, 3) for i in range(count)})


def frame_name(stem: str, at: float) -> str:
    return f"{stem}_t{at:05.1f}s.jpg"  # e.g. 20261108_180800_t010.0s.jpg


def scale_filter(max_size: int) -> str:
    # limit the longest side, keep aspect ratio (works for portrait and landscape)
    return f"scale='if(gte(iw,ih),min({max_size},iw),-2)':'if(gte(iw,ih),-2,min({max_size},ih))'"


def probe_duration(source: str, input_args: List[str]) -> float:
    cmd = ["ffprobe", "-v", "error", *input_args, "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", source]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def extract_frame(source: str, input_args: List[str], at: float, dest: Path, max_size: int) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        *input_args, "-ss", f"{at:.3f}", "-i", source,
        "-frames:v", "1", "-vf", scale_filter(max_size), "-q:v", "3",
        str(dest),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timed out at %.1fs", at)
        return False
    if res.returncode != 0:
        log.debug("ffmpeg failed at %.1fs: %s", at, res.stderr.strip()[:300])
    return res.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def grab_one_pass(source: str, input_args: List[str], interval: float, limit: int,
                  stem: str, workdir: Path, max_size: int) -> List[Frame]:
    """Read the video once and keep one frame every `interval` seconds (fast for short videos)."""
    raw = workdir / "raw"
    raw.mkdir(exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        *input_args, "-i", source,
        "-vf", f"fps={1 / interval:.6g},{scale_filter(max_size)}", "-frames:v", str(limit), "-q:v", "3",
        str(raw / "%05d.jpg"),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if res.returncode != 0:
            log.debug("ffmpeg one-pass failed: %s", res.stderr.strip()[:300])
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timed out reading the video")
    frames = []
    for path in sorted(raw.glob("*.jpg")):
        at = (int(path.stem) - 1) * interval
        dest = workdir / frame_name(stem, at)
        path.rename(dest)
        frames.append(Frame(at, dest))
    shutil.rmtree(raw, ignore_errors=True)
    return frames


def grab_seeking(source: str, input_args: List[str], duration: float, count: int,
                 stem: str, workdir: Path, max_size: int) -> List[Frame]:
    """Jump to `count` evenly spaced positions (for long videos: reads only what's needed)."""
    frames = []
    for at in frame_times(duration, count):
        dest = workdir / frame_name(stem, at)
        if extract_frame(source, input_args, at, dest, max_size):
            frames.append(Frame(at, dest))
    return frames


def grab_frames(source: str, input_args: List[str], duration: float, cfg: Config,
                dense: bool, stem: str, workdir: Path) -> List[Frame]:
    if not dense:  # no face check: only the frames we will upload
        return grab_seeking(source, input_args, duration, cfg.frames_per_video, stem, workdir, cfg.frame_max_size)
    if duration <= cfg.one_pass_max_seconds:
        interval = cfg.candidate_interval
        if duration > 0:
            interval = max(interval, duration / cfg.candidate_max)
        frames = grab_one_pass(source, input_args, interval, cfg.candidate_max, stem, workdir, cfg.frame_max_size)
        if frames:
            return frames
    count = min(cfg.candidate_max, max(1, int(duration / cfg.candidate_interval)))
    return grab_seeking(source, input_args, duration, count, stem, workdir, cfg.frame_max_size)


def extract_frames(client: ImmichClient, video: dict, workdir: Path, cfg: Config, dense: bool) -> List[Frame]:
    stem = Path(video.get("originalFileName") or video["id"]).stem
    # Stream straight from Immich; nothing is stored on the Pi's disk.
    source = client.video_url(video["id"])
    input_args = ["-headers", f"x-api-key: {client.api_key}\r\n"]
    duration = parse_duration(video.get("duration")) or probe_duration(source, input_args)

    frames = grab_frames(source, input_args, duration, cfg, dense, stem, workdir)
    if frames:
        return frames

    log.info("  streaming failed, downloading the original video instead")
    download_dir = Path(tempfile.mkdtemp(prefix="immich-video-faces-dl-"))  # on disk, not in RAM
    try:
        local = download_dir / "original"
        client.download_original(video["id"], local)
        duration = duration or probe_duration(str(local), [])
        return grab_frames(str(local), [], duration, cfg, dense, stem, workdir)
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)


class FaceFinder:
    """Face check on the Pi (OpenCV YuNet), used to upload only frames that contain faces."""

    def __init__(self, min_face_px: int, score: float = 0.9):
        import cv2  # noqa: optional dependency (apt install python3-opencv)

        version = tuple(int(x) for x in cv2.__version__.split(".")[:2])
        # the 2023 model needs OpenCV >= 4.8; older OpenCV only runs the 2022 model
        model = "face_detection_yunet_2023mar.onnx" if version >= (4, 8) else "face_detection_yunet_2022mar.onnx"
        self.cv2 = cv2
        # low threshold here so weak hits can be shown in the debug images; `score` decides
        self.detector = cv2.FaceDetectorYN.create(str(MODELS_DIR / model), "", (320, 320), 0.5)
        self.min_face_px = min_face_px
        self.score = score
        log.debug("face check: OpenCV %s, %s", cv2.__version__, model)

    def check(self, frame: Frame) -> None:
        """Fill frame.boxes (everything seen) and frame.faces (faces good enough to upload for)."""
        img = self.cv2.imread(str(frame.path))
        if img is None:
            return
        h, w = img.shape[:2]
        # full resolution: faces in videos are often only 20-40px wide
        self.detector.setInputSize((w, h))
        _, found = self.detector.detect(img)
        for f in [] if found is None else found:
            x, y, fw, fh, score = int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[-1])
            usable = (
                score >= self.score
                and min(fw, fh) >= self.min_face_px
                and not (fw > 0.9 * w and fh > 0.9 * h)  # whole-frame false positive (e.g. black frame)
            )
            frame.boxes.append((x, y, int(fw), int(fh), score, usable))
            if usable:
                frame.faces.append(fw * fh)


def load_face_finder(cfg: Config) -> Optional[FaceFinder]:
    if cfg.face_filter == "off":
        return None
    try:
        return FaceFinder(cfg.min_face_px, cfg.face_score)
    except Exception as e:  # ImportError, missing model, ...
        log.warning("Face check on the Pi unavailable (%s); uploading frames without checking", e)
        return None


def select_frames(frames: List[Frame], limit: int) -> List[Frame]:
    """Up to `limit` frames, spread over the video: the best frame from each time slice
    (most faces, then largest faces). Adjacent moments matter: a face recognised at 18.2s
    can be missed at 18.0s, so with enough slots every frame with faces is uploaded."""
    if not frames:
        return []
    best: Dict[int, Frame] = {}
    for index, frame in enumerate(frames):
        if not frame.faces:
            continue
        slot = index * limit // len(frames)
        current = best.get(slot)
        if current is None or (len(frame.faces), sum(frame.faces)) > (len(current.faces), sum(current.faces)):
            best[slot] = frame
    return sorted(best.values(), key=lambda f: f.at)


def work_dir(label: str) -> Path:
    """Per-video temp folder, in RAM (/dev/shm) when possible to spare the Pi's SD card."""
    shm = Path("/dev/shm")
    base = shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(tempfile.gettempdir())
    path = base / "immich-video-faces" / label
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def write_debug_frames(folder: Path, frames: List[Frame], chosen: List[Frame]) -> None:
    """Copy every checked frame to `folder`, with the faces found drawn on it."""
    folder.mkdir(parents=True, exist_ok=True)
    chosen_paths = {f.path for f in chosen}
    try:
        import cv2
    except ImportError:
        cv2 = None
    for frame in frames:
        name = ("UPLOADED_" if frame.path in chosen_paths else "") + frame.path.name
        if cv2 is None:
            shutil.copy(frame.path, folder / name)
            continue
        img = cv2.imread(str(frame.path))
        if img is None:
            continue
        for x, y, w, h, score, usable in frame.boxes:
            color = (0, 200, 0) if usable else (0, 165, 255)  # green = counted, orange = ignored
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{score:.2f} {w}px", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(folder / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


# --------------------------------------------------------------------------- faces


def _jobs_pending(client: ImmichClient, queues: Iterable[str], ids: Set[str]) -> bool:
    """Is any job for `ids` still queued or running in `queues`?"""
    counts = client.queue_counts() or {}
    for queue in queues:
        if not counts.get(queue):
            continue
        listed = client.queue_job_ids(queue, PENDING)
        if listed is None or listed & ids:
            return True  # can't see which jobs are queued (old Immich): wait for an empty queue
    return False


def check_immich(client: ImmichClient, frame_ids: List[str]) -> Tuple[Dict[str, List[dict]], Optional[bool]]:
    """Faces per frame, and whether Immich has finished every background job for these frames.

    True/False, or None when the API key can't see Immich's jobs (non-admin)."""
    if client.queue_counts() is None:
        return {fid: client.get_faces(fid) for fid in frame_ids}, None
    # 1. frames still going through metadata -> thumbnails -> face detection?
    if _jobs_pending(client, PIPELINE_QUEUES, set(frame_ids)):
        return {fid: client.get_faces(fid) for fid in frame_ids}, False
    # 2. detection is done, so the faces are final; is any of them still being recognised?
    faces = {fid: client.get_faces(fid) for fid in frame_ids}
    face_ids = {str(f["id"]) for per_frame in faces.values() for f in per_frame}
    return faces, not _jobs_pending(client, (RECOGNITION_QUEUE,), face_ids)


def failed_frames(client: ImmichClient, frame_ids: List[str]) -> Set[str]:
    """Frames whose thumbnail or face detection job failed in Immich (e.g. machine learning down)."""
    failed: Set[str] = set()
    for queue in PIPELINE_QUEUES:
        failed |= (client.queue_job_ids(queue, ("failed",)) or set()) & set(frame_ids)
    return failed


def wait_for_faces(client: ImmichClient, frame_ids: List[str], cfg: Config) -> Tuple[Dict[str, List[dict]], bool]:
    """Wait until Immich has finished its background face jobs for the frames.

    Returns the faces per frame and whether Immich finished (False = still busy at the timeout)."""
    start = time.monotonic()
    previous = None
    last_seen, last_change = None, start
    while True:
        time.sleep(cfg.poll_interval)
        faces, done = check_immich(client, frame_ids)
        found = [f for per_frame in faces.values() for f in per_frame]
        assigned = sum(1 for f in found if f.get("person"))
        signature = sorted((str(f["id"]), str((f.get("person") or {}).get("id"))) for f in found)
        now = time.monotonic()
        elapsed = now - start
        log.debug("  %3ds: %d faces, %d recognised, immich done=%s", elapsed, len(found), assigned, done)

        if done is not None:
            # finished twice in a row with the same result: nothing can still change
            if done and signature == previous:
                return faces, True
            previous = signature if done else None
        else:  # non-admin key: no view of Immich's jobs, wait until results stop changing
            if signature != last_seen:
                last_seen, last_change = signature, now
            if elapsed >= cfg.min_wait and found and assigned == len(found):
                return faces, True
            if elapsed >= cfg.min_wait and now - last_change >= cfg.settle_time:
                return faces, True
        if elapsed >= cfg.face_wait_timeout:
            return faces, False


def best_face_per_person(frame_faces: Dict[str, List[dict]]) -> Dict[str, dict]:
    """For each recognised person, keep their biggest face across all frames."""
    best: Dict[str, dict] = {}
    for faces in frame_faces.values():
        for face in faces:
            person = face.get("person")
            if not person:
                continue
            area = (face["boundingBoxX2"] - face["boundingBoxX1"]) * (face["boundingBoxY2"] - face["boundingBoxY1"])
            current = best.get(person["id"])
            if current is None or area > current["_area"]:
                best[person["id"]] = {**face, "_area": area}
    return best


def copy_people_to_video(client: ImmichClient, video_id: str, frame_faces: Dict[str, List[dict]]) -> List[str]:
    already = {f["person"]["id"] for f in client.get_faces(video_id) if f.get("person")}
    added = []
    for person_id, face in best_face_per_person(frame_faces).items():
        if person_id in already:
            continue
        client.create_face(video_id, person_id, face)
        added.append(face["person"].get("name") or "unnamed")
    return added


def people_on(client: ImmichClient, video_id: str) -> List[str]:
    """Names of everyone on the video now ("unnamed" once per person without a name)."""
    names = {}
    for face in client.get_faces(video_id):
        person = face.get("person")
        if person:
            names[person["id"]] = person.get("name") or "unnamed"
    return sorted(names.values())


def describe_faces(faces: List[dict]) -> str:
    if not faces:
        return "no face"
    names = [(f.get("person") or {}).get("name") or ("unnamed person" if f.get("person") else "not recognised")
             for f in faces]
    return f"{len(faces)} face(s): " + ", ".join(names)


# --------------------------------------------------------------------------- run


def file_checksum(path: Path) -> str:
    """Base64 SHA1 of a file, the same fingerprint Immich shows as an asset's checksum."""
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def is_script_frame(asset: dict, expected: Optional[sqlite3.Row], user_id: Optional[str]) -> Tuple[bool, str]:
    """Is this Immich asset really a frame this script uploaded? Anything else must never be deleted."""
    if asset.get("type") != "IMAGE":
        return False, f"it is a {asset.get('type') or 'unknown asset'}, not a frame"
    if user_id and asset.get("ownerId") and asset["ownerId"] != user_id:
        return False, "it belongs to another user"
    name = asset.get("originalFileName") or ""
    if expected is not None and expected["file_name"]:
        if name != expected["file_name"]:
            return False, f"its name is not the name of the uploaded frame ({expected['file_name']})"
    elif not FRAME_NAME_RE.fullmatch(name):
        return False, "its name is not a frame name"
    if expected is not None and expected["checksum"] and asset.get("checksum") != expected["checksum"]:
        return False, "its content is not the content of the uploaded frame"
    return True, ""


def delete_frames(client: ImmichClient, state: State, ids: List[str]) -> bool:
    """Delete frames this script uploaded. Returns True when none of them is left in Immich.

    Safety: every asset is fetched first, and only a photo with the exact name and content (SHA1)
    of a frame this script uploaded is deleted. Anything else is refused and never retried."""
    if not ids:
        return True
    verified = []
    try:
        for asset_id in ids:
            try:
                asset = client.get_asset(asset_id)
            except ImmichError as e:
                if e.status in (400, 404):  # already gone
                    state.clear_pending_deletes([asset_id])
                    continue
                raise
            ok, why = is_script_frame(asset, state.pending_info(asset_id), client.user_id)
            if ok:
                verified.append(asset_id)
            else:
                log.error("NOT deleting %s (%r): %s", asset_id, asset.get("originalFileName"), why)
                state.clear_pending_deletes([asset_id])
        if verified:
            client.delete_assets(verified)
            state.clear_pending_deletes(verified)
            log.info("  deleted %d frame(s) from Immich", len(verified))
        return True
    except Exception as e:  # network, Immich down...: next run retries
        log.warning("Could not delete %d frame(s), will retry next run: %s", len(ids), e)
        return False


def delete_album_if_safe(client: ImmichClient, state: State, album_id: str) -> None:
    """Delete an album this script created, and only when it holds no photo at all.

    (Deleting an album in Immich never deletes the photos in it; this check also keeps any album
    you added your own photos to.)"""
    try:
        album = client.get_album(album_id)
    except ImmichError as e:
        if e.status in (400, 404):  # already gone
            state.remove_album(album_id)
        else:
            log.warning("Could not check album %s (next run retries): %s", album_id, e)
        return
    name = album.get("albumName")
    count = album.get("assetCount")
    if count is None or count > 0:
        if count and not state.pending_deletes():
            log.warning("Keeping album %r: it contains %s photo(s) not made by this script", name, count)
            state.remove_album(album_id)  # not ours to manage any more
        return
    # Immich can answer HTTP 500 while it is still removing the frames just deleted: retry a few times
    for attempt in range(ALBUM_DELETE_TRIES):
        try:
            client.delete_album(album_id)
            state.remove_album(album_id)
            log.info("  deleted album %r", name)
            return
        except ImmichError as e:
            if e.status < 500 or attempt == ALBUM_DELETE_TRIES - 1:
                log.warning("Could not delete album %r (next run retries): %s", name, e)
                return
            time.sleep(ALBUM_DELETE_PAUSE * (attempt + 1))


def harvest_waiting(client: ImmichClient, state: State, cfg: Config) -> Set[str]:
    """Finish videos whose frames Immich was still processing at the end of an earlier run.

    Returns the ids of videos that are still waiting (their frames must be kept)."""
    still_waiting = set()
    for row in state.waiting_videos():
        video = {"id": row["asset_id"], "originalFileName": row["file_name"], "checksum": row["checksum"]}
        frame_ids = state.pending_deletes(row["asset_id"])
        if not frame_ids:
            state.record(video, "failed", error="temporary frames were lost before Immich finished")
            continue
        faces, done = check_immich(client, frame_ids)
        if done is False:
            if row["attempts"] >= cfg.max_attempts:
                state.record(video, "failed", error="Immich did not finish the frames in time")
            else:
                log.info("Immich is still busy with %s, collecting next run", row["file_name"])
                state.record(video, "waiting", candidates=row["candidates"], frames=row["frames"])
                still_waiting.add(row["asset_id"])
            continue
        found = sum(len(f) for f in faces.values())
        recognised = sum(1 for per_frame in faces.values() for f in per_frame if f.get("person"))
        added = copy_people_to_video(client, row["asset_id"], faces)
        log.info("Collected %s from an earlier run: %d face(s), %d recognised, people added: %s",
                 row["file_name"], found, recognised, ", ".join(added) or "none")
        state.record(video, "done", candidates=row["candidates"], frames=len(frame_ids), faces_found=found,
                     faces_recognised=recognised, people_added=len(added), people_names=added,
                     people_present=people_on(client, row["asset_id"]))
    return still_waiting


def cleanup_leftovers(client: ImmichClient, state: State, cfg: Config) -> None:
    """Finish or delete what an earlier run left behind (Immich busy, crash, network loss, --keep-frames)."""
    still_waiting = harvest_waiting(client, state, cfg)
    delete_frames(client, state, state.pending_deletes(skip_videos=still_waiting))
    # albums made before albums were recorded: adopt them only if they carry our exact description
    known = {r["album_id"] for r in state.albums()}
    for album in client.list_albums():
        if (album["id"] not in known and (album.get("albumName") or "").startswith(cfg.album_prefix)
                and (album.get("description") or "").startswith(ALBUM_DESCRIPTION)):
            state.add_album(album["id"], album.get("albumName"), None)
    for row in state.albums():
        if row["video_id"] in still_waiting or (row["video_id"] and state.pending_deletes(row["video_id"])):
            continue  # its frames are still needed or not deleted yet
        delete_album_if_safe(client, state, row["album_id"])


def process_video(client: ImmichClient, state: State, cfg: Config, finder: Optional[FaceFinder],
                  video: dict) -> bool:
    name = video.get("originalFileName") or video["id"]
    stem = Path(name).stem
    started = dt.datetime.now()
    label = f"{started:%Y-%m-%d_%H%M}_{stem}"  # folder names: date + video, easy to find
    log.info("Scanning %s (%s)", name, video["id"])
    uploaded: List[str] = []
    album_id: Optional[str] = None
    keep_for_later = False
    debug_folder = cfg.debug_dir / label if cfg.debug_dir else None
    workdir = work_dir(label)
    try:
        t0 = time.monotonic()
        candidates = extract_frames(client, video, workdir, cfg, dense=finder is not None)
        if not candidates:
            raise RuntimeError("could not extract any frame")

        if finder:
            for frame in candidates:
                finder.check(frame)
            chosen = select_frames(candidates, cfg.frames_per_video)
            log.info("  checked %d frame(s) on the Pi in %.0fs, %d with faces, uploading %d: %s",
                     len(candidates), time.monotonic() - t0, sum(1 for f in candidates if f.faces), len(chosen),
                     ", ".join(f"{f.at:g}s ({len(f.faces)})" for f in chosen) or "-")
        else:
            chosen = candidates
        if debug_folder:
            write_debug_frames(debug_folder, candidates, chosen)
            log.info("  debug frames saved in %s", debug_folder)
        if not chosen:
            state.record(video, "done", candidates=len(candidates), frames=0, faces_found=0,
                         faces_recognised=0, people_added=0, people_present=people_on(client, video["id"]))
            log.info("  no faces in this video, nothing uploaded")
            return True

        created_at = video.get("fileCreatedAt") or dt.datetime.now(dt.timezone.utc).isoformat()
        names = {}
        for frame in chosen:
            asset_id, status = client.upload_frame(frame.path, created_at)
            if status == "duplicate" and not state.is_pending_delete(asset_id):
                # Same file already in Immich and not uploaded by us: leave it alone.
                log.warning("  %s already exists in Immich as %s, skipping it", frame.path.name, asset_id)
                continue
            state.add_pending_delete(asset_id, video["id"], frame.path.name, file_checksum(frame.path))
            uploaded.append(asset_id)
            names[asset_id] = frame.path.name
        if not uploaded:
            raise RuntimeError("no frame could be uploaded")
        album_name = f"{cfg.album_prefix} {started:%Y-%m-%d %H:%M} {name}"
        try:
            album_id = client.create_album(album_name, f"{ALBUM_DESCRIPTION} for {name}. Deleted automatically.")
            state.add_album(album_id, album_name, video["id"])
            client.add_to_album(album_id, uploaded)
        except ImmichError as e:
            log.warning("  could not create album %r: %s", album_name, e)

        log.info("  uploaded %d frame(s) to album %r, waiting for Immich's background jobs...",
                 len(uploaded), album_name)
        frame_faces, finished = wait_for_faces(client, uploaded, cfg)
        if not finished:
            # Immich is still busy (e.g. a big backup): keep frames + album, the next run collects the result
            log.warning("  Immich still busy after %ds: frames kept, results collected next run",
                        cfg.face_wait_timeout)
            state.record(video, "waiting", candidates=len(candidates), frames=len(uploaded))
            keep_for_later = True
            return True
        empty = [fid for fid, faces in frame_faces.items() if not faces]
        broken = failed_frames(client, empty) if empty else set()
        if broken:
            raise RuntimeError(f"Immich failed to process {len(broken)} frame(s), see Immich > Jobs; retried later")
        found = sum(len(f) for f in frame_faces.values())
        recognised = sum(1 for faces in frame_faces.values() for f in faces if f.get("person"))
        for asset_id, faces in frame_faces.items():
            log.info("    %s: %s", names[asset_id], describe_faces(faces))
        added = copy_people_to_video(client, video["id"], frame_faces)
        log.info("  %d face(s) found by Immich, %d recognised, people added to video: %s",
                 found, recognised, ", ".join(added) if added else "none")
        if debug_folder:
            lines = [f"{name}  ({video['id']})", f"scanned {started:%Y-%m-%d %H:%M}", ""]
            lines += [f"{names[a]}: Immich found {describe_faces(f)}" for a, f in frame_faces.items()]
            lines += ["", f"people added to the video: {', '.join(added) or 'none'}"]
            (debug_folder / "summary.txt").write_text("\n".join(lines) + "\n")
        state.record(video, "done", candidates=len(candidates), frames=len(uploaded), faces_found=found,
                     faces_recognised=recognised, people_added=len(added), people_names=added,
                     people_present=people_on(client, video["id"]))
        return True
    except Exception as e:
        log.error("  failed: %s", e)
        log.debug("details", exc_info=True)
        state.record(video, "failed", error=str(e)[:500])
        return False
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if keep_for_later:
            pass
        elif cfg.keep_frames:
            if uploaded:
                log.info("  keeping the frames and album in Immich until the next run (keep frames is on)")
        elif delete_frames(client, state, uploaded) and album_id:
            delete_album_if_safe(client, state, album_id)


def compute_deadline(stop_after: Optional[dt.time], now: dt.datetime) -> Optional[dt.datetime]:
    if stop_after is None:
        return None
    deadline = now.replace(hour=stop_after.hour, minute=stop_after.minute, second=0, microsecond=0)
    if deadline <= now:
        deadline += dt.timedelta(days=1)
    return deadline


@contextlib.contextmanager
def single_instance(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another run is already in progress")
        yield


def scan(cfg: Config, state: State, client: ImmichClient, args: argparse.Namespace) -> int:
    version = client.server_version()
    owner_id = client.my_user_id()
    log.info("Connected to Immich %s", version)

    if args.video_id:
        video = client.get_asset(args.video_id)
        if video.get("type") != "VIDEO":
            raise SystemExit(f"{args.video_id} is not a video")
        todo = [video]
    else:
        videos = list(client.iter_videos())
        todo = pick_videos(videos, state, cfg.max_attempts, owner_id)
        log.info("%d video(s) in Immich, %d still to scan", len(videos), len(todo))

    if args.status:
        summary = state.summary()
        print(f"Immich {version}")
        print(f"Scanned:          {summary.get('done', 0)}")
        print(f"Failed:           {summary.get('failed', 0)} (retried up to {cfg.max_attempts} times)")
        print(f"Still to scan:    {len(todo)}")
        print(f"People added:     {summary['people_added']}")
        print(f"Frames to delete: {summary['pending_frame_deletes']}")
        return 0

    batch = todo[: cfg.max_videos]
    if args.dry_run:
        for v in batch:
            print(f"{v['id']}  {v.get('fileCreatedAt', '')[:10]}  {v.get('originalFileName', '')}")
        return 0

    cleanup_leftovers(client, state, cfg)
    if not batch:
        log.info("Nothing to do")
        state.set_meta("remaining", str(len(todo)))
        return 0

    finder = load_face_finder(cfg)
    deadline = compute_deadline(cfg.stop_after, dt.datetime.now())
    failures = 0
    for video in batch:
        if deadline and dt.datetime.now() >= deadline:
            log.info("Reached STOP_AFTER %s, stopping", cfg.stop_after.strftime("%H:%M"))
            break
        if not process_video(client, state, cfg, finder, video):
            failures += 1
    if not args.video_id:
        state.set_meta("remaining", str(len(pick_videos(todo, state, cfg.max_attempts))))
    return 1 if failures else 0


def run(cfg: Config, args: argparse.Namespace) -> int:
    state = State(cfg.state_db)

    if args.forget:
        print(f"Forgot {state.forget(args.forget)} video(s)")
        return 0
    if args.retry_empty:
        print(f"{state.forget_empty()} video(s) with no faces will be scanned again")
        return 0
    if args.retry_failed:
        print(f"{state.forget_failed()} failed video(s) will be scanned again")
        return 0
    if args.report:
        return send_report(cfg, state, print_only=args.dry_run)

    client = ImmichClient(cfg.url, cfg.api_key)
    if args.status or args.dry_run:
        return scan(cfg, state, client, args)

    run_id = state.start_run()
    try:
        result = scan(cfg, state, client, args)
        state.finish_run(run_id)
        return result
    except BaseException as e:
        state.finish_run(run_id, error=str(e)[:300] or type(e).__name__)
        raise


# --------------------------------------------------------------------------- Telegram report


def build_report(state: State, since: str) -> str:
    rows = state.db.execute("SELECT * FROM videos WHERE updated_at > ? ORDER BY updated_at", (since,)).fetchall()
    runs = state.db.execute("SELECT * FROM runs WHERE started_at > ? ORDER BY id", (since,)).fetchall()
    done = [r for r in rows if r["status"] == "done"]
    failed = [r for r in rows if r["status"] == "failed"]
    waiting = state.waiting_videos()

    lines = ["🎬 Immich – volti nei video, riepilogo della notte", ""]
    if not runs:
        lines.append("⚠️ La scansione non è partita dall'ultimo riepilogo.")
    for r in runs:
        if r["error"]:
            lines.append(f"⚠️ La scansione delle {r['started_at'][11:16]} si è fermata con un errore: {r['error']}")
        elif not r["finished_at"]:
            lines.append(f"⚠️ La scansione delle {r['started_at'][11:16]} non è terminata.")
    if runs and runs[0]["finished_at"]:  # the nightly scan is the first run since the last report
        start = dt.datetime.fromisoformat(runs[0]["started_at"])
        end = dt.datetime.fromisoformat(runs[0]["finished_at"])
        lines.append(f"🕑 {start:%H:%M} → {end:%H:%M} ({round((end - start).total_seconds() / 60)} min)")

    checked = sum(r["candidates"] or 0 for r in done)
    uploaded = sum(r["frames"] or 0 for r in done)
    faces = sum(r["faces_found"] or 0 for r in done)
    recognised = sum(r["faces_recognised"] or 0 for r in done)
    with_faces = sum(1 for r in done if r["frames"])
    with_people = [r for r in done if r["people_names"]]
    lines += [
        f"Video analizzati: {len(done)}" + (f" (+{len(failed)} con errore)" if failed else ""),
        f"Video con volti: {with_faces}",
        f"Video a cui sono state aggiunte persone: {len(with_people)}",
        f"Fotogrammi controllati sul Pi: {checked}, caricati su Immich: {uploaded}",
        f"Volti trovati da Immich: {faces} ({recognised} riconosciuti)",
    ]

    def count(column: str) -> Tuple[Dict[str, int], int]:
        named: Dict[str, int] = {}
        unnamed = 0
        for r in done:
            for person in {n.strip() for n in (r[column] or "").split(",") if n.strip()}:
                if person == "unnamed":
                    unnamed += 1
                else:
                    named[person] = named.get(person, 0) + 1
        return named, unnamed

    present, present_unnamed = count("people_present")
    added_by_scan, _ = count("people_names")
    if present or present_unnamed:
        lines += ["", f"👤 Persone presenti nei {len(done)} video:"]
        for person, n in sorted(present.items(), key=lambda kv: (-kv[1], kv[0])):
            extra = f" (aggiunto stanotte in {added_by_scan[person]})" if added_by_scan.get(person) else ""
            lines.append(f"• {person}: in {n} video{extra}")
        if present_unnamed:
            lines.append(f"• persone senza nome in Immich: in {present_unnamed} video")
    if with_people:
        lines += ["", "🎞 Dettaglio:"]
        for r in with_people[:15]:
            names = r["people_names"].replace("unnamed", "senza nome")
            lines.append(f"• {r['file_name']}: {names}")
        if len(with_people) > 15:
            lines.append(f"• … e altri {len(with_people) - 15}")
    if failed:
        lines.append("")
        for r in failed[:5]:
            lines.append(f"✗ {r['file_name']}: {(r['last_error'] or '')[:120]}")

    lines.append("")
    if waiting:
        lines.append(f"⏳ {len(waiting)} video ancora in elaborazione su Immich, completati la prossima notte.")
    pending = len(state.pending_deletes(skip_videos=[r["asset_id"] for r in waiting]))
    albums = len([a for a in state.albums() if a["video_id"] not in {r["asset_id"] for r in waiting}])
    if not pending and not albums:
        lines.append("🧹 Tutti i fotogrammi e gli album temporanei sono stati cancellati.")
    else:
        if pending:
            lines.append(f"⚠️ {pending} fotogrammi ancora su Immich, verranno cancellati alla prossima scansione.")
        if albums:
            lines.append(f"⚠️ {albums} album temporanei ancora su Immich, verranno cancellati alla prossima scansione.")
    remaining = state.get_meta("remaining")
    if remaining is not None:
        lines.append(f"Video ancora da analizzare: {remaining}")
    return "\n".join(lines)


def send_telegram(token: str, chat_id: str, text: str) -> None:
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=30,
    )
    if resp.status_code != 200:
        # never log the URL: it contains the bot token
        raise RuntimeError(f"Telegram answered HTTP {resp.status_code}: {resp.text[:200]}")


def send_report(cfg: Config, state: State, print_only: bool = False) -> int:
    now = _now()
    since = state.get_meta("last_report_at") or (dt.datetime.now() - dt.timedelta(days=1)).isoformat(timespec="seconds")
    text = build_report(state, since)
    if print_only or not (cfg.telegram_token and cfg.telegram_chat_id):
        print(text)
        if not print_only:
            log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set: report printed only")
        return 0
    send_telegram(cfg.telegram_token, cfg.telegram_chat_id, text)
    state.set_meta("last_report_at", now)
    log.info("Report sent to Telegram")
    return 0


# --------------------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tag people in Immich videos using sampled frames.")
    parser.add_argument("--env-file", type=Path, help=f"config file (default: {DEFAULT_ENV_FILE} if present)")
    parser.add_argument("--max-videos", type=int, help="videos to scan this run (default 3)")
    parser.add_argument("--frames", type=int, help="max frames uploaded per video (default 60)")
    parser.add_argument("--video-id", help="scan this one video now, even if already scanned")
    parser.add_argument("--dry-run", action="store_true", help="only list the videos the next run would scan")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    parser.add_argument("--report", action="store_true",
                        help="send the summary since the last report to Telegram (with --dry-run: print it)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="leave frames and album in Immich until the next run, to inspect them")
    parser.add_argument("--debug-dir", type=Path,
                        help="save every checked frame (faces drawn on it) in a dated folder per video")
    parser.add_argument("--forget", metavar="VIDEO_ID", help="mark a video as not scanned")
    parser.add_argument("--retry-empty", action="store_true", help="rescan videos where no face was found")
    parser.add_argument("--retry-failed", action="store_true", help="rescan videos that failed too many times")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    env_file = args.env_file or (DEFAULT_ENV_FILE if DEFAULT_ENV_FILE.exists() else None)
    if env_file:
        load_env_file(env_file)
    cfg = build_config(args)

    if args.report:  # read-only, may run while a scan is still going
        return run(cfg, args)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (sudo apt install ffmpeg)")
    with single_instance(cfg.state_db.with_suffix(".lock")):
        return run(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
