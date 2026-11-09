#!/usr/bin/env python3
"""Tag the people that appear in Immich videos.

Immich only looks for faces in a video's thumbnail, so most people in a video
are never recognised. For each video this script:

  1. grabs a handful of frames with ffmpeg (streamed from Immich, no full download)
  2. uploads them to Immich as archived photos in a dedicated album
  3. waits for Immich to detect and recognise the faces in those frames
  4. adds every recognised person to the video itself
  5. deletes the frames again

Videos already scanned are remembered in a small SQLite database, so each run
only handles new videos, a few per run (3 by default).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

__version__ = "0.1.0"

log = logging.getLogger("immich-video-faces")

DEFAULT_ENV_FILE = Path.home() / ".config" / "immich-video-faces.env"
DEFAULT_STATE_DB = Path.home() / ".local" / "share" / "immich-video-faces" / "state.db"

# Immich queues that must be empty before a frame is fully processed.
WATCHED_QUEUES = ("metadataExtraction", "thumbnailGeneration", "faceDetection", "facialRecognition")


# --------------------------------------------------------------------------- config


@dataclass
class Config:
    url: str
    api_key: str
    state_db: Path = DEFAULT_STATE_DB
    max_videos: int = 3
    frames_per_video: int = 8
    frame_max_width: int = 1920
    album_name: str = "Video face scan (temp)"
    face_wait_timeout: int = 900  # max seconds to wait for Immich per video
    poll_interval: int = 15
    min_wait: int = 60  # never decide "done" before this many seconds
    settle_time: int = 180  # used when queue status can't be read (non-admin key)
    max_attempts: int = 3
    stop_after: Optional[dt.time] = None
    keep_frames: bool = False


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
    cfg.frames_per_video = int(env.get("FRAMES_PER_VIDEO", cfg.frames_per_video))
    cfg.frame_max_width = int(env.get("FRAME_MAX_WIDTH", cfg.frame_max_width))
    cfg.album_name = env.get("ALBUM_NAME", cfg.album_name)
    cfg.face_wait_timeout = int(env.get("FACE_WAIT_TIMEOUT", cfg.face_wait_timeout))
    cfg.max_attempts = int(env.get("MAX_ATTEMPTS", cfg.max_attempts))
    if env.get("STOP_AFTER"):
        cfg.stop_after = parse_hhmm(env["STOP_AFTER"])
    cfg.keep_frames = env.get("KEEP_FRAMES", "").lower() in ("1", "true", "yes")

    if args.max_videos is not None:
        cfg.max_videos = args.max_videos
    if args.frames is not None:
        cfg.frames_per_video = args.frames
    if args.keep_frames:
        cfg.keep_frames = True
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
        self._can_read_queues = True
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
        return self._call("GET", "/users/me")["id"]

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

    def find_or_create_album(self, name: str) -> str:
        for album in self._call("GET", "/albums") or []:
            if album.get("albumName") == name:
                return album["id"]
        album = self._call(
            "POST",
            "/albums",
            json={
                "albumName": name,
                "description": "Temporary video frames used by immich-video-faces. Safe to ignore.",
            },
        )
        return album["id"]

    def add_to_album(self, album_id: str, ids: List[str]) -> None:
        self._call("PUT", f"/albums/{album_id}/assets", json={"ids": ids})

    def queues_busy(self) -> Optional[bool]:
        """True/False if Immich is still processing; None if we can't tell (non-admin key)."""
        if not self._can_read_queues:
            return None
        try:
            queues = self._call("GET", "/jobs")
        except ImmichError as e:
            if e.status in (401, 403):
                log.info("API key can't read job queues (needs an admin user + job.read); using timed waits")
                self._can_read_queues = False
                return None
            raise
        pending = 0
        for name in WATCHED_QUEUES:
            counts = (queues.get(name) or {}).get("jobCounts") or {}
            pending += sum(int(counts.get(k) or 0) for k in ("active", "waiting", "delayed"))
        return pending > 0

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
    status       TEXT NOT NULL,          -- done | failed
    attempts     INTEGER NOT NULL DEFAULT 0,
    frames       INTEGER,
    faces_found  INTEGER,
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
"""


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def get(self, asset_id: str) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM videos WHERE asset_id = ?", (asset_id,)).fetchone()

    def record(self, video: dict, status: str, error: Optional[str] = None, **stats) -> None:
        row = self.get(video["id"])
        attempts = (row["attempts"] if row else 0) + 1
        with self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO videos
                   (asset_id, file_name, checksum, status, attempts, frames, faces_found,
                    people_added, last_error, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video["id"],
                    video.get("originalFileName"),
                    video.get("checksum"),
                    status,
                    attempts,
                    stats.get("frames"),
                    stats.get("faces_found"),
                    stats.get("people_added"),
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

    def add_pending_delete(self, asset_id: str, video_id: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO pending_deletes (asset_id, video_id, created_at) VALUES (?, ?, ?)",
                (asset_id, video_id, _now()),
            )

    def is_pending_delete(self, asset_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM pending_deletes WHERE asset_id = ?", (asset_id,)).fetchone() is not None

    def pending_deletes(self) -> List[str]:
        return [r["asset_id"] for r in self.db.execute("SELECT asset_id FROM pending_deletes")]

    def clear_pending_deletes(self, ids: List[str]) -> None:
        with self.db:
            self.db.executemany("DELETE FROM pending_deletes WHERE asset_id = ?", [(i,) for i in ids])

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
        elif row["status"] != "done" and row["attempts"] < max_attempts:
            retry.append(v)

    def newest_first(v: dict) -> str:
        return v.get("fileCreatedAt") or ""

    new.sort(key=newest_first, reverse=True)
    retry.sort(key=newest_first, reverse=True)
    return new + retry


# --------------------------------------------------------------------------- frames


def parse_duration(value: Optional[str]) -> float:
    """Immich durations look like '0:01:23.456000'."""
    try:
        hours, minutes, seconds = (value or "").split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0


def frame_times(duration: float, count: int) -> List[float]:
    """Evenly spaced timestamps, skipping the first and last 5% (often black/blurry)."""
    if duration <= 0:
        return [0.0]
    count = max(1, min(count, int(duration)))  # at most ~1 frame per second
    start, span = duration * 0.05, duration * 0.90
    return sorted({round(start + span * (i + 0.5) / count, 3) for i in range(count)})


def probe_duration(source: str, input_args: List[str]) -> float:
    cmd = ["ffprobe", "-v", "error", *input_args, "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", source]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def extract_frame(source: str, input_args: List[str], at: float, dest: Path, max_width: int) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        *input_args, "-ss", f"{at:.3f}", "-i", source,
        "-frames:v", "1", "-vf", f"scale='min({max_width},iw)':-2", "-q:v", "2",
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


def extract_frames(client: ImmichClient, video: dict, workdir: Path, cfg: Config) -> List[Path]:
    stem = Path(video.get("originalFileName") or video["id"]).stem
    # Stream straight from Immich: ffmpeg seeks with HTTP range requests, so only
    # the needed parts of the video are transferred.
    source = client.video_url(video["id"])
    input_args = ["-headers", f"x-api-key: {client.api_key}\r\n"]
    duration = parse_duration(video.get("duration")) or probe_duration(source, input_args)

    frames = _grab(source, input_args, duration, stem, workdir, cfg)
    if frames:
        return frames

    log.info("Streaming failed, downloading the original video instead")
    local = workdir / "original"
    client.download_original(video["id"], local)
    duration = duration or probe_duration(str(local), [])
    return _grab(str(local), [], duration, stem, workdir, cfg)


def _grab(source: str, input_args: List[str], duration: float, stem: str, workdir: Path, cfg: Config) -> List[Path]:
    frames = []
    for i, at in enumerate(frame_times(duration, cfg.frames_per_video)):
        dest = workdir / f"{stem}_frame{i:02d}.jpg"
        if extract_frame(source, input_args, at, dest, cfg.frame_max_width):
            frames.append(dest)
    return frames


# --------------------------------------------------------------------------- faces


def wait_for_faces(client: ImmichClient, frame_ids: List[str], cfg: Config) -> Dict[str, List[dict]]:
    """Poll until Immich has finished detecting + recognising faces in the frames."""
    start = time.monotonic()
    last_seen, last_change = None, start
    while True:
        time.sleep(cfg.poll_interval)
        faces = {fid: client.get_faces(fid) for fid in frame_ids}
        found = [f for per_frame in faces.values() for f in per_frame]
        assigned = sum(1 for f in found if f.get("person"))
        now = time.monotonic()
        elapsed = now - start
        if (len(found), assigned) != last_seen:
            last_seen, last_change = (len(found), assigned), now
        busy = client.queues_busy()
        log.debug("  %3ds: %d faces, %d recognised, immich busy=%s", elapsed, len(found), assigned, busy)

        if elapsed >= cfg.min_wait:
            if busy is False:
                return faces
            if busy is None and found and assigned == len(found):
                return faces
            if busy is None and now - last_change >= cfg.settle_time:
                return faces
        if elapsed >= cfg.face_wait_timeout:
            log.warning("Gave up waiting for Immich after %ds; using what was found so far", elapsed)
            return faces


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
        added.append(face["person"].get("name") or person_id)
    return added


# --------------------------------------------------------------------------- run


def delete_frames(client: ImmichClient, state: State, ids: List[str]) -> None:
    if not ids:
        return
    try:
        client.delete_assets(ids)
        state.clear_pending_deletes(ids)
        log.info("Deleted %d temporary frame(s)", len(ids))
    except Exception as e:  # keep going; next run retries
        log.warning("Could not delete %d frame(s), will retry next run: %s", len(ids), e)


def process_video(client: ImmichClient, state: State, cfg: Config, album_id: Optional[str], video: dict) -> bool:
    name = video.get("originalFileName") or video["id"]
    log.info("Scanning %s (%s)", name, video["id"])
    uploaded: List[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="immich-video-faces-"))
    try:
        frames = extract_frames(client, video, workdir, cfg)
        if not frames:
            raise RuntimeError("could not extract any frame")
        log.info("  extracted %d frame(s)", len(frames))

        created_at = video.get("fileCreatedAt") or dt.datetime.now(dt.timezone.utc).isoformat()
        for frame in frames:
            asset_id, status = client.upload_frame(frame, created_at)
            if status == "duplicate" and not state.is_pending_delete(asset_id):
                # Same file already in Immich and not uploaded by us: leave it alone.
                log.warning("  %s already exists in Immich as %s, skipping it", frame.name, asset_id)
                continue
            state.add_pending_delete(asset_id, video["id"])
            uploaded.append(asset_id)
        if not uploaded:
            raise RuntimeError("no frame could be uploaded")
        if album_id:
            try:
                client.add_to_album(album_id, uploaded)
            except ImmichError as e:
                log.warning("  could not add frames to album: %s", e)

        log.info("  uploaded %d frame(s), waiting for Immich face recognition...", len(uploaded))
        frame_faces = wait_for_faces(client, uploaded, cfg)
        found = sum(len(f) for f in frame_faces.values())
        recognised = sum(1 for faces in frame_faces.values() for f in faces if f.get("person"))
        added = copy_people_to_video(client, video["id"], frame_faces)

        log.info(
            "  %d face(s) found, %d recognised, people added to video: %s",
            found, recognised, ", ".join(added) if added else "none",
        )
        state.record(video, "done", frames=len(frames), faces_found=found, people_added=len(added))
        return True
    except Exception as e:
        log.error("  failed: %s", e)
        log.debug("details", exc_info=True)
        state.record(video, "failed", error=str(e)[:500])
        return False
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if cfg.keep_frames:
            log.info("  keeping frames in Immich (keep_frames is on)")
            state.clear_pending_deletes(uploaded)
        else:
            delete_frames(client, state, uploaded)


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

    client = ImmichClient(cfg.url, cfg.api_key)
    version = client.server_version()
    owner_id = client.my_user_id()
    log.info("Connected to Immich %s", version)

    # Frames left behind by a previous run that crashed or lost the network.
    if not args.dry_run:
        delete_frames(client, state, state.pending_deletes())

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
    if not batch:
        log.info("Nothing to do")
        return 0

    album_id = None
    try:
        album_id = client.find_or_create_album(cfg.album_name)
    except ImmichError as e:
        log.warning("Could not use album %r (frames will still be processed): %s", cfg.album_name, e)

    deadline = compute_deadline(cfg.stop_after, dt.datetime.now())
    failures = 0
    for video in batch:
        if deadline and dt.datetime.now() >= deadline:
            log.info("Reached STOP_AFTER %s, stopping", cfg.stop_after.strftime("%H:%M"))
            break
        if not process_video(client, state, cfg, album_id, video):
            failures += 1
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tag people in Immich videos using sampled frames.")
    parser.add_argument("--env-file", type=Path, help=f"config file (default: {DEFAULT_ENV_FILE} if present)")
    parser.add_argument("--max-videos", type=int, help="videos to scan this run (default 3)")
    parser.add_argument("--frames", type=int, help="frames to sample per video (default 8)")
    parser.add_argument("--video-id", help="scan this one video now, even if already scanned")
    parser.add_argument("--dry-run", action="store_true", help="only list the videos the next run would scan")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    parser.add_argument("--keep-frames", action="store_true", help="don't delete frames from Immich (debugging)")
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

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (sudo apt install ffmpeg)")

    with single_instance(cfg.state_db.with_suffix(".lock")):
        return run(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
