"""A tiny in-memory stand-in for the parts of the Immich API the script uses.

Uploaded frames go through a simulated background job (face detection) that can be slow,
stuck, or failing, so the tests can check that the script waits for Immich properly."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

API_KEY = "test-key"
OWNER = "user-1"
QUEUES = ("sidecar", "metadataExtraction", "storageTemplateMigration", "thumbnailGeneration", "faceDetection",
          "facialRecognition", "smartSearch")


class FakeImmich:
    def __init__(self, videos: dict[str, Path]):
        self.videos = {}
        for i, (vid, path) in enumerate(videos.items()):
            self.videos[vid] = {
                "id": vid,
                "type": "VIDEO",
                "ownerId": OWNER,
                "originalFileName": path.name,
                "fileCreatedAt": f"2024-01-0{i + 1}T10:00:00.000Z",
                # Immich v3 sends milliseconds, older versions send text
                "duration": 10000 if i % 2 == 0 else "0:00:10.000000",
                "visibility": "timeline",
                "checksum": vid,
            }
        self.files = videos
        self.uploads: dict[str, dict] = {}  # frame asset id -> info
        self.deleted: list[str] = []
        self.created_faces: list[dict] = []
        self.albums: dict[str, dict] = {}  # id -> {"name", "description", "ids"}
        self.deleted_albums: list[str] = []
        self.video_faces: dict[str, list[dict]] = {}
        self.library: dict[str, dict] = {}  # the user's own photos (must never be deleted)
        # background job simulation
        self.processing_delay = 0.0  # seconds before a frame's faces are ready
        self.stuck = False  # jobs never finish
        self.failing = False  # face detection jobs fail (no faces)
        self.legacy = False  # old Immich: no /queues/{name}/jobs endpoint
        self.job_listing_calls = 0

    # --- simulation helpers
    def ready(self, aid: str) -> bool:
        if self.stuck or self.failing:
            return False
        return time.monotonic() >= self.uploads[aid]["ready_at"]

    def pending_frames(self) -> list[str]:
        return [a for a in self.uploads if a not in self.deleted and not self.failing and not self.ready(a)]

    def faces_for_frame(self, aid: str) -> list[dict]:
        if not self.ready(aid):
            return []
        index = self.uploads[aid]["index"]
        # every frame shows Alice; frame 1 also shows Bob (big); plus an unknown face
        faces = [_face(f"{aid}-0", "p-alice", "Alice", 10, 10, 60, 60)]
        if index == 1:
            faces.append(_face(f"{aid}-1", "p-bob", "Bob", 100, 100, 300, 300))
        faces.append(_face(f"{aid}-2", None, None, 0, 0, 5, 5))
        return faces

    def start(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, body=None):
                data = b"" if body is None else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n)

            def _auth(self):
                if self.headers.get("x-api-key") != API_KEY:
                    self._json(401, {"message": "bad key"})
                    return False
                return True

            def _asset(self, aid):
                if aid in fake.deleted:
                    return None
                if aid in fake.uploads:
                    u = fake.uploads[aid]
                    return {"id": aid, "type": "IMAGE", "originalFileName": u["name"], "checksum": u["checksum"],
                            "ownerId": u.get("ownerId", OWNER), "visibility": "archive"}
                return fake.videos.get(aid) or fake.library.get(aid)

            def _album(self, album_id):
                a = fake.albums[album_id]
                count = sum(1 for i in a["ids"] if self._asset(i) is not None)
                return {"id": album_id, "albumName": a["name"], "description": a["description"], "assetCount": count}

            def do_GET(self):
                if not self._auth():
                    return
                url = urlparse(self.path)
                p = url.path
                if p == "/api/server/version":
                    return self._json(200, {"major": 3, "minor": 3, "patch": 6})
                if p == "/api/users/me":
                    return self._json(200, {"id": OWNER})
                if p == "/api/albums":
                    return self._json(200, [self._album(k) for k in fake.albums])
                m = re.fullmatch(r"/api/albums/([^/]+)", p)
                if m:
                    if m.group(1) not in fake.albums:
                        return self._json(400, {"message": "Not found or no album.read access"})
                    return self._json(200, self._album(m.group(1)))
                if p == "/api/jobs":
                    counts = {q: {"jobCounts": {"active": 0, "waiting": 0, "delayed": 0, "paused": 0}} for q in QUEUES}
                    counts["faceDetection"]["jobCounts"]["waiting"] = len(fake.pending_frames())
                    return self._json(200, counts)
                m = re.fullmatch(r"/api/queues/([^/]+)/jobs", p)
                if m:
                    if fake.legacy:
                        return self._json(404, {"message": "Not found"})
                    fake.job_listing_calls += 1
                    statuses = parse_qs(url.query).get("status", [])
                    if len(statuses) < 2:  # like the real API: a single value is not a list
                        return self._json(400, {"message": "Validation failed"})
                    jobs = []
                    if m.group(1) == "faceDetection":
                        if "waiting" in statuses:
                            jobs += [{"name": "AssetDetectFaces", "data": {"id": a}} for a in fake.pending_frames()]
                        if "failed" in statuses and fake.failing:
                            jobs += [{"name": "AssetDetectFaces", "data": {"id": a}} for a in fake.uploads]
                    return self._json(200, jobs)
                if p == "/api/faces":
                    aid = parse_qs(url.query)["id"][0]
                    if aid in fake.uploads:
                        return self._json(200, fake.faces_for_frame(aid))
                    return self._json(200, fake.video_faces.get(aid, []))
                m = re.fullmatch(r"/api/assets/([^/]+)/video/playback", p)
                if m:
                    data = fake.files[m.group(1)].read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                m = re.fullmatch(r"/api/assets/([^/]+)", p)
                if m:
                    asset = self._asset(m.group(1))
                    if asset is None:
                        return self._json(400, {"message": "Not found or no asset.read access"})
                    return self._json(200, asset)
                self._json(404, {"message": "not found " + p})

            def do_POST(self):
                if not self._auth():
                    return
                p = urlparse(self.path).path
                raw = self._body()
                if p == "/api/search/metadata":
                    body = json.loads(raw)
                    items = list(fake.videos.values()) if body.get("type") == "VIDEO" else []
                    return self._json(200, {"assets": {"items": items, "nextPage": None}, "albums": {}})
                if p == "/api/assets":
                    name = re.search(rb'filename="([^"]+)"', raw).group(1).decode()
                    video = name.rsplit("_t", 1)[0]
                    index = sum(1 for u in fake.uploads.values() if u["name"].rsplit("_t", 1)[0] == video)
                    assert b'name="visibility"' in raw and b"archive" in raw
                    boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
                    part = next(x for x in raw.split(b"--" + boundary) if b'name="assetData"' in x)
                    content = part.split(b"\r\n\r\n", 1)[1][:-2]
                    aid = str(uuid.uuid4())
                    fake.uploads[aid] = {"name": name, "index": index, "size": len(raw),
                                         "checksum": base64.b64encode(hashlib.sha1(content).digest()).decode(),
                                         "ready_at": time.monotonic() + fake.processing_delay}
                    return self._json(201, {"id": aid, "status": "created"})
                if p == "/api/albums":
                    body = json.loads(raw)
                    album_id = str(uuid.uuid4())
                    fake.albums[album_id] = {"name": body["albumName"], "description": body.get("description", ""),
                                             "ids": []}
                    return self._json(201, {"id": album_id})
                if p == "/api/faces":
                    fake.created_faces.append(json.loads(raw))
                    return self._json(201)
                self._json(404, {"message": "not found " + p})

            def do_PUT(self):
                if not self._auth():
                    return
                p = urlparse(self.path).path
                body = json.loads(self._body())
                m = re.fullmatch(r"/api/albums/([^/]+)/assets", p)
                if m:
                    fake.albums[m.group(1)]["ids"].extend(body["ids"])
                    return self._json(200, [])
                self._json(404, {"message": "not found " + p})

            def do_DELETE(self):
                if not self._auth():
                    return
                p = urlparse(self.path).path
                m = re.fullmatch(r"/api/albums/([^/]+)", p)
                if m:
                    album = fake.albums.pop(m.group(1))
                    fake.deleted_albums.append(album["name"])
                    return self._json(204)
                body = json.loads(self._body())
                assert body["force"] is True
                fake.deleted.extend(body["ids"])
                for album in fake.albums.values():
                    album["ids"] = [i for i in album["ids"] if i not in body["ids"]]
                return self._json(204)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()


def _face(face_id, pid, name, x1, y1, x2, y2):
    return {
        "id": face_id,
        "imageWidth": 640,
        "imageHeight": 360,
        "boundingBoxX1": x1,
        "boundingBoxY1": y1,
        "boundingBoxX2": x2,
        "boundingBoxY2": y2,
        "person": {"id": pid, "name": name} if pid else None,
    }
