"""A tiny in-memory stand-in for the parts of the Immich API the script uses."""
from __future__ import annotations

import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

API_KEY = "test-key"
OWNER = "user-1"


class FakeImmich:
    def __init__(self, videos: dict[str, Path]):
        self.videos = {
            vid: {
                "id": vid,
                "type": "VIDEO",
                "ownerId": OWNER,
                "originalFileName": path.name,
                "fileCreatedAt": f"2024-01-0{i + 1}T10:00:00.000Z",
                "duration": "0:00:10.000000",
                "visibility": "timeline",
                "checksum": vid,
            }
            for i, (vid, path) in enumerate(videos.items())
        }
        self.files = videos
        self.uploads: dict[str, dict] = {}  # frame asset id -> form info
        self.deleted: list[str] = []
        self.created_faces: list[dict] = []
        self.albums: dict[str, dict] = {}  # id -> {"name", "ids"}
        # frames recognise "Alice" in every frame, and "Bob" (bigger) in the 2nd one
        self.video_faces: dict[str, list[dict]] = {}

    def faces_for_frame(self, index: int) -> list[dict]:
        faces = [_face("p-alice", "Alice", 10, 10, 60, 60)]
        if index == 1:
            faces.append(_face("p-bob", "Bob", 100, 100, 300, 300))
        faces.append(_face(None, None, 0, 0, 5, 5))  # unknown person
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

            def do_GET(self):
                if not self._auth():
                    return
                url = urlparse(self.path)
                p = url.path
                if p == "/api/server/version":
                    return self._json(200, {"major": 2, "minor": 3, "patch": 1})
                if p == "/api/users/me":
                    return self._json(200, {"id": OWNER})
                if p == "/api/albums":
                    return self._json(200, [{"id": k, "albumName": a["name"]} for k, a in fake.albums.items()])
                if p == "/api/jobs":
                    return self._json(200, {q: {"jobCounts": {"active": 0, "waiting": 0, "delayed": 0}}
                                            for q in ("faceDetection", "facialRecognition")})
                if p == "/api/faces":
                    aid = parse_qs(url.query)["id"][0]
                    if aid in fake.uploads:
                        return self._json(200, fake.faces_for_frame(fake.uploads[aid]["index"]))
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
                if m and m.group(1) in fake.videos:
                    return self._json(200, fake.videos[m.group(1)])
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
                    index = int(re.search(r"frame(\d+)", name).group(1))
                    assert b'name="visibility"' in raw and b"archive" in raw
                    aid = str(uuid.uuid4())
                    fake.uploads[aid] = {"name": name, "index": index, "size": len(raw)}
                    return self._json(201, {"id": aid, "status": "created"})
                if p == "/api/albums":
                    body = json.loads(raw)
                    album_id = str(uuid.uuid4())
                    fake.albums[album_id] = {"name": body["albumName"], "ids": []}
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
                body = json.loads(self._body())
                assert body["force"] is True
                fake.deleted.extend(body["ids"])
                return self._json(204)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()


def _face(pid, name, x1, y1, x2, y2):
    return {
        "id": str(uuid.uuid4()),
        "imageWidth": 640,
        "imageHeight": 360,
        "boundingBoxX1": x1,
        "boundingBoxY1": y1,
        "boundingBoxX2": x2,
        "boundingBoxY2": y2,
        "person": {"id": pid, "name": name} if pid else None,
    }
