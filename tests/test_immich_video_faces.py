import datetime as dt
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import immich_video_faces as ivf  # noqa: E402
from fake_immich import API_KEY, FakeImmich  # noqa: E402

try:
    import cv2  # noqa: F401
    import numpy as np

    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def frame(at, *faces):
    return ivf.Frame(at, Path(f"f{at}.jpg"), list(faces))


class HelperTests(unittest.TestCase):
    def test_parse_duration(self):
        self.assertAlmostEqual(ivf.parse_duration("0:01:23.500000"), 83.5)  # Immich v1/v2
        self.assertAlmostEqual(ivf.parse_duration(83500), 83.5)  # Immich v3: milliseconds
        self.assertEqual(ivf.parse_duration(None), 0.0)
        self.assertEqual(ivf.parse_duration("garbage"), 0.0)

    def test_frame_times(self):
        times = ivf.frame_times(100, 24)
        self.assertEqual(len(times), 24)
        self.assertGreaterEqual(times[0], 5)
        self.assertLessEqual(times[-1], 95)
        self.assertEqual(ivf.frame_times(0, 8), [0.0])
        self.assertEqual(len(ivf.frame_times(3, 24)), 6)  # max ~2 frames per second
        self.assertEqual(len(ivf.frame_times(0.3, 8)), 1)

    def test_deadline_crosses_midnight(self):
        start = dt.datetime(2026, 1, 1, 23, 0)
        self.assertEqual(ivf.compute_deadline(dt.time(6, 0), start), dt.datetime(2026, 1, 2, 6, 0))
        start = dt.datetime(2026, 1, 1, 2, 0)
        self.assertEqual(ivf.compute_deadline(dt.time(6, 0), start), dt.datetime(2026, 1, 1, 6, 0))
        self.assertIsNone(ivf.compute_deadline(None, start))

    def test_best_face_per_person(self):
        small = {"person": {"id": "a"}, "boundingBoxX1": 0, "boundingBoxY1": 0, "boundingBoxX2": 10, "boundingBoxY2": 10}
        big = {**small, "boundingBoxX2": 50, "boundingBoxY2": 50}
        nobody = {**small, "person": None}
        best = ivf.best_face_per_person({"f1": [small, nobody], "f2": [big]})
        self.assertEqual(list(best), ["a"])
        self.assertEqual(best["a"]["boundingBoxX2"], 50)

    def test_select_frames_only_faces_and_spread(self):
        frames = [frame(i) for i in range(24)]
        frames[1].faces = [100]
        frames[2].faces = [100, 50]  # same slot as 1, more faces -> wins
        frames[3].faces = [900]  # same slot, fewer faces -> loses
        frames[20].faces = [400]
        chosen = ivf.select_frames(frames, 6)
        self.assertEqual([f.at for f in chosen], [2, 20])

    def test_select_frames_limit(self):
        frames = [frame(i, 100) for i in range(24)]
        chosen = ivf.select_frames(frames, 6)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(ivf.select_frames([frame(i) for i in range(24)], 6), [])

    def test_evenly(self):
        frames = [frame(i) for i in range(10)]
        self.assertEqual([f.at for f in ivf.evenly(frames, 5)], [0, 2, 4, 6, 8])
        self.assertEqual(len(ivf.evenly(frames[:3], 5)), 3)


@unittest.skipUnless(HAVE_CV2, "OpenCV not installed")
class FaceFinderTests(unittest.TestCase):
    def test_blank_and_noise_frames_have_no_faces(self):
        finder = ivf.FaceFinder(min_face_px=48)
        tmp = Path(tempfile.mkdtemp())
        try:
            for name, img in {
                "black.jpg": np.zeros((1080, 1920, 3), np.uint8),
                "gray.jpg": np.full((1440, 1080, 3), 128, np.uint8),
                "noise.jpg": np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            }.items():
                cv2.imwrite(str(tmp / name), img)
                self.assertEqual(finder.faces(tmp / name), [], name)
            self.assertEqual(finder.faces(tmp / "missing.jpg"), [])
        finally:
            shutil.rmtree(tmp)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.state = ivf.State(self.tmp / "state.db")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pick_videos(self):
        videos = [
            {"id": "old", "fileCreatedAt": "2020-01-01", "ownerId": "me"},
            {"id": "new", "fileCreatedAt": "2024-01-01", "ownerId": "me"},
            {"id": "done", "fileCreatedAt": "2023-01-01", "ownerId": "me"},
            {"id": "retry", "fileCreatedAt": "2025-01-01", "ownerId": "me"},
            {"id": "gave-up", "fileCreatedAt": "2025-01-01", "ownerId": "me"},
            {"id": "motion", "visibility": "hidden", "ownerId": "me"},
            {"id": "partner", "ownerId": "someone-else"},
        ]
        by_id = {v["id"]: v for v in videos}
        self.state.record(by_id["done"], "done")
        self.state.record(by_id["retry"], "failed", error="x")
        for _ in range(3):
            self.state.record(by_id["gave-up"], "failed", error="x")
        picked = [v["id"] for v in ivf.pick_videos(videos, self.state, max_attempts=3, owner_id="me")]
        self.assertEqual(picked, ["new", "old", "retry"])

    def test_retry_empty(self):
        self.state.record({"id": "a"}, "done", faces_found=0)
        self.state.record({"id": "b"}, "done", faces_found=2)
        self.assertEqual(self.state.forget_empty(), 1)
        self.assertIsNone(self.state.get("a"))
        self.assertIsNotNone(self.state.get("b"))

    def test_pending_deletes(self):
        self.state.add_pending_delete("f1", "v")
        self.state.add_pending_delete("f2", "v")
        self.assertTrue(self.state.is_pending_delete("f1"))
        self.state.clear_pending_deletes(["f1"])
        self.assertEqual(self.state.pending_deletes(), ["f2"])

    def test_old_database_is_upgraded(self):
        old = self.tmp / "old.db"
        db = sqlite3.connect(old)
        db.execute("CREATE TABLE videos (asset_id TEXT PRIMARY KEY, file_name TEXT, checksum TEXT, status TEXT NOT NULL,"
                   " attempts INTEGER NOT NULL DEFAULT 0, frames INTEGER, faces_found INTEGER, people_added INTEGER,"
                   " last_error TEXT, updated_at TEXT NOT NULL)")
        db.execute("INSERT INTO videos (asset_id, status, updated_at) VALUES ('v', 'done', '2026-01-01')")
        db.commit()
        db.close()
        state = ivf.State(old)
        state.record({"id": "w", "originalFileName": "w.mp4"}, "done", candidates=24, frames=2,
                     faces_found=3, people_added=1, people_names=["Alice"])
        self.assertEqual(state.get("w")["people_names"], "Alice")
        self.assertEqual(state.get("v")["status"], "done")

    def test_report(self):
        since = "2000-01-01T00:00:00"
        self.assertIn("did not run", ivf.build_report(self.state, since))

        run = self.state.start_run()
        self.state.record({"id": "a", "originalFileName": "a.mp4"}, "done", candidates=24, frames=3,
                          faces_found=5, people_added=2, people_names=["Alice", "Bob"])
        self.state.record({"id": "b", "originalFileName": "b.mp4"}, "done", candidates=24, frames=0,
                          faces_found=0, people_added=0)
        self.state.record({"id": "c", "originalFileName": "c.mp4"}, "failed", error="boom")
        self.state.finish_run(run)
        self.state.set_meta("remaining", "42")
        text = ivf.build_report(self.state, since)
        for expected in ("Videos processed: 2 (+1 failed)", "Videos with faces: 1", "Frames checked on the Pi: 48",
                         "Frames uploaded to Immich: 3", "Faces detected by Immich: 5", "People added to videos: 2",
                         "a.mp4: Alice, Bob", "c.mp4: boom", "All temporary frames deleted", "Videos left to scan: 42"):
            self.assertIn(expected, text)
        self.assertNotIn("did not run", text)

        self.state.add_pending_delete("f1", "a")
        self.assertIn("1 frame(s) still in Immich", ivf.build_report(self.state, since))


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.videos = {}
        for vid in ("vid-a", "vid-b", "vid-c", "vid-d"):
            path = self.tmp / f"{vid}.mp4"
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=10:size=640x360:rate=10",
                 "-pix_fmt", "yuv420p", str(path)],
                check=True,
            )
            self.videos[vid] = path
        self.fake = FakeImmich(self.videos)
        # vid-a already shows Alice (from its thumbnail): only Bob should be added
        self.fake.video_faces["vid-a"] = [{"person": {"id": "p-alice", "name": "Alice"}}]
        self.url = self.fake.start()

    def tearDown(self):
        self.fake.stop()
        shutil.rmtree(self.tmp)

    def run_script(self, *extra, env=""):
        env_file = self.tmp / "test.env"
        env_file.write_text(
            f"IMMICH_URL={self.url}\nIMMICH_API_KEY={API_KEY}\nSTATE_DB={self.tmp / 'state.db'}\n{env}"
        )
        orig = ivf.Config.__init__

        def fast_init(cfg, *a, **kw):  # no real waiting in tests
            orig(cfg, *a, **kw)
            cfg.poll_interval = 0
            cfg.min_wait = 0

        keys = ("IMMICH_URL", "IMMICH_API_KEY", "STATE_DB", "FACE_FILTER", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
        with mock.patch.object(ivf.Config, "__init__", fast_init), mock.patch.dict(ivf.os.environ):
            for key in keys:
                ivf.os.environ.pop(key, None)
            return ivf.main(["--env-file", str(env_file), *extra])

    def assert_cleaned_up(self):
        self.assertEqual(sorted(self.fake.deleted), sorted(self.fake.uploads))
        self.assertEqual(self.fake.albums, {})  # temporary album removed
        state = ivf.State(self.tmp / "state.db")
        self.assertEqual(state.pending_deletes(), [])

    def test_full_run_without_face_check(self):
        self.assertEqual(self.run_script("--frames", "4", env="FACE_FILTER=off\n"), 0)

        # 3 videos per run, newest first -> d, c, b
        tagged = sorted({f["assetId"] for f in self.fake.created_faces})
        self.assertEqual(tagged, ["vid-b", "vid-c", "vid-d"])
        # each video gets Alice + Bob once; Bob using his biggest face box
        for vid in tagged:
            people = sorted(f["personId"] for f in self.fake.created_faces if f["assetId"] == vid)
            self.assertEqual(people, ["p-alice", "p-bob"])
        bob = next(f for f in self.fake.created_faces if f["personId"] == "p-bob")
        self.assertEqual((bob["x"], bob["y"], bob["width"], bob["height"]), (100, 100, 200, 200))

        # 4 frames per video, all deleted afterwards, album gone
        self.assertEqual(len(self.fake.uploads), 12)
        self.assertEqual(self.fake.deleted_albums, ["Video face scan (temp)"])
        self.assert_cleaned_up()

        # second run: only the remaining video, and Alice is not duplicated on it
        self.fake.created_faces.clear()
        self.assertEqual(self.run_script("--frames", "4", env="FACE_FILTER=off\n"), 0)
        self.assertEqual([(f["assetId"], f["personId"]) for f in self.fake.created_faces], [("vid-a", "p-bob")])
        self.assert_cleaned_up()

        # third run: nothing left
        self.fake.created_faces.clear()
        uploads = len(self.fake.uploads)
        self.assertEqual(self.run_script(env="FACE_FILTER=off\n"), 0)
        self.assertEqual(len(self.fake.uploads), uploads)
        self.assertEqual(self.fake.created_faces, [])

    @unittest.skipUnless(HAVE_CV2, "OpenCV not installed")
    def test_videos_without_faces_upload_nothing(self):
        # the ffmpeg test pattern has no faces: the Pi check must stop every upload
        self.assertEqual(self.run_script(), 0)
        self.assertEqual(self.fake.uploads, {})
        self.assertEqual(self.fake.created_faces, [])
        self.assertEqual(self.fake.albums, {})  # album never even created
        state = ivf.State(self.tmp / "state.db")
        row = state.get("vid-d")
        self.assertEqual((row["status"], row["candidates"], row["frames"]), ("done", 20, 0))

    def test_only_frames_with_faces_are_uploaded(self):
        class FakeFinder:
            def faces(self, path):  # faces only in candidate frames 3 and 15
                index = int(re.search(r"frame(\d+)", path.name).group(1))
                return [5000.0] if index in (3, 15) else []

        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script(), 0)
        names = sorted(u["name"] for u in self.fake.uploads.values())
        self.assertEqual(names, [f"{v}_frame{i}.jpg" for v in ("vid-b", "vid-c", "vid-d") for i in ("03", "15")])
        self.assert_cleaned_up()

        state = ivf.State(self.tmp / "state.db")
        row = state.get("vid-d")
        self.assertEqual((row["candidates"], row["frames"], row["people_names"]), (20, 2, "Alice, Bob"))

    def test_report_to_telegram(self):
        self.assertEqual(self.run_script("--frames", "2", env="FACE_FILTER=off\n"), 0)
        sent = []
        with mock.patch.object(ivf, "send_telegram", side_effect=lambda t, c, text: sent.append((t, c, text))):
            self.run_script("--report", env="TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_CHAT_ID=42\n")
            self.assertEqual(len(sent), 1)
            token, chat, text = sent[0]
            self.assertEqual((token, chat), ("123:abc", "42"))
            self.assertIn("Videos processed: 3", text)
            self.assertIn("Videos left to scan: 1", text)
            self.assertIn("All temporary frames deleted", text)
            # next report only covers what happened after this one
            self.run_script("--report", env="TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_CHAT_ID=42\n")
            self.assertIn("did not run", sent[1][2])

    def test_dry_run_changes_nothing(self):
        self.assertEqual(self.run_script("--dry-run"), 0)
        self.assertEqual(self.fake.uploads, {})
        self.assertEqual(self.fake.created_faces, [])


if __name__ == "__main__":
    unittest.main()
