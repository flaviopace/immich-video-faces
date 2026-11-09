import datetime as dt
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import immich_video_faces as ivf  # noqa: E402
from fake_immich import API_KEY, FakeImmich  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_parse_duration(self):
        self.assertAlmostEqual(ivf.parse_duration("0:01:23.500000"), 83.5)
        self.assertEqual(ivf.parse_duration(None), 0.0)
        self.assertEqual(ivf.parse_duration("garbage"), 0.0)

    def test_frame_times(self):
        times = ivf.frame_times(100, 8)
        self.assertEqual(len(times), 8)
        self.assertGreaterEqual(times[0], 5)
        self.assertLessEqual(times[-1], 95)
        self.assertEqual(ivf.frame_times(0, 8), [0.0])
        self.assertEqual(len(ivf.frame_times(3, 8)), 3)  # ~1 frame per second max
        self.assertEqual(len(ivf.frame_times(0.5, 8)), 1)

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

    def run_script(self, *extra):
        env_file = self.tmp / "test.env"
        env_file.write_text(f"IMMICH_URL={self.url}\nIMMICH_API_KEY={API_KEY}\nSTATE_DB={self.tmp / 'state.db'}\n")
        # speed up the waiting loop
        orig = ivf.Config.__init__

        def fast_init(cfg, *a, **kw):
            orig(cfg, *a, **kw)
            cfg.poll_interval = 0
            cfg.min_wait = 0

        ivf.Config.__init__ = fast_init
        try:
            for key in ("IMMICH_URL", "IMMICH_API_KEY", "STATE_DB"):
                ivf.os.environ.pop(key, None)
            return ivf.main(["--env-file", str(env_file), "--frames", "4", *extra])
        finally:
            ivf.Config.__init__ = orig

    def test_full_run(self):
        self.assertEqual(self.run_script(), 0)

        # 3 videos per run, newest first -> d, c, b
        tagged = sorted({f["assetId"] for f in self.fake.created_faces})
        self.assertEqual(tagged, ["vid-b", "vid-c", "vid-d"])
        # each video gets Alice + Bob once; Bob using his biggest face box
        for vid in tagged:
            people = sorted(f["personId"] for f in self.fake.created_faces if f["assetId"] == vid)
            self.assertEqual(people, ["p-alice", "p-bob"])
        bob = next(f for f in self.fake.created_faces if f["personId"] == "p-bob")
        self.assertEqual((bob["x"], bob["y"], bob["width"], bob["height"]), (100, 100, 200, 200))

        # 4 frames per video, all uploaded into the album and then deleted
        self.assertEqual(len(self.fake.uploads), 12)
        self.assertEqual(sorted(self.fake.deleted), sorted(self.fake.uploads))
        [album] = self.fake.albums.values()
        self.assertEqual(album["name"], "Video face scan (temp)")
        self.assertEqual(len(album["ids"]), 12)

        # second run: only the remaining video, and Alice is not duplicated on it
        self.fake.created_faces.clear()
        self.assertEqual(self.run_script(), 0)
        self.assertEqual([(f["assetId"], f["personId"]) for f in self.fake.created_faces], [("vid-a", "p-bob")])

        # third run: nothing left
        self.fake.created_faces.clear()
        uploads = len(self.fake.uploads)
        self.assertEqual(self.run_script(), 0)
        self.assertEqual(len(self.fake.uploads), uploads)
        self.assertEqual(self.fake.created_faces, [])

    def test_dry_run_changes_nothing(self):
        self.assertEqual(self.run_script("--dry-run"), 0)
        self.assertEqual(self.fake.uploads, {})
        self.assertEqual(self.fake.created_faces, [])


if __name__ == "__main__":
    unittest.main()
