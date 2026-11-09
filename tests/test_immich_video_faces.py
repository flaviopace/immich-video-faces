import datetime as dt
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
    import cv2
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

    def test_frame_name(self):
        self.assertEqual(ivf.frame_name("20261108_180800", 10), "20261108_180800_t010.0s.jpg")

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
        self.assertEqual(len(ivf.select_frames(frames, 6)), 6)
        self.assertEqual(ivf.select_frames([frame(i) for i in range(24)], 6), [])


@unittest.skipUnless(HAVE_CV2, "OpenCV not installed")
class FaceFinderTests(unittest.TestCase):
    def test_blank_and_noise_frames_have_no_faces(self):
        finder = ivf.FaceFinder(min_face_px=20)
        tmp = Path(tempfile.mkdtemp())
        try:
            for name, img in {
                "black.jpg": np.zeros((810, 1440, 3), np.uint8),
                "gray.jpg": np.full((1440, 810, 3), 128, np.uint8),
                "noise.jpg": np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            }.items():
                cv2.imwrite(str(tmp / name), img)
                f = ivf.Frame(0, tmp / name)
                finder.check(f)
                self.assertEqual(f.faces, [], name)
            missing = ivf.Frame(0, tmp / "missing.jpg")
            finder.check(missing)
            self.assertEqual(missing.faces, [])
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
            {"id": "waiting", "fileCreatedAt": "2025-02-01", "ownerId": "me"},
            {"id": "motion", "visibility": "hidden", "ownerId": "me"},
            {"id": "partner", "ownerId": "someone-else"},
        ]
        by_id = {v["id"]: v for v in videos}
        self.state.record(by_id["done"], "done")
        self.state.record(by_id["retry"], "failed", error="x")
        self.state.record(by_id["waiting"], "waiting")
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
        self.state.add_pending_delete("f1", "v1")
        self.state.add_pending_delete("f2", "v2")
        self.assertTrue(self.state.is_pending_delete("f1"))
        self.assertEqual(self.state.pending_deletes("v2"), ["f2"])
        self.assertEqual(self.state.pending_deletes(skip_videos=["v2"]), ["f1"])
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
                     faces_found=3, faces_recognised=1, people_added=1, people_names=["Alice"])
        self.assertEqual(state.get("w")["people_names"], "Alice")
        self.assertEqual(state.get("w")["faces_recognised"], 1)
        self.assertEqual(state.get("v")["status"], "done")

    def test_report(self):
        since = "2000-01-01T00:00:00"
        self.assertIn("did not run", ivf.build_report(self.state, since))

        run = self.state.start_run()
        self.state.record({"id": "a", "originalFileName": "a.mp4"}, "done", candidates=29, frames=3,
                          faces_found=5, faces_recognised=4, people_added=2, people_names=["Alice", "Bob"])
        self.state.record({"id": "b", "originalFileName": "b.mp4"}, "done", candidates=19, frames=0,
                          faces_found=0, people_added=0)
        self.state.record({"id": "c", "originalFileName": "c.mp4"}, "failed", error="boom")
        self.state.finish_run(run)
        self.state.set_meta("remaining", "42")
        text = ivf.build_report(self.state, since)
        for expected in ("Videos processed: 2 (+1 failed)", "Videos with faces: 1", "Frames checked on the Pi: 48",
                         "Frames uploaded to Immich: 3", "Faces detected by Immich: 5 (4 recognised)",
                         "People added to videos: 2", "a.mp4: Alice, Bob", "c.mp4: boom",
                         "All temporary frames and albums deleted", "Videos left to scan: 42"):
            self.assertIn(expected, text)
        self.assertNotIn("did not run", text)

        self.state.add_pending_delete("f1", "a")
        self.assertIn("1 frame(s) still in Immich", ivf.build_report(self.state, since))
        self.state.record({"id": "d", "originalFileName": "d.mp4"}, "waiting")
        self.state.add_pending_delete("f2", "d")
        text = ivf.build_report(self.state, since)
        self.assertIn("1 video(s) still being processed by Immich", text)
        self.assertIn("1 frame(s) still in Immich", text)  # frames of the waiting video are not "leftovers"


class FakeFinder:
    """Pretends faces are visible at second 3 and 7 of every video."""

    def check(self, frame):
        if round(frame.at) in (3, 7):
            frame.faces = [5000.0]
            frame.boxes = [(10, 10, 70, 70, 0.99, True)]
        else:
            frame.boxes = [(5, 5, 12, 12, 0.6, False)]


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
        self.fake.video_faces["vid-a"] = [{"id": "x", "person": {"id": "p-alice", "name": "Alice"}}]
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

        def fast_init(cfg, *a, **kw):  # short waits in tests
            orig(cfg, *a, **kw)
            cfg.poll_interval = 0.05
            cfg.min_wait = 0

        keys = ("IMMICH_URL", "IMMICH_API_KEY", "STATE_DB", "FACE_FILTER", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                "FACE_WAIT_TIMEOUT", "DEBUG_DIR")
        with mock.patch.object(ivf.Config, "__init__", fast_init), mock.patch.dict(ivf.os.environ):
            for key in keys:
                ivf.os.environ.pop(key, None)
            return ivf.main(["--env-file", str(env_file), *extra])

    def state(self):
        return ivf.State(self.tmp / "state.db")

    def assert_cleaned_up(self):
        self.assertEqual(sorted(self.fake.deleted), sorted(self.fake.uploads))
        self.assertEqual(self.fake.albums, {})  # per-video albums removed
        self.assertEqual(self.state().pending_deletes(), [])

    def people_on(self, vid):
        return sorted(f["personId"] for f in self.fake.created_faces if f["assetId"] == vid)

    def test_full_run_without_face_check(self):
        self.assertEqual(self.run_script("--frames", "4", env="FACE_FILTER=off\n"), 0)

        # 3 videos per run, newest first -> d, c, b; each gets Alice + Bob once
        for vid in ("vid-b", "vid-c", "vid-d"):
            self.assertEqual(self.people_on(vid), ["p-alice", "p-bob"])
        self.assertEqual(self.people_on("vid-a"), [])
        bob = next(f for f in self.fake.created_faces if f["personId"] == "p-bob")
        self.assertEqual((bob["x"], bob["y"], bob["width"], bob["height"]), (100, 100, 200, 200))

        # 4 frames per video; one album per video, named with date and video, deleted afterwards
        self.assertEqual(len(self.fake.uploads), 12)
        self.assertEqual(len(self.fake.deleted_albums), 3)
        for name in self.fake.deleted_albums:
            self.assertRegex(name, r"^Video faces \d{4}-\d\d-\d\d \d\d:\d\d vid-[bcd]\.mp4$")
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
        row = self.state().get("vid-d")
        self.assertEqual(row["status"], "done")
        self.assertGreaterEqual(row["candidates"], 19)  # one frame every 0.5s of the 10s video
        self.assertEqual(row["frames"], 0)

    def test_only_frames_with_faces_are_uploaded(self):
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script(), 0)
        names = sorted(u["name"] for u in self.fake.uploads.values())
        expected = [f"{v}_t{t}s.jpg" for v in ("vid-b", "vid-c", "vid-d") for t in ("003.0", "007.0")]
        self.assertEqual(names, expected)
        self.assert_cleaned_up()
        row = self.state().get("vid-d")
        self.assertEqual((row["frames"], row["people_names"], row["faces_recognised"]), (2, "Alice, Bob", 3))

    def test_waits_for_immich_background_jobs(self):
        # Immich needs a moment to detect faces: the script must wait for the job, not read too early
        self.fake.processing_delay = 0.6
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1"), 0)
        self.assertEqual(self.people_on("vid-d"), ["p-alice", "p-bob"])
        self.assertGreater(self.fake.job_listing_calls, 0)  # it looked at Immich's queued jobs
        self.assert_cleaned_up()

    def test_old_immich_without_job_listing(self):
        # before v2.4 Immich can't list jobs: wait for its queues to be empty instead
        self.fake.processing_delay = 0.6
        self.fake.legacy = True
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1"), 0)
        self.assertEqual(self.people_on("vid-d"), ["p-alice", "p-bob"])
        self.assert_cleaned_up()

    def test_immich_still_busy_results_collected_next_run(self):
        self.fake.stuck = True
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "2", env="FACE_WAIT_TIMEOUT=1\n"), 0)
        # nothing tagged yet, frames and albums kept, videos marked waiting
        self.assertEqual(self.fake.created_faces, [])
        self.assertEqual(self.fake.deleted, [])
        self.assertEqual(len(self.fake.albums), 2)
        self.assertEqual({r["asset_id"] for r in self.state().waiting_videos()}, {"vid-c", "vid-d"})
        report = ivf.build_report(self.state(), "2000-01-01T00:00:00")
        self.assertIn("2 video(s) still being processed by Immich", report)

        # Immich catches up; the next run collects the results first, then cleans up
        self.fake.stuck = False
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1"), 0)
        for vid in ("vid-b", "vid-c", "vid-d"):
            self.assertEqual(self.people_on(vid), ["p-alice", "p-bob"])
        self.assertEqual(self.state().waiting_videos(), [])
        self.assertEqual(self.state().get("vid-d")["people_names"], "Alice, Bob")
        self.assert_cleaned_up()

    def test_failed_immich_job_is_retried_not_marked_empty(self):
        self.fake.failing = True
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1"), 1)
        row = self.state().get("vid-d")
        self.assertEqual(row["status"], "failed")
        self.assertIn("Immich failed", row["last_error"])
        self.assert_cleaned_up()

    def test_keep_frames_until_next_run(self):
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1", "--keep-frames"), 0)
        self.assertEqual(self.people_on("vid-d"), ["p-alice", "p-bob"])
        self.assertEqual(self.fake.deleted, [])
        [album] = self.fake.albums.values()
        self.assertTrue(album["name"].endswith("vid-d.mp4"))
        self.assertEqual(len(album["ids"]), 2)
        # the next run deletes them before doing anything else
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1"), 0)
        self.assert_cleaned_up()

    def test_debug_dir(self):
        debug = self.tmp / "debug"
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1", "--debug-dir", str(debug)), 0)
        [folder] = list(debug.iterdir())
        self.assertRegex(folder.name, r"^\d{4}-\d\d-\d\d_\d{4}_vid-d$")
        files = sorted(p.name for p in folder.iterdir())
        self.assertIn("UPLOADED_vid-d_t003.0s.jpg", files)
        self.assertIn("UPLOADED_vid-d_t007.0s.jpg", files)
        self.assertIn("vid-d_t005.0s.jpg", files)
        self.assertIn("vid-d_t005.5s.jpg", files)
        summary = (folder / "summary.txt").read_text()
        self.assertIn("Alice", summary)
        self.assertIn("people added to the video: Alice, Bob", summary)

    def add_user_photo(self, asset_id="user-photo", name="IMG_0001.jpg"):
        self.fake.library[asset_id] = {"id": asset_id, "type": "IMAGE", "originalFileName": name,
                                       "checksum": "dXNlciBwaG90bw==", "ownerId": "user-1"}

    def test_leftover_albums_cleaned_safely(self):
        self.add_user_photo()
        ours = "Temporary frames from immich-video-faces for old.mp4. Deleted automatically."
        self.fake.albums = {
            "a1": {"name": "Video faces 2026-01-01 03:00 old.mp4", "description": ours, "ids": []},
            "a2": {"name": "Video faces from my holiday", "description": "my own album", "ids": []},
            "a3": {"name": "Holiday", "description": ours, "ids": []},
            "a4": {"name": "Video faces 2026-01-01 03:05 old2.mp4", "description": ours, "ids": ["user-photo"]},
        }
        self.assertEqual(self.run_script("--max-videos", "0"), 0)
        # only the empty album made by the script is gone; a4 holds one of your photos
        self.assertEqual(sorted(self.fake.albums), ["a2", "a3", "a4"])
        self.assertNotIn("user-photo", self.fake.deleted)

    def test_never_deletes_anything_but_its_own_frames(self):
        self.add_user_photo()
        self.add_user_photo("user-frame-lookalike", "holiday_t001.0s.jpg")
        state = self.state()
        # even if the list of frames to delete were wrong, these must survive
        state.add_pending_delete("user-photo", "vid-x", "vid-x_t001.0s.jpg", "c29tZXRoaW5n")  # name differs
        state.add_pending_delete("user-frame-lookalike", "vid-x", "holiday_t001.0s.jpg", "b3RoZXI=")  # content differs
        state.add_pending_delete("vid-a", "vid-x")  # a video
        state.add_pending_delete("user-photo-2", "vid-x")  # unknown id: treated as gone
        self.add_user_photo("user-photo-3", "IMG_0003.jpg")
        state.add_pending_delete("user-photo-3", "vid-x")  # old entry without fingerprint: name must look like a frame
        self.assertEqual(self.run_script("--max-videos", "0"), 0)
        self.assertEqual(self.fake.deleted, [])
        self.assertEqual(self.state().pending_deletes(), [])  # refused entries are dropped, never retried

    def test_changed_frame_and_album_with_your_photo_are_kept(self):
        self.add_user_photo()
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "1", "--keep-frames"), 0)
        [(album_id, album)] = self.fake.albums.items()
        frame_a, frame_b = album["ids"]
        album["ids"].append("user-photo")  # you put one of your photos in the script's album
        self.fake.uploads[frame_b]["checksum"] = "Y2hhbmdlZA=="  # this frame is not what was uploaded any more
        with mock.patch.object(ivf, "load_face_finder", return_value=FakeFinder()):
            self.assertEqual(self.run_script("--max-videos", "0"), 0)
        self.assertEqual(self.fake.deleted, [frame_a])  # only the verified frame
        self.assertIn(album_id, self.fake.albums)  # album kept: it still holds photos
        self.assertEqual(self.state().albums(), [])  # and the script stops managing it

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
            self.assertIn("All temporary frames and albums deleted", text)
            # next report only covers what happened after this one
            self.run_script("--report", env="TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_CHAT_ID=42\n")
            self.assertIn("did not run", sent[1][2])

    def test_dry_run_changes_nothing(self):
        self.assertEqual(self.run_script("--dry-run"), 0)
        self.assertEqual(self.fake.uploads, {})
        self.assertEqual(self.fake.created_faces, [])


if __name__ == "__main__":
    unittest.main()
