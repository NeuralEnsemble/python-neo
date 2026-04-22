import os
import shutil
import tempfile
import unittest
from pathlib import Path

from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import numpy as np


def _synthesize_gap_fixture(src_dir, dst_dir, gap_start=450, gap_size=50):
    """
    Copy a clean OpenEphys Binary recording and introduce an aligned gap of
    ``gap_size`` samples starting at position ``gap_start`` in every continuous
    stream's ``sample_numbers.npy``. Returns the destination path.

    Fixture source files often come from datalad with read-only permissions, so
    the copy is chmodded to allow mutation.
    """
    dst_dir = Path(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    for root, _dirs, files in os.walk(dst_dir):
        os.chmod(root, 0o755)
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)

    continuous_root = next(dst_dir.rglob("continuous"))
    for stream_dir in continuous_root.iterdir():
        if not stream_dir.is_dir():
            continue
        sn_path = stream_dir / "sample_numbers.npy"
        if not sn_path.is_file():
            continue
        sn = np.load(sn_path)
        if sn.shape[0] <= gap_start:
            continue
        tail = sn[gap_start] + gap_size + np.arange(sn.shape[0] - gap_start, dtype=sn.dtype)
        new_sn = np.concatenate([sn[:gap_start], tail])
        np.save(sn_path, new_sn)
    return dst_dir


class TestOpenEphysBinaryRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = OpenEphysBinaryRawIO
    entities_to_download = ["openephysbinary"]
    entities_to_test = [
        "openephysbinary/v0.5.3_two_neuropixels_stream",
        "openephysbinary/v0.4.4.1_with_video_tracking",
        "openephysbinary/v0.5.x_two_nodes",
        "openephysbinary/v0.6.x_neuropixels_multiexp_multistream",
        "openephysbinary/v0.6.x_neuropixels_with_sync",
        "openephysbinary/v0.6.x_neuropixels_missing_folders",
        "openephysbinary/v0.6.x_onebox_neuropixels",
        "openephysbinary/neural_and_non_neural_data_mixed",
    ]

    def test_sync(self):
        with self.assertWarns(DeprecationWarning):
            rawio_with_sync = OpenEphysBinaryRawIO(
                self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=True
            )
            rawio_with_sync.parse_header()
        stream_name = [s_name for s_name in rawio_with_sync.header["signal_streams"]["name"] if "AP" in s_name][0]
        stream_index = list(rawio_with_sync.header["signal_streams"]["name"]).index(stream_name)

        # AP stream has 385 channels
        chunk = rawio_with_sync.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 385

        rawio_no_sync = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=False
        )
        rawio_no_sync.parse_header()

        # AP stream has 384 channels
        chunk = rawio_no_sync.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 384

    def test_sync_channel_access(self):
        """Test that sync channels can be accessed as separate streams when load_sync_channel=False."""
        rawio = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=False
        )
        rawio.parse_header()

        # Find sync channel streams
        sync_stream_names = [s_name for s_name in rawio.header["signal_streams"]["name"] if "SYNC" in s_name]
        assert len(sync_stream_names) > 0, "No sync channel streams found"

        # Get the stream index for the first sync channel
        sync_stream_index = list(rawio.header["signal_streams"]["name"]).index(sync_stream_names[0])

        # Check that we can access the sync channel data
        chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=sync_stream_index
        )

        # Sync channel should have only one channel
        assert chunk.shape[1] == 1, f"Expected sync channel to have 1 channel, got {chunk.shape[1]}"

    def test_no_sync(self):
        # requesting sync channel when there is none raises an error
        with self.assertRaises(ValueError):
            with self.assertWarns(DeprecationWarning):
                rawio_no_sync = OpenEphysBinaryRawIO(
                    self.get_local_path("openephysbinary/v0.6.x_neuropixels_multiexp_multistream"),
                    load_sync_channel=True,
                )
                rawio_no_sync.parse_header()

    def test_missing_folders(self):
        # missing folders should raise an error
        with self.assertWarns(UserWarning):
            rawio = OpenEphysBinaryRawIO(
                self.get_local_path("openephysbinary/v0.6.x_neuropixels_missing_folders"), load_sync_channel=False
            )
            rawio.parse_header()

    def test_multiple_ttl_events_parsing(self):
        rawio = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=False
        )
        rawio.parse_header()
        rawio.header = rawio.header
        # Testing co
        # This is the TTL events from the NI Board channel
        ttl_events = rawio._evt_streams[0][0][1]
        assert "rising" in ttl_events.keys()
        assert "labels" in ttl_events.keys()
        assert "durations" in ttl_events.keys()
        assert "timestamps" in ttl_events.keys()

        # Check that durations of different event streams are correctly parsed:
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "1"], 0.5, atol=0.001)
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "6"], 0.025, atol=0.001)
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "7"], 0.016666, atol=0.001)

    def test_separating_stream_for_non_neural_data(self):
        rawio = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/neural_and_non_neural_data_mixed"), load_sync_channel=False
        )
        rawio.parse_header()
        # Check that the non-neural data stream is correctly separated
        assert len(rawio.header["signal_streams"]["name"]) == 2
        assert rawio.header["signal_streams"]["name"].tolist() == ["Rhythm_FPGA-100.0", "Rhythm_FPGA-100.0_ADC"]

    def test_gap_default_raises(self):
        """Default behavior (gap_tolerance_ms=None) raises ValueError when gaps exist."""
        src = self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync")
        with tempfile.TemporaryDirectory() as tmp:
            dst = _synthesize_gap_fixture(src, Path(tmp) / "synth")
            rawio = OpenEphysBinaryRawIO(dirname=str(dst), load_sync_channel=False)
            with self.assertRaises(ValueError) as ctx:
                rawio.parse_header()
            msg = str(ctx.exception)
            self.assertIn("gap", msg.lower())
            self.assertIn("gap_tolerance_ms", msg)
            self.assertIn("ignore_integrity_checks", msg)

    def test_gap_tolerance_ms_segments(self):
        """A small tolerance that does not absorb the gap splits the recording in two."""
        src = self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync")
        with tempfile.TemporaryDirectory() as tmp:
            dst = _synthesize_gap_fixture(src, Path(tmp) / "synth")
            rawio = OpenEphysBinaryRawIO(
                dirname=str(dst), load_sync_channel=False, gap_tolerance_ms=0.1
            )
            rawio.parse_header()
            self.assertEqual(rawio.segment_count(0), 2)

    def test_gap_tolerance_ms_absorbs(self):
        """A large tolerance absorbs the gap back into a single segment."""
        src = self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync")
        with tempfile.TemporaryDirectory() as tmp:
            dst = _synthesize_gap_fixture(src, Path(tmp) / "synth")
            rawio = OpenEphysBinaryRawIO(
                dirname=str(dst), load_sync_channel=False, gap_tolerance_ms=10_000.0
            )
            rawio.parse_header()
            self.assertEqual(rawio.segment_count(0), 1)

    def test_ignore_integrity_checks_bypasses(self):
        """``ignore_integrity_checks=True`` loads the file with a single segment."""
        src = self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync")
        with tempfile.TemporaryDirectory() as tmp:
            dst = _synthesize_gap_fixture(src, Path(tmp) / "synth")
            rawio = OpenEphysBinaryRawIO(
                dirname=str(dst), load_sync_channel=False, ignore_integrity_checks=True
            )
            rawio.parse_header()
            self.assertEqual(rawio.segment_count(0), 1)

    def test_get_openephysbinary_timestamps(self):
        """The raw-timestamps accessor returns sliced arrays that reflect each sub-segment."""
        src = self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync")
        gap_start = 450
        gap_size = 50
        with tempfile.TemporaryDirectory() as tmp:
            dst = _synthesize_gap_fixture(
                src, Path(tmp) / "synth", gap_start=gap_start, gap_size=gap_size
            )
            rawio = OpenEphysBinaryRawIO(
                dirname=str(dst), load_sync_channel=False, gap_tolerance_ms=0.1
            )
            rawio.parse_header()
            ts_seg0 = rawio._get_openephysbinary_timestamps(0, 0, 0)
            ts_seg1 = rawio._get_openephysbinary_timestamps(0, 1, 0)
            self.assertEqual(ts_seg0.shape[0], gap_start)
            self.assertGreater(ts_seg1.shape[0], 0)
            # At the boundary, the absolute index jumps by ``gap_size + 1``: the
            # normal +1 increment plus the ``gap_size`` missing samples.
            self.assertEqual(int(ts_seg1[0]) - int(ts_seg0[-1]), gap_size + 1)

    def test_fast_path_clean_file(self):
        """A clean (gap-free) fixture takes the fast path: no detected gaps, one segment."""
        src = self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync")
        rawio = OpenEphysBinaryRawIO(dirname=src, load_sync_channel=False)
        rawio.parse_header()
        self.assertEqual(rawio.segment_count(0), 1)
        for stream_info in rawio._sig_streams[0][0].values():
            self.assertEqual(stream_info.get("detected_gaps"), [])


if __name__ == "__main__":
    unittest.main()
