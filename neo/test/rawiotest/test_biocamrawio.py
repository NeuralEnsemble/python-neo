"""
Tests of neo.rawio.BiocamRawIO
"""

import unittest

import numpy as np
import pytest
import h5py

from neo.rawio import biocamrawio
from neo.rawio.biocamrawio import BiocamRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestBiocamRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = BiocamRawIO

    entities_to_download = [
        "biocam",
    ]
    entities_to_test = [
        "biocam/biocam_hw3.0_fw1.6.brw",
        "biocam/biocam_hw3.0_fw1.7.0.12_raw.brw",
    ]


def write_minimal_brw(path, n_ch, n_frames, version=102):
    """Write a minimal Biocam file whose samples are all distinct.

    Sample `s` of channel `c` holds the value `s * n_ch + c`, so an error in reshaping,
    channel selection or block stitching produces wrong values rather than plausible ones.
    Version 102 stores the raw data as a flat array, version 101 stores it 2-D.

    BioCam 3.x stores the 3BData group's format in its "Version" attribute.
    v100 lays Raw out as a (T, W) matrix; v101 as a flat (T*W, 1) interleaved array.

    Info on 100-102 file formats:

        https://resources.3brain.com/f1/downloads/BrainWave_Documentation_for_BRWv.%203.x_and_BXRv.%202.x_FileFormats_v1.2.0.pdf

    Returns the expected (n_frames, n_ch) contents.

    TODO: this can be shared with the parallel pull request for issue #1883.
    """
    values = np.arange(n_frames * n_ch, dtype=np.uint16)
    with h5py.File(path, "w") as f:
        rec_vars = f.create_group("3BRecInfo/3BRecVars")
        rec_vars.create_dataset("BitDepth", data=np.array([12], dtype=np.uint8))
        rec_vars.create_dataset("MaxVolt", data=np.array([4125.0]))
        rec_vars.create_dataset("MinVolt", data=np.array([-4125.0]))
        rec_vars.create_dataset("NRecFrames", data=np.array([n_frames], dtype=np.int64))
        rec_vars.create_dataset("SamplingRate", data=np.array([17852.77]))
        rec_vars.create_dataset("SignalInversion", data=np.array([1], dtype=np.int32))
        f.create_dataset("3BRecInfo/3BMeaStreams/Raw/Chs", data=np.arange(2 * n_ch, dtype=np.int32).reshape(n_ch, 2))
        f.create_dataset("3BData/Raw", data=values.reshape(n_frames, n_ch) if version == 100 else values)
        f["3BData"].attrs["Version"] = version
    frames = values.reshape(n_frames, n_ch)
    return frames


class TestGetAnalogsignalChunk:
    """Tests for BiocamRawIO._get_analogsignal_chunk.

    It's a little tricky to test this function, as what we care about is not input or output but performance.
    A reasonably effective way to inspect the behaviour is to count how many times the inner _read block is called,
    which is the function that actually reads the raw file. This is is done for the tests of this class. To do it, we
    replace the private _read_block method of BiocamRawIO with a custom method that tracks its calls.
    """

    n_ch = 8
    n_frames = 20
    # Some ways neo and SpikeInterface are known to ask for channels:
    channel_queries = [None, slice(None), slice(2, 6), slice(0, n_ch, 2), [0, 1, 2], [3, 0, 2], [5], np.array([1, 4])]
    # None means "the whole segment"; the rest include a non-zero start and a single frame.
    frame_ranges = [(None, None), (0, 5), (7, 13), (n_frames - 1, n_frames)]
    # 64 bytes is 4 frames of 8 uint16 channels, so requests split into several blocks,
    # the last of which is partial for most of the ranges above.
    small_budget = 64
    small_budget_frames = 4

    def make_reader(self, tmp_path, version=101):
        """Return a BiocamRawIO , and the expected frames."""
        path = tmp_path / "minimal.brw"
        frames = write_minimal_brw(path, self.n_ch, self.n_frames, version)
        reader = BiocamRawIO(filename=path)
        reader.parse_header()
        return reader, frames

    @pytest.fixture
    def frames_per_read(self, monkeypatch):
        """Record how many frames each raw read covers."""
        reads = []
        original = BiocamRawIO._read_block

        def spy(self, i_start, i_stop):
            reads.append(i_stop - i_start)
            return original(self, i_start, i_stop)

        monkeypatch.setattr(BiocamRawIO, "_read_block", spy)
        return reads

    @pytest.mark.parametrize("version", [100, 101])
    @pytest.mark.parametrize("channel_indexes", channel_queries)
    @pytest.mark.parametrize("i_start, i_stop", frame_ranges)
    def test_values(self, tmp_path, version, channel_indexes, i_start, i_stop):
        """Values, shape and dtype are correct for every way of selecting channels."""
        reader, expected_all = self.make_reader(tmp_path, version)

        sig = reader.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=i_start, i_stop=i_stop, channel_indexes=channel_indexes
        )

        start = 0 if i_start is None else i_start
        stop = self.n_frames if i_stop is None else i_stop
        expected = expected_all[start:stop][:, slice(None) if channel_indexes is None else channel_indexes]
        assert sig.dtype == expected.dtype
        np.testing.assert_array_equal(sig, expected)

    @pytest.mark.parametrize("channel_indexes", channel_queries)
    @pytest.mark.parametrize("i_start, i_stop", frame_ranges)
    def test_chunked_read_matches_direct_read(self, tmp_path, monkeypatch, channel_indexes, i_start, i_stop):
        """Shrinking the read budget must not change the data that comes back."""
        reader, _ = self.make_reader(tmp_path)
        kwargs = dict(block_index=0, seg_index=0, i_start=i_start, i_stop=i_stop, channel_indexes=channel_indexes)

        direct = reader.get_analogsignal_chunk(**kwargs)
        monkeypatch.setattr(biocamrawio, "_MAX_READ_BYTES", self.small_budget)
        chunked = reader.get_analogsignal_chunk(**kwargs)

        assert chunked.dtype == direct.dtype
        np.testing.assert_array_equal(chunked, direct)

    def test_read_budget(self, tmp_path, monkeypatch, frames_per_read):
        """A long request for few channels is split, so the whole recording is never resident.

        See https://github.com/SpikeInterface/spikeinterface/issues/3303
        """
        reader, expected_all = self.make_reader(tmp_path)
        # Another testing hack here, to make sure multiple reads are done.
        monkeypatch.setattr(biocamrawio, "_MAX_READ_BYTES", self.small_budget)

        sig = reader.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=self.n_frames, channel_indexes=[2]
        )

        np.testing.assert_array_equal(sig, expected_all[:, [2]])
        largest = max(frames_per_read)
        assert largest <= self.small_budget_frames, f"a single read covered {largest} frames"
        assert len(frames_per_read) > 2, "the request was not split, so the chunked path is untested"

    def test_requesting_all_channels(self, tmp_path, monkeypatch, frames_per_read):
        """Requesting every channel shouldn't cause multiple reads (nothing to be gained)."""
        reader, _ = self.make_reader(tmp_path)
        monkeypatch.setattr(biocamrawio, "_MAX_READ_BYTES", self.small_budget)

        sig = reader.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=self.n_frames, channel_indexes=None
        )

        assert max(frames_per_read) == self.n_frames, "the full-channel request was split, which only adds a copy"
        assert sig.base is not None, "the full-channel read should be a view on the raw read, not a copy"


if __name__ == "__main__":
    unittest.main()
