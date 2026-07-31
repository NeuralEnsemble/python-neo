"""
Tests of neo.rawio.BiocamRawIO
"""

import unittest
import pytest

import numpy as np
import h5py

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
        "biocam/raw_shape_variants/version_100.brw",
        "biocam/raw_shape_variants/version_101.brw",
    ]

    # The two files in `raw_shape_variants` hold the same recording, 64 channels over 1000 samples,
    # written in the two shapes that the BRW v3.x specification defines for `3BData/Raw`: a T x W
    # matrix for version 100 and a flat interleaved array for version 101. The value stored at a
    # given position is `sample * 64 + channel`, so no two positions share a value and a misread
    # returns a number that says where it actually came from.
    shape_variants = [
        "biocam/raw_shape_variants/version_100.brw",
        "biocam/raw_shape_variants/version_101.brw",
    ]
    variant_num_channels = 64
    variant_num_frames = 1000

    # Some of the ways that neo and SpikeInterface are known to ask for channels.
    channel_queries = [None, slice(None), slice(2, 6), [0, 1, 2], [3, 0, 2], [5], np.array([1, 4])]
    # None means the whole segment. The rest include a non-zero start, a single frame, and the
    # empty range at the end of the recording.
    frame_ranges = [(None, None), (0, 5), (7, 13), (999, 1000), (1000, 1000)]

    def test_get_analogsignal_chunk_values(self):
        """Values, shape and dtype are correct for every way of selecting channels."""
        expected_all = np.arange(self.variant_num_frames * self.variant_num_channels, dtype="uint16")
        expected_all = expected_all.reshape(self.variant_num_frames, self.variant_num_channels)

        for entity in self.shape_variants:
            reader = BiocamRawIO(filename=self.get_local_path(entity))
            reader.parse_header()
            for channel_indexes in self.channel_queries:
                for i_start, i_stop in self.frame_ranges:
                    with self.subTest(entity=entity, channel_indexes=channel_indexes, i_start=i_start, i_stop=i_stop):
                        sig = reader.get_analogsignal_chunk(
                            block_index=0,
                            seg_index=0,
                            i_start=i_start,
                            i_stop=i_stop,
                            channel_indexes=channel_indexes,
                        )
                        start = 0 if i_start is None else i_start
                        stop = self.variant_num_frames if i_stop is None else i_stop
                        columns = slice(None) if channel_indexes is None else channel_indexes
                        expected = expected_all[start:stop][:, columns]
                        assert sig.dtype == expected.dtype
                        np.testing.assert_array_equal(sig, expected)

    def test_chunked_read_matches_direct_read(self):
        """Shrinking the read budget must not change the data that comes back.

        A request for a few channels over a long stretch of a large recording is served by
        reading all the channels in blocks and copying out the requested ones, the BioCAM
        layout interleaving the channels so that they cannot be read independently.
        """
        for entity in self.entities_to_test:
            reader = BiocamRawIO(filename=self.get_local_path(entity))
            reader.parse_header()
            for channel_indexes in self.channel_queries:
                with self.subTest(entity=entity, channel_indexes=channel_indexes):
                    kwargs = dict(block_index=0, seg_index=0, i_start=0, i_stop=100, channel_indexes=channel_indexes)
                    direct = reader.get_analogsignal_chunk(**kwargs)
                    # A budget of ten frames splits the hundred frames requested above into ten
                    # blocks that the reader has to stitch back together.
                    reader.max_read_bytes_dense = 10 * reader._num_channels * np.dtype("uint16").itemsize
                    chunked = reader.get_analogsignal_chunk(**kwargs)
                    del reader.max_read_bytes_dense

                    assert chunked.dtype == direct.dtype
                    np.testing.assert_array_equal(chunked, direct)

    def test_all_channel_read_is_a_view(self):
        """Asking for every channel returns the raw read itself rather than a copy of it.

        Copying there doubles the memory of a full read and costs a pass over the data, which is
        the regression that pull request #1885 fixed.
        """
        for entity in self.shape_variants:
            with self.subTest(entity=entity):
                reader = BiocamRawIO(filename=self.get_local_path(entity))
                reader.parse_header()
                sig = reader.get_analogsignal_chunk(block_index=0, seg_index=0, channel_indexes=None)
                assert sig.base is not None


def test_biocamrawio_gain(tmp_path):
    """Test that BiocamRawIO correctly reads the gain from a Biocam HDF5 file.

    A test case, from Issue #1883 (https://github.com/NeuralEnsemble/python-neo/issues/1883).
    Previously, BiocamRawIO would return a gain of `inf`, due to a numpy dtype
    overflow bug.
    """
    # Setup
    n_ch = 4
    n_frames = 10
    path = tmp_path / "minimal_v3.brw"
    bit_depth = 12
    max_volt = 4125.0
    min_volt = -4125.0
    with h5py.File(path, "w") as f:
        rv = f.create_group("3BRecInfo/3BRecVars")
        rv.create_dataset("BitDepth", data=np.array([bit_depth], dtype=np.uint8))
        rv.create_dataset("MaxVolt", data=np.array([max_volt]))
        rv.create_dataset("MinVolt", data=np.array([min_volt]))
        rv.create_dataset("NRecFrames", data=np.array([n_frames], dtype=np.int64))
        rv.create_dataset("SamplingRate", data=np.array([17852.77]))
        rv.create_dataset("SignalInversion", data=np.array([1], dtype=np.int32))
        f.create_dataset("3BRecInfo/3BMeaStreams/Raw/Chs", data=np.arange(2 * n_ch, dtype=np.int32).reshape(n_ch, 2))
        f.create_dataset("3BData/Raw", data=np.zeros(n_ch * n_frames, dtype=np.uint16))
        f["3BData"].attrs["Version"] = 102

    # Test
    r = BiocamRawIO(filename=path)
    r.parse_header()
    expected_gain = (max_volt - min_volt) / 2**bit_depth  # ~ 2.014
    gain = r.header["signal_channels"]["gain"][0]
    assert expected_gain == pytest.approx(gain)


if __name__ == "__main__":
    unittest.main()
