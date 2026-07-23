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
    ]


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
