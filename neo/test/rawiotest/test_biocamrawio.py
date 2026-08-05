"""
Tests of neo.rawio.BiocamRawIO
"""

import json
import unittest
import pytest

import numpy as np
import h5py

from neo.core import NeoReadWriteError
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


def _write_minimal_brw4(path, n_ch, n_frames):
    """Write a minimal uncompressed BRW 4.x file whose ``Raw`` dataset is flat and frame-major.

    Returns the reference (n_frames, n_ch) view of the samples that were written.
    """
    experiment_settings = {
        "ValueConverter": {
            "MaxAnalogValue": 4125.0,
            "MinAnalogValue": -4125.0,
            "MaxDigitalValue": 4096,
            "MinDigitalValue": 0,
            "ScaleFactor": 1.0,
        },
        "TimeConverter": {"FrameRate": 19753.775},
    }
    raw = np.arange(n_ch * n_frames, dtype=np.uint16)
    with h5py.File(path, "w") as f:
        f.create_dataset("ExperimentSettings", data=[json.dumps(experiment_settings).encode()])
        f.create_dataset("TOC", data=np.array([[0, n_frames]], dtype=np.int64))
        well = f.create_group("Well_A1")
        well.create_dataset("StoredChIdxs", data=np.arange(n_ch, dtype=np.int32))
        well.create_dataset("Raw", data=raw)
    return raw.reshape(n_frames, n_ch)


@pytest.mark.parametrize(
    "channel_indexes",
    [
        None,
        slice(None),
        slice(0, 4),
        slice(2, None),
        slice(None, 3),
        slice(None, None, 2),
        slice(0, 0),
        slice(0, -1),
        slice(-2, None),
        slice(None, None, -1),
        [3, 1, 15],
        np.array([0, 5]),
    ],
)
def test_biocamrawio_flat_layout_channel_selection(tmp_path, channel_indexes):
    """Selecting channels from a flat (frame-major) Biocam dataset must match a plain reshape.

    A test case from Issue #1892 (https://github.com/NeuralEnsemble/python-neo/issues/1892).
    The flat branch used to expand a slice with ``range(start or 0, stop or n_ch, step or 1)``,
    which silently mishandles every slice carrying a negative or zero bound: ``slice(0, -1)``
    and ``slice(None, None, -1)`` returned no channels, ``slice(0, 0)`` returned all of them,
    and ``slice(-2, None)`` returned more columns than the file has channels.
    """
    n_ch, n_frames = 16, 40
    path = tmp_path / "minimal_v4.brw"
    reference = _write_minimal_brw4(path, n_ch, n_frames)

    reader = BiocamRawIO(filename=path)
    reader.parse_header()

    i_start, i_stop = 5, 25
    chunk = reader.get_analogsignal_chunk(
        block_index=0,
        seg_index=0,
        i_start=i_start,
        i_stop=i_stop,
        stream_index=0,
        channel_indexes=channel_indexes,
    )

    expected = reference[i_start:i_stop][:, slice(None) if channel_indexes is None else channel_indexes]
    assert chunk.shape == expected.shape
    assert np.array_equal(chunk, expected)


def test_biocamrawio_flat_layout_out_of_bounds(tmp_path):
    """A frame range past the end of a flat dataset raises instead of silently mis-shaping."""
    n_ch, n_frames = 16, 40
    path = tmp_path / "minimal_v4.brw"
    _write_minimal_brw4(path, n_ch, n_frames)

    reader = BiocamRawIO(filename=path)
    reader.parse_header()

    with pytest.raises(NeoReadWriteError):
        reader.get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            i_start=0,
            i_stop=n_frames + 3,
            stream_index=0,
            channel_indexes=None,
        )


if __name__ == "__main__":
    unittest.main()
