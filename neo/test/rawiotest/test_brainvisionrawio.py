""" """

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neo.rawio.brainvisionrawio import BrainVisionRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestBrainVisionRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = BrainVisionRawIO
    entities_to_test = [
        "brainvision/File_brainvision_1.vhdr",
        "brainvision/File_brainvision_2.vhdr",
        "brainvision/File_brainvision_3_float32.vhdr",
        "brainvision/File_brainvision_3_int16.vhdr",
        "brainvision/File_brainvision_3_int32.vhdr",
        "brainvision/File_brainvision_4_float32.vhdr",
    ]

    entities_to_download = ["brainvision"]


class TestBrainVisionRawIOMarkers(unittest.TestCase):
    def test_marker_positions_are_one_based(self):
        vhdr = (
            "Brain Vision Data Exchange Header File Version 1.0\n"
            "[Common Infos]\n"
            "Codepage=UTF-8\n"
            "DataFile=test.eeg\n"
            "MarkerFile=test.vmrk\n"
            "DataFormat=BINARY\n"
            "DataOrientation=MULTIPLEXED\n"
            "NumberOfChannels=1\n"
            "SamplingInterval=1000\n"
            "[Binary Infos]\n"
            "BinaryFormat=INT_16\n"
            "[Channel Infos]\n"
            "Ch1=Fp1,,1,µV\n"
        )
        vmrk = (
            "Brain Vision Data Exchange Marker File, Version 1.0\n"
            "[Marker Infos]\n"
            "Mk1=New Segment,,1,1,0\n"
            "Mk2=Stimulus,S  1,11,1,0\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            (tmp_dir / "test.vhdr").write_text(vhdr, encoding="utf8")
            (tmp_dir / "test.vmrk").write_text(vmrk, encoding="utf8")
            np.arange(20, dtype="int16").tofile(tmp_dir / "test.eeg")

            reader = BrainVisionRawIO(filename=tmp_dir / "test.vhdr")
            reader.parse_header()
            event_names = list(reader.header["event_channels"]["name"])

            # position 1 is the first data point, so it is sample 0 and time 0 s
            timestamps, _, _ = reader.get_event_timestamps(event_channel_index=event_names.index("New Segment"))
            np.testing.assert_array_equal(timestamps, [0])

            # position 11 is sample 10, 10 ms into the recording at 1 kHz
            timestamps, _, labels = reader.get_event_timestamps(event_channel_index=event_names.index("Stimulus"))
            np.testing.assert_array_equal(timestamps, [10])
            np.testing.assert_array_equal(labels, ["S  1"])
            times = reader.rescale_event_timestamp(timestamps, event_channel_index=event_names.index("Stimulus"))
            np.testing.assert_allclose(times, [0.010])


if __name__ == "__main__":
    unittest.main()
