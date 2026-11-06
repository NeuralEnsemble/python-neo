""" """

import unittest

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

    def test_marker_positions_are_one_based(self):
        # SegmentDataPoints=294 in the .vhdr and a "New Segment" marker on the first sample of each of the 34 segments
        reader = BrainVisionRawIO(self.get_local_path("brainvision/File_brainvision_3_int16.vhdr"))
        reader.parse_header()
        event_names = list(reader.header["event_channels"]["name"])
        timestamps, _, _ = reader.get_event_timestamps(event_channel_index=event_names.index("New Segment"))
        np.testing.assert_array_equal(timestamps, np.arange(34) * 294)


if __name__ == "__main__":
    unittest.main()
