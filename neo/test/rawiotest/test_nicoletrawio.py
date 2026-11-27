import unittest

from neo.rawio.nicoletrawio import NicoletRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNicoletRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = NicoletRawIO
    entities_to_download = [
        "nicolet/scalp_eeg/scalp_eeg.e",
        "nicolet/multi_segment/multi_segment.e",
        "nicolet/intracranial_high_rate/intracranial_high_rate.e",
    ]

    entities_to_test = [
        "nicolet/scalp_eeg/scalp_eeg.e",
        "nicolet/multi_segment/multi_segment.e",
        "nicolet/intracranial_high_rate/intracranial_high_rate.e",
    ]


if __name__ == "__main__":
    unittest.main()
