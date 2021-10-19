import unittest

from neo.rawio.spike2rawio import Spike2RawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpike2RawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = Spike2RawIO
    entities_to_download = [
        'spike2'
    ]
    entities_to_test = [
        'spike2/File_spike2_1.smr',
        'spike2/File_spike2_2.smr',
        'spike2/File_spike2_3.smr',
        'spike2/130322-1LY.smr',  # this is for bug 182
        'spike2/multi_sampling.smr',  # this is for bug 466
        'spike2/Two-mice-bigfile-test000.smr',  # SONv9 file
    ]


if __name__ == "__main__":
    unittest.main()
