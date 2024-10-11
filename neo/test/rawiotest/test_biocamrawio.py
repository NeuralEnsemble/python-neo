"""
Tests of neo.rawio.BiocamRawIO
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
