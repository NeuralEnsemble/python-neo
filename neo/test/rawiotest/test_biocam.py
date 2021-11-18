"""
Tests of neo.rawio.BiocamRawIO
"""

import unittest

from neo.rawio.biocamrawio import BiocamRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestBiocamRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = BiocamRawIO

    entities_to_download = [
        'biocam/biocam_hw3.0_fw1.6.brw'
        ]

    entities_to_download = [
        'biocam',
    ]


if __name__ == "__main__":
    unittest.main()
