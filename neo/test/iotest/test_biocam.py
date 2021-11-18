"""
Tests of neo.io.BiocamIO
"""

import unittest

from neo.io import BiocamIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestBiocamIO(BaseTestIO, unittest.TestCase, ):
    ioclass = BiocamIO
    entities_to_download = [
        'biocam'
    ]
    entities_to_test = [
        'biocam/biocam_hw3.0_fw1.6.brw'
    ]


if __name__ == "__main__":
    unittest.main()
