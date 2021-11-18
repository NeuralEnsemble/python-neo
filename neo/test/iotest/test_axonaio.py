"""
Tests of neo.io.axonaio
"""

import unittest

from neo.io.axonaio import AxonaIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.rawiotest.test_axonarawio import TestAxonaRawIO


class TestAxonaIO(BaseTestIO, unittest.TestCase, ):
    ioclass = AxonaIO
    entities_to_download = [
        'axona'
    ]

    entities_to_test = TestAxonaRawIO.entities_to_test


if __name__ == "__main__":
    unittest.main()
