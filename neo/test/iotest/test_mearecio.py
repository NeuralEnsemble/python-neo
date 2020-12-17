"""
Tests of neo.io.mearecio
"""

import unittest

from neo.io import MEArecIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestMEArecIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['mearec_test_10s.h5']
    files_to_download = ['mearec_test_10s.h5']
    ioclass = MEArecIO


if __name__ == "__main__":
    unittest.main()
