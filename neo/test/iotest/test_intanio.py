"""
Tests of neo.io.intanio
"""

import unittest

from neo.io import IntanIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestIntanIO(BaseTestIO, unittest.TestCase, ):
    ioclass = IntanIO
    files_to_download = [
        'intan_rhs_test_1.rhs',
        'intan_rhd_test_1.rhd',
    ]
    files_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
