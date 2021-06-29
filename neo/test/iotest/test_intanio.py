"""
Tests of neo.io.intanio
"""

import unittest

from neo.io import IntanIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestIntanIO(BaseTestIO, unittest.TestCase, ):
    ioclass = IntanIO
    entities_to_download = [
        'intan'
    ]
    entities_to_test = [
        'intan/intan_rhs_test_1.rhs',
        'intan/intan_rhd_test_1.rhd',
    ]


if __name__ == "__main__":
    unittest.main()
