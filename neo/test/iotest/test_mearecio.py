"""
Tests of neo.io.mearecio
"""

import unittest

from neo.io import MEArecIO
from neo.test.iotest.common_io_test import BaseTestIO


try:
    import MEArec as mr
    HAVE_MEAREC = True
except ImportError:
    HAVE_MEAREC = False


@unittest.skipUnless(HAVE_MEAREC, "requires MEArec package")
class TestMEArecIO(BaseTestIO, unittest.TestCase):
    entities_to_download = [
        'mearec'
    ]
    entities_to_test = [
        'mearec/mearec_test_10s.h5'
    ]
    ioclass = MEArecIO


if __name__ == "__main__":
    unittest.main()
