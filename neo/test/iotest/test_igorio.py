"""
Tests of neo.io.igorproio
"""

import unittest

try:
    import igor

    HAVE_IGOR = True
except ImportError:
    HAVE_IGOR = False
from neo.io.igorproio import IgorIO
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipUnless(HAVE_IGOR, "requires igor")
class TestIgorIO(BaseTestIO, unittest.TestCase):
    ioclass = IgorIO
    entities_to_download = [
        'igor'
    ]
    entities_to_test = [
        'igor/mac-version2.ibw',
        'igor/win-version2.ibw'
    ]


if __name__ == "__main__":
    unittest.main()
