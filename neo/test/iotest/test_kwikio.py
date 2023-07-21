"""
Tests of neo.io.kwikio
"""

import unittest

try:
    from klusta import kwik
    HAVE_KWIK = True
except ImportError:
    HAVE_KWIK = False

from neo.io import kwikio
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipUnless(HAVE_KWIK, "requires klusta")
class TestKwikIO(BaseTestIO, unittest.TestCase):
    ioclass = kwikio.KwikIO
    entities_to_download = [
        'kwik'
    ]
    entities_to_test = [
        'kwik/neo.kwik'
    ]


if __name__ == "__main__":
    unittest.main()
