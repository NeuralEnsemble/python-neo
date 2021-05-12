"""
Tests of neo.io.kwikio
"""

import unittest

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
from neo.io import kwikio
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipUnless(HAVE_H5PY, "requires h5py")
@unittest.skipUnless(kwikio.HAVE_KWIK, "requires klusta")
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
