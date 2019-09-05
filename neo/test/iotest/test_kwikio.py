# -*- coding: utf-8 -*-
"""
Tests of neo.io.kwikio
"""

# needed for python 3 compatibility
from __future__ import division

import sys
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
    files_to_test = ['neo.kwik']
    files_to_download = ['neo.kwik',
                         'neo.kwx',
                         'neo.dat']


if __name__ == "__main__":
    unittest.main()
