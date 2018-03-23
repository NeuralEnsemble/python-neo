# -*- coding: utf-8 -*-
"""
Tests of neo.io.elphyio
"""

# needed for python 3 compatibility
from __future__ import division

import sys

import unittest

try:
    from neo.io import ElphyIO
except ImportError:
    ElphyIO = None
from neo.test.iotest.common_io_test import BaseTestIO


# @unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
@unittest.skip("ElphyIO not yet updated to the new API")
class TestElphyIO(BaseTestIO, unittest.TestCase):
    ioclass = ElphyIO
    files_to_test = ['ElphyExample.DAT',
                     'ElphyExample_Mode1.dat',
                     'ElphyExample_Mode2.dat',
                     'ElphyExample_Mode3.dat',
                     ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
