# -*- coding: utf-8 -*-
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
    files_to_test = ['mac-version2.ibw',
                     'win-version2.ibw']
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
