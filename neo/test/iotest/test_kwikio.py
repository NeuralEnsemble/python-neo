# -*- coding: utf-8 -*-
"""
Tests of neo.io.kwikio
"""

# needed for python 3 compatibility
from __future__ import division

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import KwikIO
from neo.test.iotest.common_io_test import BaseTestIO

class TestKwikIO(BaseTestIO, unittest.TestCase):
    ioclass = KwikIO
    files_to_test = ['experiment0.kwik',
                    #  'experiment0.raw.kwd',
                    #  'experiment0.kwx',
                     'test_hybrid_120sec.kwik',
                    #  'test_hybrid_120sec.raw.kwd',
                    #  'test_hybrid_120sec.kwx'
                     ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
