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
    files_to_test = ['experiment1.kwik']
    files_to_download =  ['experiment1.kwik',
                          'experiment1.kwx',
                          'experiment1_100.raw.kwd']


if __name__ == "__main__":
    unittest.main()
