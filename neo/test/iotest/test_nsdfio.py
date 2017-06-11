# -*- coding: utf-8 -*-
"""
Tests of neo.io.NSDFIO
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io.nsdfio import HAVE_NSDF, NSDFIO
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipUnless(HAVE_NSDF, "Requires NSDF")
class CommonTests(BaseTestIO, unittest.TestCase):
    ioclass = NSDFIO
    read_and_write_is_bijective = False

if __name__ == "__main__":
    unittest.main()
