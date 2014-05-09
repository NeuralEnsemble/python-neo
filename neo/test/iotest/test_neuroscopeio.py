# -*- coding: utf-8 -*-
"""
Tests of neo.io.neuroscopeio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import NeuroScopeIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestNeuroScopeIO(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuroScopeIO
    files_to_test = ['test1/test1.xml']
    files_to_download = ['test1/test1.xml',
                         'test1/test1.dat',
                         ]


if __name__ == "__main__":
    unittest.main()
