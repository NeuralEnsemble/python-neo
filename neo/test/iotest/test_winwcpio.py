# -*- coding: utf-8 -*-
"""
Tests of neo.io.winwcpio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import WinWcpIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestRawBinarySignalIO(BaseTestIO, unittest.TestCase, ):
    ioclass = WinWcpIO
    files_to_test = ['File_winwcp_1.wcp']
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
