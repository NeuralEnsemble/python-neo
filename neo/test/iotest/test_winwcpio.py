"""
Tests of neo.io.winwcpio
"""

import unittest

from neo.io import WinWcpIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestRawBinarySignalIO(BaseTestIO, unittest.TestCase, ):
    ioclass = WinWcpIO
    files_to_test = ['File_winwcp_1.wcp']
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
