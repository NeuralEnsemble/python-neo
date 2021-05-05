"""
Tests of neo.io.winwcpio
"""

import unittest

from neo.io import WinWcpIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestRawBinarySignalIO(BaseTestIO, unittest.TestCase, ):
    ioclass = WinWcpIO
    entities_to_download = [
        'winwcp'
    ]
    entities_to_test = [
        'winwcp/File_winwcp_1.wcp'
    ]


if __name__ == "__main__":
    unittest.main()
