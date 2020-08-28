import unittest

from neo.rawio.winwcprawio import WinWcpRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestWinWcpRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = WinWcpRawIO
    entities_to_test = ['File_winwcp_1.wcp']
    files_to_download = entities_to_test


if __name__ == "__main__":
    unittest.main()
