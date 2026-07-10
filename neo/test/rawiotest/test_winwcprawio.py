import unittest

from neo.rawio.winwcprawio import WinWcpRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestWinWcpRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = WinWcpRawIO
    entities_to_download = ["winwcp"]
    entities_to_test = [
        "winwcp/File_winwcp_1.wcp",
        "winwcp/file_with_recording_time/File_winwcp_2.wcp",
    ]


if __name__ == "__main__":
    unittest.main()
