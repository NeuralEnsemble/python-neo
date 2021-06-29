import unittest

from neo.rawio.winedrrawio import WinEdrRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestWinEdrRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = WinEdrRawIO
    entities_to_download = [
        'winedr'
    ]
    entities_to_test = [
        'winedr/File_WinEDR_1.EDR',
        'winedr/File_WinEDR_2.EDR',
        'winedr/File_WinEDR_3.EDR',
    ]


if __name__ == "__main__":
    unittest.main()
