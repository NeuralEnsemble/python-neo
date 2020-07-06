import unittest

from neo.rawio.winedrrawio import WinEdrRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestWinEdrRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = WinEdrRawIO
    files_to_download = [
        'File_WinEDR_1.EDR',
        'File_WinEDR_2.EDR',
        'File_WinEDR_3.EDR',
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
