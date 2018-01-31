# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.winedrrawio import WinEdrRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


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
