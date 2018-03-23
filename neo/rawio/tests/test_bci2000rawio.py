# -*- coding: utf-8 -*-
"""
Tests of neo.rawio.bci2000rawio
"""

import unittest

from neo.rawio.bci2000rawio import BCI2000RawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestBCI2000RawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = BCI2000RawIO

    files_to_download = ['eeg1_1.dat', 'eeg1_2.dat', 'eeg1_3.dat']
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
