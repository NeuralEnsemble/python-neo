"""
Tests of neo.rawio.bci2000rawio
"""

import unittest

from neo.rawio.bci2000rawio import BCI2000RawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestBCI2000RawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = BCI2000RawIO

    entities_to_download = [
        'bci2000/eeg1_1.dat',
        'bci2000/eeg1_2.dat',
        'bci2000/eeg1_3.dat']

    entities_to_download = [
        'bci2000',
    ]


if __name__ == "__main__":
    unittest.main()
