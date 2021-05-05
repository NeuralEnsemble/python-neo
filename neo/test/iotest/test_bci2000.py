"""
Tests of neo.io.bci2000io
"""

import unittest

from neo.io import BCI2000IO
from neo.test.iotest.common_io_test import BaseTestIO


class TestBCI2000IO(BaseTestIO, unittest.TestCase, ):
    ioclass = BCI2000IO
    entities_to_download = [
        'bci2000'
    ]
    entities_to_test = [
        'bci2000/eeg1_1.dat',
        'bci2000/eeg1_2.dat',
        'bci2000/eeg1_3.dat',
    ]


if __name__ == "__main__":
    unittest.main()
