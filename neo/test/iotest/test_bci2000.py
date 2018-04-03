# -*- coding: utf-8 -*-
"""
Tests of neo.io.bci2000io
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import unittest

from neo.io import BCI2000IO
from neo.test.iotest.common_io_test import BaseTestIO


class TestBCI2000IO(BaseTestIO, unittest.TestCase, ):
    ioclass = BCI2000IO
    files_to_test = [
        'eeg1_1.dat',
        'eeg1_2.dat',
        'eeg1_3.dat',
    ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
