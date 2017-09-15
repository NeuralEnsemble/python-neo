# -*- coding: utf-8 -*-
"""
Tests of neo.io.neomatlabio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import sys

import unittest

from neo.io import MicromedIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestMicromedIO(BaseTestIO, unittest.TestCase, ):
    ioclass = MicromedIO
    files_to_test = ['File_micromed_1.TRC']
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
