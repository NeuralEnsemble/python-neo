# -*- coding: utf-8 -*-
"""
Tests of neo.io.axographio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

import unittest

from neo.io import AxographIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestAxographIO(BaseTestIO, unittest.TestCase):
    files_to_test = [
        'File_axograph.axgd'
    ]
    files_to_download = files_to_test
    ioclass = AxographIO


if __name__ == "__main__":
    unittest.main()
