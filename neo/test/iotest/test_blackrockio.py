# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import os
import struct
import sys
import tempfile

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.io.blackrockio import BlackrockIO

from neo.test.iotest.common_io_test import BaseTestIO


class CommonTests(BaseTestIO, unittest.TestCase):
    ioclass =BlackrockIO

    files_to_test = [
        #'test2/test.ns5'
        ]

    files_to_download = [
        #'test2/test.ns5'
        ]

    files_to_test = ['FileSpec2.3001']

    files_to_download = ['FileSpec2.3001.nev',
                     'FileSpec2.3001.ns5',
                     'FileSpec2.3001.ccf',
                     'FileSpec2.3001.mat']
    ioclass = BlackrockIO

if __name__ == '__main__':
    unittest.main()
