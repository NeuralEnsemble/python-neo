"""
Tests of neo.io.neuroscopeio
"""

import unittest

from neo.io import NeuroScopeIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestNeuroScopeIO(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuroScopeIO
    entities_to_download = [
        'neuroscope'
    ]
    entities_to_test = ['neuroscope/test1/test1.xml']


if __name__ == "__main__":
    unittest.main()
