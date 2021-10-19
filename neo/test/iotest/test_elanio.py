"""
Tests of neo.io.elanio
"""

import unittest

from neo.io import ElanIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestElanIO(BaseTestIO, unittest.TestCase, ):
    ioclass = ElanIO
    entities_to_download = [
        'elan'
    ]
    entities_to_test = [
        'elan/File_elan_1.eeg',
    ]


if __name__ == "__main__":
    unittest.main()
