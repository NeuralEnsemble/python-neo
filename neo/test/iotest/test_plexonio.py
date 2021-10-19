"""
Tests of neo.io.plexonio
"""

import unittest

from neo.io import PlexonIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestPlexonIO(BaseTestIO, unittest.TestCase, ):
    ioclass = PlexonIO
    entities_to_download = [
        'plexon'
    ]
    entities_to_test = [
        'plexon/File_plexon_1.plx',
        'plexon/File_plexon_2.plx',
        'plexon/File_plexon_3.plx',
    ]


if __name__ == "__main__":
    unittest.main()
