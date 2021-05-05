import unittest

from neo.rawio.plexonrawio import PlexonRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestPlexonRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = PlexonRawIO
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
