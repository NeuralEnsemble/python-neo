import unittest

from neo.rawio.plexonrawio import PlexonRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestPlexonRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = PlexonRawIO
    files_to_download = [
        'File_plexon_1.plx',
        'File_plexon_2.plx',
        'File_plexon_3.plx',
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
