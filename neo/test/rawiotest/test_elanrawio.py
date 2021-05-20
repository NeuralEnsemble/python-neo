import unittest

from neo.rawio.elanrawio import ElanRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestElanRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = ElanRawIO
    entities_to_test = [
        'elan/File_elan_1.eeg'
    ]
    entities_to_download = [
        'elan',
    ]


if __name__ == "__main__":
    unittest.main()
