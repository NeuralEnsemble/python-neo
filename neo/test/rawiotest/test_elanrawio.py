import unittest

from neo.rawio.elanrawio import ElanRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestElanRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = ElanRawIO
    entities_to_test = ['File_elan_1.eeg']
    files_to_download = [
        'File_elan_1.eeg',
        'File_elan_1.eeg.ent',
        'File_elan_1.eeg.pos',
    ]


if __name__ == "__main__":
    unittest.main()
