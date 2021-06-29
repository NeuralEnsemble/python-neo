import unittest

from neo.rawio.neuroexplorerrawio import NeuroExplorerRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNeuroExplorerRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuroExplorerRawIO
    entities_to_download = [
        'neuroexplorer'
    ]
    files_to_download = [
        'neuroexplorer/File_neuroexplorer_1.nex',
        'neuroexplorer/File_neuroexplorer_2.nex',
    ]


if __name__ == "__main__":
    unittest.main()
