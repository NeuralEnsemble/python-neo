import unittest

from neo.rawio.neuroexplorerrawio import NeuroExplorerRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNeuroExplorerRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuroExplorerRawIO
    files_to_download = [
        'File_neuroexplorer_1.nex',
        'File_neuroexplorer_2.nex',
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
