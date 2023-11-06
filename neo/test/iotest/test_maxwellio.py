import unittest
import os

from neo.io import MaxwellIO
from neo.test.iotest.common_io_test import BaseTestIO

from neo.rawio.maxwellrawio import auto_install_maxwell_hdf5_compression_plugin


class TestMaxwellIO(BaseTestIO, unittest.TestCase, ):
    ioclass = MaxwellIO
    entities_to_download = [
        'maxwell'
    ]
    entities_to_test = files_to_test = [
        'maxwell/MaxOne_data/Record/000011/data.raw.h5',
        'maxwell/MaxTwo_data/Network/000028/data.raw.h5'
    ]

    def setUp(self):
        auto_install_maxwell_hdf5_compression_plugin(force_download=False)
        BaseTestIO.setUp(self)


if __name__ == "__main__":
    unittest.main()
