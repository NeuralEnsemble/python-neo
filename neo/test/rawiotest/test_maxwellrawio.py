import unittest
import os

from neo.rawio.maxwellrawio import (MaxwellRawIO,
    auto_install_maxwell_hdf5_compression_plugin)
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestMaxwellRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MaxwellRawIO
    entities_to_download = [
        'maxwell'
    ]
    entities_to_test = files_to_test = [
        'maxwell/MaxOne_data/Record/000011/data.raw.h5',
        'maxwell/MaxTwo_data/Network/000028/data.raw.h5'
    ]

    def setUp(self):
        auto_install_maxwell_hdf5_compression_plugin(force_download=False)
        BaseTestRawIO.setUp(self)


if __name__ == "__main__":
    unittest.main()
