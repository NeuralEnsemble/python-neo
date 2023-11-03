import unittest
import os

from neo.rawio.maxwellrawio import (MaxwellRawIO,
    auto_install_maxwell_hdf5_compression_plugin)
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))

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
        if ON_GITHUB:
            # set a custom existing path for the hdf5 plugin
            hdf5_plugin_path = '/home/runner/work/hdf5_plugin_path_maxwell'
            os.environ['HDF5_PLUGIN_PATH'] = hdf5_plugin_path
        else:
            hdf5_plugin_path = None
        auto_install_maxwell_hdf5_compression_plugin(hdf5_plugin_path=hdf5_plugin_path,
                                                     force_download=False)
        BaseTestRawIO.setUp(self)


if __name__ == "__main__":
    unittest.main()
