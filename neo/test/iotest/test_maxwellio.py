import unittest
import os

from neo.io import MaxwellIO
from neo.test.iotest.common_io_test import BaseTestIO

from neo.rawio.maxwellrawio import auto_install_maxwell_hdf5_compression_plugin

ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))


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
        if ON_GITHUB:
            # set a custom existing path for the hdf5 plugin
            hdf5_plugin_path = '/home/runner/work/hdf5_plugin_path_maxwell'
            os.environ['HDF5_PLUGIN_PATH'] = hdf5_plugin_path
        else:
            hdf5_plugin_path = None
        auto_install_maxwell_hdf5_compression_plugin(hdf5_plugin_path=hdf5_plugin_path,
                                                     force_download=False)
        BaseTestIO.setUp(self)


if __name__ == "__main__":
    unittest.main()
