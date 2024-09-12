import unittest

from neo.rawio.maxwellrawio import MaxwellRawIO, auto_install_maxwell_hdf5_compression_plugin
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

from neo.utils.datasets import download_dataset, default_testing_repo

try:
    import datalad

    HAVE_DATALAD = True
except:
    HAVE_DATALAD = False

# url_for_tests = "https://portal.g-node.org/neo/" #This is the old place
repo_for_test = default_testing_repo

class TestMaxwellRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):

    def setUp(self):
        auto_install_maxwell_hdf5_compression_plugin(force_download=False)


if __name__ == "__main__":
    unittest.main()
