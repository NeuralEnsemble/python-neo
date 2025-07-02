import unittest

from neo.rawio.nicoletrawio import NicoletRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNicoletRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = NicoletRawIO
    entities_to_download = ['nicolet']

    entities_to_test = ["nicolet/e_files/test.e"]
    
if __name__ == "__main__":
    unittest.main()
