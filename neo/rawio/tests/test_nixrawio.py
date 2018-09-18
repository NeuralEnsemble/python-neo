import unittest
from neo.rawio.nixrawio import NIXRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


testfname = "neoraw.nix"

class TestNixRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NIXRawIO
    entities_to_test = [testfname]
    files_to_download = [testfname]


if __name__ == "__main__":
    unittest.main()
