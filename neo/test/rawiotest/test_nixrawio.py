import unittest
from neo.rawio.nixrawio import NIXRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


testfname = ""


class TestNixRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = NIXRawIO
    entities_to_download = [
        'nix/nixrawio-1.5.nix'
    ]
    entities_to_test = [
        'nix/nixrawio-1.5.nix'
    ]


if __name__ == "__main__":
    unittest.main()
