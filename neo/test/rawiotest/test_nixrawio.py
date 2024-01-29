import unittest
from neo.rawio.nixrawio import NIXRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNixRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = NIXRawIO
    entities_to_download = [
        'nix'
    ]
    entities_to_test = [
        'nix/nixrawio-1.5.nix'
    ]
    
    nix_versions = ['0.6.1', '0.7.2', '0.8.0', '0.9.0', '0.10.2', '0.11.1', '0.12.0']
    nix_version_testfiles = [f'nix/generated_file_neo{ver}.nix' for ver in nix_versions]
    entities_to_test.extend(nix_version_testfiles)


if __name__ == "__main__":
    unittest.main()
