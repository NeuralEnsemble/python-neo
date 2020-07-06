import unittest

from neo.rawio.axonrawio import AxonRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestAxonRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = AxonRawIO
    entities_to_test = [
        'File_axon_1.abf',  # V2.0
        'File_axon_2.abf',  # V1.8
        'File_axon_3.abf',  # V1.8
        'File_axon_4.abf',  # 2.0
        'File_axon_5.abf',  # V.20
        'File_axon_6.abf',  # V.20
        'File_axon_7.abf',  # V2.6
        'test_file_edr3.abf',  # EDR3
    ]
    files_to_download = entities_to_test

    def test_read_raw_protocol(self):
        reader = AxonRawIO(filename=self.get_filename_path('File_axon_7.abf'))
        reader.parse_header()

        reader.read_raw_protocol()


if __name__ == "__main__":
    unittest.main()
