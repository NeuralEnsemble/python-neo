import unittest

from neo.rawio.axonrawio import AxonRawIO
from neo.core import NeoReadWriteError

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestAxonRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = AxonRawIO
    entities_to_test = [
        "axon/File_axon_1.abf",  # V2.0
        "axon/File_axon_2.abf",  # V1.8
        "axon/File_axon_3.abf",  # V1.8
        "axon/File_axon_4.abf",  # 2.0
        "axon/File_axon_5.abf",  # V.20
        "axon/File_axon_6.abf",  # V.20
        "axon/File_axon_7.abf",  # V2.6
        "axon/test_file_edr3.abf",  # EDR3
    ]
    entities_to_download = ["axon"]

    def test_read_raw_protocol(self):
        reader = AxonRawIO(filename=self.get_local_path("axon/File_axon_7.abf"))
        reader.parse_header()

        reader.read_raw_protocol()

    def test_integer_overflow_size_raises(self):
        # An ABF header that claims more samples than the file can hold must raise a
        # clear error instead of silently returning an overflowed signal size.
        path = self.get_local_path("axon/intracellular_data/files_with_errors/integer_overflow_size.abf")
        expected_error = (
            "ABF header implies 3221225472 samples ending at byte 6442457600, which exceeds the "
            f"file size of 8704 bytes for {path}; the file header is corrupt or the file is truncated."
        )
        reader = AxonRawIO(filename=path)
        with self.assertRaises(NeoReadWriteError) as cm:
            reader.parse_header()
        self.assertEqual(str(cm.exception), expected_error)

    def test_negative_segment_size_raises(self):
        # An ABF header with a negative segment size must raise a clear error
        # instead of silently returning a negative signal size.
        path = self.get_local_path("axon/intracellular_data/files_with_errors/negative_synch_length.abf")
        expected_error = f"Negative segment size (-1041598657) parsed from {path}; the file header is corrupt."
        reader = AxonRawIO(filename=path)
        with self.assertRaises(NeoReadWriteError) as cm:
            reader.parse_header()
        self.assertEqual(str(cm.exception), expected_error)


if __name__ == "__main__":
    unittest.main()
