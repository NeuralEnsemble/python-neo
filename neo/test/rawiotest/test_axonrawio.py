import unittest

from neo.rawio.axonrawio import AxonRawIO, parse_axon_soup

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

    def test_invalid_date_falls_back_to_none(self):
        # Some ABF files store an out-of-range / "no date" sentinel (e.g. 0xFFFFFFFF)
        # in the acquisition date header fields. The date is non-essential annotation,
        # so parsing must fall back to rec_datetime=None rather than raising and
        # blocking access to the signal.
        for fixture in [
            "axon/intracellular_data/files_with_errors/invalid_date_abf1.abf",  # ABF v1
            "axon/intracellular_data/files_with_errors/invalid_date_abf2.abf",  # ABF v2
        ]:
            header = parse_axon_soup(self.get_local_path(fixture))
            self.assertIsNone(header["rec_datetime"])


if __name__ == "__main__":
    unittest.main()
