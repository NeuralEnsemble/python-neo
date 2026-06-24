import unittest

from neo.rawio.axonrawio import AxonRawIO

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

    def test_non_unique_channel_ids_are_deduplicated(self):
        # Some ABF files (e.g. re-saved exports) store duplicate signal_channel ids. Rather than
        # refuse an otherwise-readable file, neo makes the ids unique and warns, so the data stays
        # readable and channels remain addressable by id.
        path = self.get_local_path("axon/intracellular_data/files_with_errors/non_unique_channel_ids.abf")
        reader = AxonRawIO(filename=path)
        with self.assertLogs(reader.logger, level="WARNING"):
            reader.parse_header()
        channel_ids = list(reader.header["signal_channels"]["id"])
        self.assertEqual(len(channel_ids), len(set(channel_ids)))


if __name__ == "__main__":
    unittest.main()
