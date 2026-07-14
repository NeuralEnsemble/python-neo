import datetime
import unittest

from neo.rawio.axonrawio import AxonRawIO, parse_axon_soup
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

    def test_empty_channel_name_gets_fallback(self):
        # Some ABF files store a blank ADC channel name, which collapses to "" after space
        # stripping and leaves the channel unaddressable by name. A positional fallback (ch{id})
        # must be used instead so every channel keeps a usable name.
        path = self.get_local_path("axon/intracellular_data/abf1_episodic_empty_channel_name.abf")
        reader = AxonRawIO(filename=path)
        reader.parse_header()
        names = list(reader.header["signal_channels"]["name"])
        self.assertNotIn("", names)
        self.assertEqual(names, ["ch0"])

    def test_channel_name_keeps_interior_space(self):
        # Channel names are stripped of padding but keep interior spaces (e.g. "IN 1", not "IN1")
        # and are returned as str.
        reader = AxonRawIO(filename=self.get_local_path("axon/File_axon_7.abf"))
        reader.parse_header()
        names = list(reader.header["signal_channels"]["name"])
        self.assertEqual(names, ["IN 1"])

    def test_protocol_path_decoded_to_str(self):
        # String header fields should be decoded to str, not left as raw bytes; otherwise a caller
        # doing str(value) gets the "b'...'" byte-literal repr baked into the path.
        for fixture in ["axon/File_axon_1.abf", "axon/File_axon_2.abf"]:  # v2 and v1
            reader = AxonRawIO(filename=self.get_local_path(fixture))
            reader.parse_header()
            self.assertIsInstance(reader._axon_info["sProtocolPath"], str)

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

    def test_unparseable_file_raises(self):
        # A file whose header does not start with a valid ABF signature must raise a clear error
        # rather than a cryptic NoneType error deep in parsing. The fixture has a zeroed signature.
        path = self.get_local_path("axon/intracellular_data/files_with_errors/unparseable_header.abf")
        expected_msg = (
            f"Could not parse {path} as an ABF file: expected the header to start with signature "
            f"b'ABF ' or b'ABF2', but found b'\\x00\\x00\\x00\\x00'. The file is not an ABF file, "
            f"is corrupt, or is an unsupported variant."
        )
        reader = AxonRawIO(filename=path)
        with self.assertRaises(NeoReadWriteError) as cm:
            reader.parse_header()
        self.assertEqual(str(cm.exception), expected_msg)

    def test_v1_reads_real_acquisition_date(self):
        # ABF1 stores the calendar date in lFileStartDate (a YYYYMMDD-packed integer). Older neo
        # ignored that field and hardcoded 1900-01-01, so the recording date was always wrong for
        # ABF1 files. It must now be read from the header.
        expected_datetime = datetime.datetime(2005, 6, 11, 14, 15, 0)
        header = parse_axon_soup(self.get_local_path("axon/File_axon_2.abf"))
        rec_datetime = header["rec_datetime"]
        # Drop sub-second precision: the millisecond field round-trips through a float, so the
        # microsecond is a rounding artifact, not a meaningful value to assert.
        self.assertEqual(rec_datetime.replace(microsecond=0), expected_datetime)

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

    def test_non_unique_channel_ids_fall_back_to_sequential_ids(self):
        # Some version < 2.0 ABF files (e.g. re-saved exports) corrupt nADCSamplingSeq so every
        # entry is identical, which yields non-unique channel ids. AxonRawIO detects this, warns,
        # and falls back to sequential ids (the same default used for version >= 2.0) so the file
        # stays readable and channels remain addressable by id.
        path = self.get_local_path("axon/intracellular_data/files_with_errors/non_unique_channel_ids.abf")
        reader = AxonRawIO(filename=path)
        with self.assertLogs(reader.logger, level="WARNING"):
            reader.parse_header()
        channel_ids = list(reader.header["signal_channels"]["id"])
        self.assertEqual(len(channel_ids), len(set(channel_ids)))


if __name__ == "__main__":
    unittest.main()
