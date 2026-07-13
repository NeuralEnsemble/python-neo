import datetime
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

    def test_v1_reads_two_digit_year_date(self):
        # Old ABF1 files (before ~v1.65) pack the date as YYMMDD (2-digit year) rather than
        # YYYYMMDD. lFileStartDate = 180618 is 2018-06-18, not year 18, so the 2-digit year must be
        # expanded to the 2000s.
        expected_datetime = datetime.datetime(2018, 6, 18, 17, 34, 27)
        header = parse_axon_soup(self.get_local_path("axon/intracellular_data/abf1_episodic_empty_channel_name.abf"))
        rec_datetime = header["rec_datetime"]
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
