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

    def test_read_raw_protocol_epoch_shapes(self):
        # One file per epoch shape, on DAC 1. The first holding is 1/64 of the sweep (312 of 20000
        # samples), then a step to -20 mV, a step to -10 mV and the shape from -10 mV to 25 mV over
        # samples 5312 to 15312, with trains of period 1000 and width 200. Expected values from the
        # vendor library (ABF_GetWaveform, ABF File Support Pack 2.03).
        start = 5312
        expected = {
            "step": {0: 25.0, 9999: 25.0},
            "ramp": {0: -9.99825, 5000: 7.50175, 9999: 24.99825},
            "pulse_train": {0: 25.0, 199: 25.0, 200: -10.0, 1000: 25.0},
            "triangle_train": {100: 7.5, 200: 25.0, 600: 7.5},
            "cosine_train": {0: -10.0, 250: 7.5, 500: 25.0},
            "biphasic_train": {50: 25.0, 150: -45.0, 250: -10.0},
        }
        for shape, values in expected.items():
            with self.subTest(shape=shape):
                filename = self.get_local_path(f"axon/intracellular_data/read_raw_protocol/{shape}.abf")
                reader = AxonRawIO(filename=filename)
                reader.parse_header()
                sigs_by_segments, _, _ = reader.read_raw_protocol()
                waveform = sigs_by_segments[0][1]
                self.assertEqual(waveform[311], -10.0)
                self.assertEqual(waveform[312], -20.0)
                self.assertEqual(waveform[start - 1], -10.0)
                for offset, value in values.items():
                    self.assertAlmostEqual(waveform[start + offset], value, places=4)
                self.assertEqual(waveform[start + 10000], -10.0)

    def test_read_raw_protocol_increments(self):
        # A level that grows by 10 pA each sweep (DAC 0), and a ramp on DAC 1 whose duration grows
        # by 20 samples each sweep, so the step after it starts later. Expected values from the
        # vendor library.
        reader = AxonRawIO(
            filename=self.get_local_path("axon/intracellular_data/read_raw_protocol/level_increment.abf")
        )
        reader.parse_header()
        sigs_by_segments, _, _ = reader.read_raw_protocol()
        self.assertEqual([sigs[0][2937] for sigs in sigs_by_segments], [-50.0, -40.0, -30.0])

        filename = self.get_local_path("axon/intracellular_data/read_raw_protocol/duration_increment.abf")
        reader = AxonRawIO(filename=filename)
        reader.parse_header()
        sigs_by_segments, _, _ = reader.read_raw_protocol()
        for sweep, sigs in enumerate(sigs_by_segments):
            step_start = 4170 + 20 * sweep
            self.assertLess(sigs[1][step_start - 2], 1.0 + sweep)
            self.assertEqual(sigs[1][step_start], 1.0 + sweep)

    def test_read_raw_protocol_last_epoch_level(self):
        # A ramp to 50 mV that grows by 20 mV each sweep, with the level between sweeps set to the
        # last epoch's level, so each sweep starts where the previous one ended.
        reader = AxonRawIO(
            filename=self.get_local_path("axon/intracellular_data/read_raw_protocol/last_epoch_level_ramp.abf")
        )
        reader.parse_header()
        sigs_by_segments, _, _ = reader.read_raw_protocol()
        self.assertEqual([sigs[0][0] for sigs in sigs_by_segments], [0.0, 50.0, 70.0, 90.0])
        self.assertAlmostEqual(sigs_by_segments[0][0][-1], 50.0, places=4)

    def test_read_raw_protocol_abf1(self):
        # ABF 1.83, four channels, a step on DAC 1 from 0 to -400 pA growing by 50 pA each sweep.
        # Expected values from the vendor library.
        reader = AxonRawIO(filename=self.get_local_path("axon/intracellular_data/tags/episodic_inside_sweep.abf"))
        reader.parse_header()
        sigs_by_segments, names, units = reader.read_raw_protocol()
        self.assertEqual((names[1], units[1]), ("Cmd 1", "pA"))
        waveform = sigs_by_segments[2][1]
        self.assertEqual(waveform.size, 20000)
        self.assertEqual(waveform[1311], 0.0)
        self.assertEqual(waveform[1312], -300.0)
        self.assertEqual(waveform[7311], -300.0)
        self.assertEqual(waveform[7312], 0.0)

    def test_read_raw_protocol_warns_on_unsupported_features(self):
        reader = AxonRawIO(filename=self.get_local_path("axon/intracellular_data/read_raw_protocol/user_list.abf"))
        reader.parse_header()
        with self.assertWarnsRegex(UserWarning, "user lists"):
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
