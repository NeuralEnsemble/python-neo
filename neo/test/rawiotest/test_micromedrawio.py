""" """

import struct
import tempfile
import unittest
from pathlib import Path

from neo.core import NeoReadWriteError
from neo.io import MicromedIO
from neo.rawio.micromedrawio import MicromedRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import numpy as np


class TestMicromedRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = MicromedRawIO
    entities_to_download = ["micromed"]
    entities_to_test = [
        "micromed/File_micromed_1.TRC",
        "micromed/File_mircomed2.TRC",
        "micromed/File_mircomed2_2segments.TRC",
    ]

    def test_micromed_multi_segments(self):
        file_full = self.get_local_path("micromed/File_mircomed2.TRC")
        file_splitted = self.get_local_path("micromed/File_mircomed2_2segments.TRC")

        # the second file contains 2 pieces of the first file
        # so it is 2 segments with the same traces but reduced
        # note that traces in the splited can differ at the very end of the cut

        reader1 = MicromedRawIO(file_full)
        reader1.parse_header()
        assert reader1.segment_count(block_index=0) == 1
        assert reader1.get_signal_t_start(block_index=0, seg_index=0, stream_index=0) == 0.0
        traces1 = reader1.get_analogsignal_chunk(stream_index=0)

        reader2 = MicromedRawIO(file_splitted)
        reader2.parse_header()
        print(reader2)
        assert reader2.segment_count(block_index=0) == 2

        # check that pieces of the second file is equal to the first file (except a truncation at the end)
        for seg_index in range(2):
            t_start = reader2.get_signal_t_start(block_index=0, seg_index=seg_index, stream_index=0)
            assert t_start > 0
            sr = reader2.get_signal_sampling_rate(stream_index=0)
            ind_start = int(t_start * sr)
            traces2 = reader2.get_analogsignal_chunk(block_index=0, seg_index=seg_index, stream_index=0)
            traces1_chunk = traces1[ind_start : ind_start + traces2.shape[0]]
            # we remove the last 100 sample because tools that cut traces is truncating the last buffer
            assert np.array_equal(traces2[:-100], traces1_chunk[:-100])


ZONE_NAMES = [
    "ORDER",
    "LABCOD",
    "NOTE",
    "FLAGS",
    "TRONCA",
    "IMPED_B",
    "IMPED_E",
    "MONTAGE",
    "COMPRESS",
    "AVERAGE",
    "HISTORY",
    "DVIDEO",
    "EVENT A",
    "EVENT B",
    "TRIGGER",
]


def write_micromed_trc(
    path, codes, sampling_rate=256, header_version=4, tronca=(), notes=(), triggers=(), epochs=(), zone_names=None
):
    """
    Write a minimal Micromed .trc file, for tests that do not need a real recording.

    The header has the 15 zones neo reads, and no patient data (the name and the other free text fields stay empty).

    Parameters
    ----------
    path: str or Path
        The file to write.
    codes: np.ndarray of shape (n_samples, n_channels)
        The ADC codes (unsigned 16 bits) of the data block. The channels are in uV, with the logical ground at 32768.
    sampling_rate: int, default: 256
        Rate_Min of the header, and the rate of every channel.
    header_version: int, default: 4
        Version 4 pads the zone names with spaces, version 5 with NULs.
    tronca: list of (seg_start, trace_offset)
        The TRONCA table: the sample at index trace_offset of the data block is at sample seg_start of the recording clock.
    notes: list of (start, text)
        NOTE events, start is an index in the data block.
    triggers: list of (start, label)
        TRIGGER events, start is an index in the data block.
    epochs: list of (label, start, stop)
        EVENT A entries, start and stop are indices in the data block.
    zone_names: dict, default: None
        Names to write in the zone table instead of the standard ones, by standard name.
    """
    codes = np.asarray(codes, dtype="<u2")
    n_samples, n_channels = codes.shape

    zones = {name: bytes(16) for name in ZONE_NAMES}
    zones["ORDER"] = np.arange(n_channels, dtype="<u2").tobytes()

    # one 128 bytes record per channel: name, ground, logical min/max/ground, physical min/max, unit (0 = uV), rate
    labcod = bytearray(n_channels * 128)
    for c in range(n_channels):
        fmt = "<6s6s5ihxxxxxxxxH"
        struct.pack_into(fmt, labcod, c * 128 + 2, f"Ch{c + 1}".encode(), b"", 0, 65535, 32768, -4096, 4096, 0, 1)
    zones["LABCOD"] = bytes(labcod)

    # the unused slots at the end of the NOTE, TRIGGER and EVENT zones start at 0
    note_zone = bytearray((len(notes) + 2) * 44)
    for i, (start, text) in enumerate(notes):
        struct.pack_into("<I40s", note_zone, i * 44, start, text.encode("cp1252"))
    zones["NOTE"] = bytes(note_zone)

    trigger_zone = bytearray((len(triggers) + 2) * 6)
    for i, (start, label) in enumerate(triggers):
        struct.pack_into("<IH", trigger_zone, i * 6, start, label)
    zones["TRIGGER"] = bytes(trigger_zone)

    epoch_zone = bytearray((len(epochs) + 1) * 12)
    for i, (label, start, stop) in enumerate(epochs):
        struct.pack_into("<III", epoch_zone, i * 12, label, start, stop)
    zones["EVENT A"] = bytes(epoch_zone)
    zones["EVENT B"] = bytes(12)

    # the table is ended by a (0, 0) pair
    tronca_zone = bytearray((len(tronca) + 1) * 8)
    for i, (seg_start, trace_offset) in enumerate(tronca):
        struct.pack_into("<II", tronca_zone, i * 8, seg_start, trace_offset)
    zones["TRONCA"] = bytes(tronca_zone)

    padding = b" " if header_version == 4 else b"\x00"
    zone_names = zone_names or {}
    zone_table_end = 176 + 16 * len(ZONE_NAMES)
    position = zone_table_end
    zone_table = b""
    for name in ZONE_NAMES:
        zone_name = zone_names.get(name, name).encode().ljust(8, padding)
        zone_table += struct.pack("<8sII", zone_name, position, len(zones[name]))
        position += len(zones[name])
    data_start = position

    header = bytearray(zone_table_end)
    struct.pack_into("<6b", header, 128, 9, 3, 124, 10, 30, 15)  # day, month, year - 1900, hour, minute, second
    struct.pack_into("<IHHHH", header, 138, data_start, n_channels, 1, sampling_rate, 2)
    header[175] = header_version
    header[176:zone_table_end] = zone_table

    with open(path, "wb") as f:
        f.write(bytes(header) + b"".join(zones[name] for name in ZONE_NAMES) + codes.tobytes())


class TestMicromedSyntheticFiles(unittest.TestCase):
    """
    Tests on small files written by write_micromed_trc, so that they need no downloaded file.

    A TRONCA entry is (seg_start, trace_offset): the sample at index trace_offset of the data block is at sample
    seg_start of the recording clock, which is the time axis of a segment. Before the first entry, the clock and
    the data block agree.
    """

    sampling_rate = 256
    # 3000 samples: 1000 before the first cut, 1000 between the cuts, 1000 after the second one
    codes = np.random.default_rng(0).integers(30000, 35000, size=(3000, 4))
    one_cut = [(3560, 1000)]
    two_cuts = [(3560, 1000), (5840, 2000)]

    def make_reader(self, **kwargs):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "synthetic.TRC"
        write_micromed_trc(path, self.codes, self.sampling_rate, **kwargs)
        reader = MicromedRawIO(str(path))
        reader.parse_header()
        return reader

    def segment_sizes_and_t_starts(self, reader):
        n_segment = reader.segment_count(block_index=0)
        sizes = [reader.get_signal_size(0, seg_index, 0) for seg_index in range(n_segment)]
        t_starts = [reader.get_signal_t_start(0, seg_index, 0) for seg_index in range(n_segment)]
        return sizes, t_starts

    def note_times(self, reader, seg_index, **kwargs):
        timestamps, _, labels = reader.get_event_timestamps(0, seg_index, event_channel_index=1, **kwargs)
        times = reader.rescale_event_timestamp(timestamps, "float64", event_channel_index=1)
        return [str(label) for label in labels], times

    def test_file_without_table_is_one_segment(self):
        reader = self.make_reader(notes=[(500, "A"), (1500, "B")])
        self.assertEqual(self.segment_sizes_and_t_starts(reader), ([3000], [0.0]))
        np.testing.assert_array_equal(reader.get_analogsignal_chunk(stream_index=0), self.codes)
        labels, times = self.note_times(reader, 0)
        self.assertEqual(labels, ["A", "B"])
        np.testing.assert_allclose(times, [500 / 256, 1500 / 256])

    def test_samples_before_the_first_trace_offset_are_the_first_segment(self):
        reader = self.make_reader(tronca=self.one_cut)

        sizes, t_starts = self.segment_sizes_and_t_starts(reader)
        self.assertEqual(sizes, [1000, 2000])
        np.testing.assert_allclose(t_starts, [0.0, 3560 / 256])
        for seg_index, (first, last) in enumerate([(0, 1000), (1000, 3000)]):
            chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=seg_index, stream_index=0)
            np.testing.assert_array_equal(chunk, self.codes[first:last])

    def test_every_segment_of_a_longer_table_is_read(self):
        reader = self.make_reader(tronca=self.two_cuts)

        sizes, t_starts = self.segment_sizes_and_t_starts(reader)
        self.assertEqual(sizes, [1000, 1000, 1000])
        np.testing.assert_allclose(t_starts, [0.0, 3560 / 256, 5840 / 256])
        self.assertEqual(sum(sizes), len(self.codes))

    def test_table_starting_at_the_first_sample_has_no_extra_segment(self):
        # the shape of File_mircomed2_2segments.TRC: the file starts at sample 5 of the recording clock
        reader = self.make_reader(tronca=[(5, 0), (1005, 1000)])

        sizes, t_starts = self.segment_sizes_and_t_starts(reader)
        self.assertEqual(sizes, [1000, 2000])
        np.testing.assert_allclose(t_starts, [5 / 256, 1005 / 256])

    def test_event_times_are_on_the_time_axis_of_their_segment(self):
        notes = [(500, "A"), (1500, "B"), (2500, "C")]
        reader = self.make_reader(tronca=self.two_cuts, notes=notes)

        # the sample at index s of segment k is at (seg_start + s - trace_offset) / sampling_rate
        expected = [(0, "A", 500 / 256), (1, "B", (3560 + 1500 - 1000) / 256), (2, "C", (5840 + 2500 - 2000) / 256)]
        for seg_index, label, time in expected:
            labels, times = self.note_times(reader, seg_index)
            self.assertEqual(labels, [label])
            np.testing.assert_allclose(times, [time])
            # and it is the time of the sample the note labels
            first_index = reader.get_analogsignal_chunk(block_index=0, seg_index=seg_index, stream_index=0).shape[0]
            self.assertLess(times[0], reader.get_signal_t_start(0, seg_index, 0) + first_index / 256)

    def test_event_times_do_not_change_when_the_clock_runs_on_across_the_cuts(self):
        # seg_start - trace_offset is 5 in every entry: the time of an event is start + 5, as it always was
        reader = self.make_reader(tronca=[(5, 0), (1005, 1000)], notes=[(500, "A"), (1500, "B")])

        np.testing.assert_allclose(self.note_times(reader, 0)[1], [(500 + 5) / 256])
        np.testing.assert_allclose(self.note_times(reader, 1)[1], [(1500 + 5) / 256])

    def test_event_time_limits_are_on_the_time_axis_of_the_segment(self):
        reader = self.make_reader(tronca=self.one_cut, notes=[(1100, "A"), (1500, "B"), (2500, "C")])

        # in segment 1 the notes are at (3560 + 100 / 500 / 1500) / 256 s
        labels, _ = self.note_times(reader, 1, t_start=(3560 + 200) / 256, t_stop=(3560 + 1000) / 256)
        self.assertEqual(labels, ["B"])
        labels, _ = self.note_times(reader, 1, t_start=(3560 + 600) / 256)
        self.assertEqual(labels, ["C"])
        labels, _ = self.note_times(reader, 1, t_stop=(3560 + 600) / 256)
        self.assertEqual(labels, ["A", "B"])

    def test_triggers_and_epochs_are_on_the_time_axis_of_their_segment(self):
        reader = self.make_reader(tronca=self.one_cut, triggers=[(1200, 7)], epochs=[(0, 1300, 1900)])

        timestamps, _, labels = reader.get_event_timestamps(0, 1, event_channel_index=0)
        np.testing.assert_allclose(reader.rescale_event_timestamp(timestamps, "float64", 0), [(3560 + 200) / 256])
        self.assertEqual([str(label) for label in labels], ["7"])

        timestamps, durations, _ = reader.get_event_timestamps(0, 1, event_channel_index=2)
        np.testing.assert_allclose(reader.rescale_event_timestamp(timestamps, "float64", 2), [(3560 + 300) / 256])
        np.testing.assert_allclose(reader.rescale_epoch_duration(durations, "float64", 2), [600 / 256])

    def test_io_reads_every_segment_and_the_events_with_their_times(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "synthetic.TRC"
        write_micromed_trc(path, self.codes, self.sampling_rate, tronca=self.one_cut, notes=[(500, "A"), (1500, "B")])

        block = MicromedIO(str(path)).read_block()

        self.assertEqual(len(block.segments), 2)
        self.assertEqual([segment.analogsignals[0].shape[0] for segment in block.segments], [1000, 2000])
        np.testing.assert_allclose(
            [float(segment.analogsignals[0].t_start) for segment in block.segments], [0, 3560 / 256]
        )
        times = [
            (str(label), float(time))
            for segment in block.segments
            for event in segment.events
            for time, label in zip(event.times, event.labels)
        ]
        self.assertEqual([label for label, _ in times], ["A", "B"])
        np.testing.assert_allclose([time for _, time in times], [500 / 256, 4060 / 256])

    def test_header_version_5(self):
        # the zone names are padded with NULs in version 5
        reader = self.make_reader(header_version=5, tronca=self.one_cut, notes=[(500, "A")])

        self.assertEqual(self.segment_sizes_and_t_starts(reader)[0], [1000, 2000])
        np.testing.assert_array_equal(reader.get_analogsignal_chunk(stream_index=0, seg_index=1), self.codes[1000:])
        self.assertEqual(self.note_times(reader, 0)[0], ["A"])

    def test_other_header_versions_are_refused(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "synthetic.TRC"
        write_micromed_trc(path, self.codes, self.sampling_rate, header_version=6)
        with self.assertRaises(NotImplementedError):
            MicromedRawIO(str(path)).parse_header()

    def test_a_zone_that_is_not_named_as_expected_is_refused(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "synthetic.TRC"
        write_micromed_trc(path, self.codes, self.sampling_rate, zone_names={"NOTE": "NOPE"})
        with self.assertRaises(NeoReadWriteError):
            MicromedRawIO(str(path)).parse_header()


if __name__ == "__main__":
    unittest.main()
