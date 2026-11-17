import unittest

import numpy as np

from neo.rawio.plexonrawio import PlexonRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestPlexonRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = PlexonRawIO
    entities_to_download = ["plexon"]
    entities_to_test = [
        "plexon/File_plexon_1.plx",
        "plexon/File_plexon_2.plx",
        "plexon/File_plexon_3.plx",
        "plexon/4chDemoPLX.plx",
        "plexon/plx/timestamp_clock/one_megahertz_forty_bit_timestamps/sorted_spikes_and_strobed_events.plx",
    ]

    def test_timestamps_with_lower_word_above_two_to_the_31(self):
        """The lower 32 bits of the 40-bit block timestamp must be read unsigned.

        A PLX block stores its 40-bit timestamp in two fields of its header:

            +-------------+------------------------------------+
            | upper byte  |             lower word             |
            |  (8 bits)   |             (32 bits)              |
            +-------------+------------------------------------+
              how many      ticks since the last wrap
              wraps         (0 to 2 ** 32 - 1, then back to 0)

            timestamp = upper * 2 ** 32 + lower
            e.g. upper = 1, lower = 4_294_797_782  ->  8_589_765_078 ticks (8589.77 s at 1 MHz)

        The reader used to read the lower word as a signed int32, so a lower word at or above 2 ** 31 came out
        negative and the timestamp landed 2 ** 32 ticks early. At the usual 40 kHz clock that only happens after
        about 15 hours, but this file has a 1 MHz clock (an Offline Sorter import of a Neuralynx recording),
        where it starts after 36 minutes and each affected timestamp comes out about 72 minutes early.

        The expected values are the spikes of one unit on both sides of the second wrap, at 2 * 2 ** 32 ticks,
        decoded from the block headers. The two before the wrap have lower words above 2 ** 31 and are the
        ones the signed read moved; the two after it have small lower words and were always right.
        """
        filename = self.get_local_path(
            "plexon/plx/timestamp_clock/one_megahertz_forty_bit_timestamps/sorted_spikes_and_strobed_events.plx"
        )
        reader = PlexonRawIO(filename=filename)
        reader.parse_header()

        ids = list(reader.header["spike_channels"]["id"])
        spike_timestamps = reader.get_spike_timestamps(spike_channel_index=ids.index("ch33#1"))
        around_second_wrap = spike_timestamps[(spike_timestamps > 8_589_760_000) & (spike_timestamps < 8_590_600_000)]
        np.testing.assert_array_equal(around_second_wrap, [8_589_765_078, 8_589_773_845, 8_590_497_111, 8_590_594_911])


class TestPlexonAnalogSignalChunk(unittest.TestCase):
    """Test the reading of signal chunks on synthetic data blocks (no file needed)."""

    def setUp(self):
        # one channel made of three data blocks of four int16 samples each
        self.samples = np.arange(1, 13, dtype="int16")
        data_blocks = np.zeros(3, dtype=[("pos", "int64"), ("size", "int64"), ("cumsum", "int64")])
        data_blocks["pos"] = [0, 8, 16]
        data_blocks["size"] = 8
        data_blocks["cumsum"] = [0, 4, 8]

        class FakePlexonRawIO:
            _get_analogsignal_chunk = PlexonRawIO._get_analogsignal_chunk

        self.reader = FakePlexonRawIO()
        self.reader.header = {
            "signal_channels": np.array(
                [("ch", "7", "s0")], dtype=[("name", "U8"), ("id", "U8"), ("stream_id", "U8")]
            ),
            "signal_streams": np.array([("s", "s0")], dtype=[("name", "U8"), ("id", "U8")]),
        }
        self.reader._stream_id_samples = {"s0": self.samples.size}
        self.reader._data_blocks = {5: {np.int32(7): data_blocks}}
        self.reader._memmap = self.samples.view("uint8")

    def _read(self, i_start, i_stop):
        return self.reader._get_analogsignal_chunk(0, 0, i_start, i_stop, 0, None)[:, 0]

    def test_chunk_stopping_at_the_end_of_the_signal(self):
        """The last data block must not be dropped when the chunk stops at its end."""
        np.testing.assert_array_equal(self._read(None, None), self.samples)
        np.testing.assert_array_equal(self._read(2, 12), self.samples[2:12])
        np.testing.assert_array_equal(self._read(9, 12), self.samples[9:12])

    def test_chunk_stopping_inside_or_between_blocks(self):
        for i_start, i_stop in [(0, 11), (0, 8), (3, 6), (4, 8), (5, 9)]:
            np.testing.assert_array_equal(self._read(i_start, i_stop), self.samples[i_start:i_stop])


if __name__ == "__main__":
    unittest.main()
