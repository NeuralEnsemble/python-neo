import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from neo.rawio.plexonrawio import DataBlockHeader, GlobalHeader, PlexonRawIO, SlowChannelHeader

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


class TestPlexonBlockOffsets(unittest.TestCase):
    """Check parsed offsets and reads without downloading recordings."""

    def test_unequal_signal_block_sizes(self):
        """Parsed block starts must use the number of samples in preceding blocks."""
        samples = np.arange(350, dtype="int16")
        header = np.zeros(1, dtype=GlobalHeader)
        for field, value in {
            "MagicNumber": 0x58454C50,
            "Version": 106,
            "ADFrequency": 1000,
            "NumSlowChannels": 1,
            "Year": 2020,
            "Month": 1,
            "Day": 1,
            "BitsPerSpikeSample": 16,
            "SlowMaxMagnitudeMV": 5000,
        }.items():
            header[field] = value
        channel = np.zeros(1, dtype=SlowChannelHeader)
        channel["Name"] = b"AI1"
        for field in ["Gain", "PreampGain", "Enabled"]:
            channel[field] = 1
        channel["ADFreq"] = 1000

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "unequal_blocks.plx"
            with filename.open("wb") as stream:
                header.tofile(stream)
                channel.tofile(stream)
                for start, stop in [(0, 100), (100, 300), (300, 350)]:
                    block = np.zeros(1, dtype=DataBlockHeader)
                    block["Type"] = 5
                    block["TimeStamp"] = start
                    block["NumberOfWaveforms"] = 1
                    block["NumberOfWordsInWaveform"] = stop - start
                    block.tofile(stream)
                    samples[start:stop].tofile(stream)
            reader = PlexonRawIO(filename=str(filename), progress_bar=False)
            try:
                reader.parse_header()
                np.testing.assert_array_equal(reader._data_blocks[5][0]["cumsum"], [0, 100, 300])
                chunk = reader.get_analogsignal_chunk(i_start=80, i_stop=280)
                np.testing.assert_array_equal(chunk[:, 0], samples[80:280])
            finally:
                del reader


if __name__ == "__main__":
    unittest.main()
