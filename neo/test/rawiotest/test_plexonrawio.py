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

        A PLX block stores its timestamp as an upper byte and a lower 32-bit word, and the timestamp is
        upper * 2 ** 32 + lower. The reader used to read the lower word as a signed int32, so a lower word at
        or above 2 ** 31 came out negative and the timestamp landed 2 ** 32 ticks early. At the usual 40 kHz
        clock that only happens after about 15 hours, but this file has a 1 MHz clock (an Offline Sorter
        import of a Neuralynx recording), where it starts after 36 minutes and each affected timestamp comes
        out about 72 minutes early.

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


if __name__ == "__main__":
    unittest.main()
