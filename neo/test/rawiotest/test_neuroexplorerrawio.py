import struct
import unittest

import numpy as np

from neo.rawio.neuroexplorerrawio import EntityHeader, NeuroExplorerRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNeuroExplorerRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = NeuroExplorerRawIO
    entities_to_download = ["neuroexplorer"]
    files_to_download = [
        "neuroexplorer/File_neuroexplorer_1.nex",
        "neuroexplorer/File_neuroexplorer_2.nex",
    ]

    def test_spike_waveforms(self):
        """Waveform data starts after the timestamps, which are 4 bytes each.

        The reader used to skip only 2 bytes per timestamp, so it began reading halfway
        through the timestamp array and returned plausible-looking but wrong values.
        """
        filename = self.get_local_path("neuroexplorer/File_neuroexplorer_2.nex")
        reader = NeuroExplorerRawIO(filename=filename)
        reader.parse_header()

        names = [channel["name"] for channel in reader.header["spike_channels"]]
        channel_index = names.index("sig01i_wf")

        waveforms = reader.get_spike_raw_waveforms(spike_channel_index=channel_index)
        assert waveforms.shape == (5376, 1, 40)

        expected = np.array([-60, -13, 37, 138, 261, 326, 249, 16], dtype="int16")
        np.testing.assert_array_equal(waveforms[0, 0, :8], expected)

    def test_data_offset_above_two_gigabytes(self):
        """DataOffset holds the low 32 bits of the true offset, so it must be read unsigned.

        NeuroExplorer keeps writing past 2 GB even though the specification declares the
        field signed, so a variable beyond that point reads back negative and the reader
        indexes its memmap from the wrong end of the file. A file over 2 GB is too large to
        ship as a test file, so this checks the header definition directly.
        """
        entity_dtype = np.dtype(EntityHeader)
        offset_of_field = entity_dtype.fields["offset"][1]
        assert offset_of_field == 72

        true_offset = 2167961220  # past 2 ** 31, taken from a real 2.5 GB recording
        buffer = bytearray(entity_dtype.itemsize)
        struct.pack_into("<I", buffer, offset_of_field, true_offset)

        parsed = np.frombuffer(bytes(buffer), dtype=entity_dtype)[0]
        assert parsed["offset"] == true_offset


if __name__ == "__main__":
    unittest.main()
