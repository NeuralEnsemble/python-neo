"""Test native SpikeGadgets timestamps with generated packets only."""

import mmap
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO


class TestSpikeGadgetsTimestamps(unittest.TestCase):
    """Exercise native tick extraction without downloading recordings."""

    def setUp(self):
        """Remove synthetic files after closing their mappings."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.file_index = 0

    def make_reader(self, ticks, with_aux=False):
        """Write a minimal .rec, optionally with bytes before its timestamp."""
        device = (
            '<Device name="aux" numBytes="2">' '<Channel id="aux0" dataType="analog" startByte="0" />' "</Device>"
            if with_aux
            else ""
        )
        header = (
            "<Configuration>\n<GlobalConfiguration />\n"
            '<HardwareConfiguration samplingRate="30000" numChannels="1">'
            f"{device}</HardwareConfiguration>\n"
            '<SpikeConfiguration device="intan" chanPerChip="1">'
            '<SpikeNTrode id="1" spikeScalingToUv="1"><SpikeChannel hwChan="0" />'
            "</SpikeNTrode></SpikeConfiguration>\n</Configuration>\n"
        )
        fields = [("marker", "u1")]
        if with_aux:
            fields.append(("aux", "<i2"))
        fields.extend([("tick", "<u4"), ("signal", "<i2")])
        packets = np.zeros(len(ticks), dtype=fields)
        packets["marker"] = 0x55
        packets["tick"] = ticks
        packets["signal"] = 42
        path = Path(self.directory.name) / f"{self.file_index}.rec"
        self.file_index += 1
        path.write_bytes(header.encode() + packets.tobytes())
        reader = SpikeGadgetsRawIO(filename=str(path), selected_streams="trodes")
        self.addCleanup(self.close_reader, reader)
        reader.parse_header()
        return reader

    @staticmethod
    def close_reader(reader):
        """Release the mapping before directory cleanup, including on failures."""
        mapping = getattr(reader, "_raw_memmap", None)
        while getattr(mapping, "base", None) is not None:
            mapping = mapping.base
        if isinstance(mapping, mmap.mmap):
            mapping.close()

    def test_native_tick_view_and_slices(self):
        """Retain uint32 values, including wraps, in a read-only sample-aligned view."""
        expected = [2**32 - 2, 2**32 - 1, 0, 1]
        for with_aux in (False, True):
            with self.subTest(with_aux=with_aux):
                reader = self.make_reader(expected, with_aux=with_aux)
                ticks = reader._get_spikegadgets_timestamps()
                assert_array_equal(ticks, expected)
                self.assertEqual(ticks.dtype, np.dtype("<u4"))
                self.assertTrue(np.shares_memory(ticks, reader._raw_memmap))
                self.assertFalse(ticks.flags.writeable)
                assert_array_equal(reader._get_spikegadgets_timestamps(1, 3), expected[1:3])
                self.assertEqual(reader._get_spikegadgets_timestamps(2, 2).size, 0)
                assert_array_equal(reader._get_spikegadgets_timestamps(3), [1])

    def test_native_values(self):
        """Keep native ticks without normalizing starts, gaps, repeats or resets."""
        cases = (
            [100],
            [100, 101, 102, 103],
            [100, 101, 103, 104],
            [2**32 - 2, 0, 1, 2],
            [100, 101, 101, 102],
            [100, 101, 0, 1],
        )
        for ticks in cases:
            with self.subTest(ticks=ticks):
                reader = self.make_reader(ticks)
                assert_array_equal(reader._get_spikegadgets_timestamps(), ticks)

    def test_no_packets(self):
        """Preserve the reader's existing support for header-only files."""
        for with_aux in (False, True):
            with self.subTest(with_aux=with_aux):
                reader = self.make_reader([], with_aux=with_aux)
                ticks = reader._get_spikegadgets_timestamps()
                self.assertEqual(ticks.shape, (0,))
                self.assertEqual(ticks.dtype, np.dtype("<u4"))
                self.assertFalse(ticks.flags.writeable)
