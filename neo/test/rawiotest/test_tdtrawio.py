import unittest
from pathlib import Path
from numpy.testing import assert_array_equal, assert_

from neo.rawio.tdtrawio import TdtRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestTdtRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = TdtRawIO
    entities_to_download = [
        'tdt'
    ]
    entities_to_test = [
        # test structure directory with multiple blocks
        'tdt/aep_05',
        # test single block
        'tdt/dataset_0_single_block/512ch_reconly_all-181123_B24_rest.Tdx',
        'tdt/dataset_1_single_block/ECTest-220207-135355_ECTest_B1.Tdx',
        'tdt/aep_05/Block-1/aep_05_Block-1.Tdx'
    ]

    def test_invalid_dirname(self):
        invalid_name = 'random_non_existant_tdt_filename'
        assert not Path(invalid_name).exists()

        with self.assertRaises(ValueError):
            TdtRawIO(invalid_name)

    def test_compare_load_multi_single_block(self):
        dirname = self.get_local_path('tdt/aep_05')
        filename = self.get_local_path('tdt/aep_05/Block-1/aep_05_Block-1.Tdx')

        io_single = TdtRawIO(filename)
        io_multi = TdtRawIO(dirname)

        io_single.parse_header()
        io_multi.parse_header()

        self.assertEqual(io_single.tdt_block_mode, 'single')
        self.assertEqual(io_multi.tdt_block_mode, 'multi')

        self.assertEqual(io_single.block_count(), 1)
        self.assertEqual(io_multi.block_count(), 1)

        self.assertEqual(io_single.segment_count(0), 1)
        self.assertEqual(io_multi.segment_count(0), 2)

        # compare header infos
        assert_array_equal(io_single.header['signal_streams'], io_multi.header['signal_streams'])
        assert_array_equal(io_single.header['signal_channels'], io_multi.header['signal_channels'])
        assert_array_equal(io_single.header['event_channels'], io_multi.header['event_channels'])

        # not all spiking channels are present in first tdt block (segment)
        for spike_channel in io_single.header['spike_channels']:
            self.assertIn(spike_channel, io_multi.header['spike_channels'])

        # check that extracted signal chunks are identical
        assert_array_equal(io_single.get_analogsignal_chunk(0, 0, 0, 100, 0),
                           io_multi.get_analogsignal_chunk(0, 0, 0, 100, 0))


if __name__ == "__main__":
    unittest.main()
