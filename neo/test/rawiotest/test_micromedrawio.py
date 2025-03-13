""" """

import unittest

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


if __name__ == "__main__":
    unittest.main()
