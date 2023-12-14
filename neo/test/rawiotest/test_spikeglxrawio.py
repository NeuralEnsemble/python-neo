"""
Tests of neo.rawio.spikeglxrawio
"""

import unittest

from neo.rawio.spikeglxrawio import SpikeGLXRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpikeGLXRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = SpikeGLXRawIO
    entities_to_download = ["spikeglx"]
    entities_to_test = [
        "spikeglx/Noise4Sam_g0",
        "spikeglx/TEST_20210920_0_g0",
        # this is only g0 multi index
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI0/5-19-2022-CI0_g0",
        # this is only g1 multi index
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI0/5-19-2022-CI0_g1",
        # this mix both multi gate and multi trigger (and also multi probe)
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI0",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI1",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI2",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI3",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI4",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI5",
        # different sync/sybset options with commercial NP2
        "spikeglx/NP2_with_sync",
        "spikeglx/NP2_no_sync",
        "spikeglx/NP2_subset_with_sync",
    ]

    def test_with_location(self):
        rawio = SpikeGLXRawIO(self.get_local_path("spikeglx/Noise4Sam_g0"), load_channel_location=True)
        rawio.parse_header()
        # one of the stream have channel location
        have_location = []
        for sig_anotations in rawio.raw_annotations["blocks"][0]["segments"][0]["signals"]:
            have_location.append("channel_location_0" in sig_anotations["__array_annotations__"])
        assert any(have_location)

    def test_sync(self):
        rawio_with_sync = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_with_sync"), load_sync_channel=True)
        rawio_with_sync.parse_header()
        stream_index = list(rawio_with_sync.header["signal_streams"]["name"]).index("imec0.ap")

        # AP stream has 385 channels
        chunk = rawio_with_sync.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 385

        rawio_no_sync = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_with_sync"), load_sync_channel=False)
        rawio_no_sync.parse_header()

        # AP stream has 384 channels
        chunk = rawio_no_sync.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 384

    def test_no_sync(self):
        # requesting sync channel when there is none raises an error
        with self.assertRaises(ValueError):
            rawio_no_sync = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_no_sync"), load_sync_channel=True)
            rawio_no_sync.parse_header()

    def test_subset_with_sync(self):
        rawio_sub = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_subset_with_sync"), load_sync_channel=True)
        rawio_sub.parse_header()
        stream_index = list(rawio_sub.header["signal_streams"]["name"]).index("imec0.ap")

        # AP stream has 121 channels
        chunk = rawio_sub.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 121

        rawio_sub_no_sync = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_subset_with_sync"), load_sync_channel=False)
        rawio_sub_no_sync.parse_header()
        # AP stream has 120 channels
        chunk = rawio_sub_no_sync.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 120


if __name__ == "__main__":
    unittest.main()
