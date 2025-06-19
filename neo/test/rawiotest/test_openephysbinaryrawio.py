import unittest

from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import numpy as np


class TestOpenEphysBinaryRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = OpenEphysBinaryRawIO
    entities_to_download = ["openephysbinary"]
    entities_to_test = [
        "openephysbinary/v0.5.3_two_neuropixels_stream",
        "openephysbinary/v0.4.4.1_with_video_tracking",
        "openephysbinary/v0.5.x_two_nodes",
        "openephysbinary/v0.6.x_neuropixels_multiexp_multistream",
        "openephysbinary/v0.6.x_neuropixels_with_sync",
        "openephysbinary/v0.6.x_neuropixels_missing_folders",
        "openephysbinary/neural_and_non_neural_data_mixed",
    ]

    def test_sync(self):
        rawio_with_sync = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=True
        )
        rawio_with_sync.parse_header()
        stream_name = [s_name for s_name in rawio_with_sync.header["signal_streams"]["name"] if "AP" in s_name][0]
        stream_index = list(rawio_with_sync.header["signal_streams"]["name"]).index(stream_name)

        # AP stream has 385 channels
        chunk = rawio_with_sync.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 385

        rawio_no_sync = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=False
        )
        rawio_no_sync.parse_header()

        # AP stream has 384 channels
        chunk = rawio_no_sync.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 384

    def test_sync_channel_access(self):
        """Test that sync channels can be accessed as separate streams when load_sync_channel=False."""
        rawio = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=False
        )
        rawio.parse_header()

        # Find sync channel streams
        sync_stream_names = [s_name for s_name in rawio.header["signal_streams"]["name"] if "SYNC" in s_name]
        assert len(sync_stream_names) > 0, "No sync channel streams found"

        # Get the stream index for the first sync channel
        sync_stream_index = list(rawio.header["signal_streams"]["name"]).index(sync_stream_names[0])

        # Check that we can access the sync channel data
        chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=sync_stream_index
        )

        # Sync channel should have only one channel
        assert chunk.shape[1] == 1, f"Expected sync channel to have 1 channel, got {chunk.shape[1]}"

    def test_no_sync(self):
        # requesting sync channel when there is none raises an error
        with self.assertRaises(ValueError):
            rawio_no_sync = OpenEphysBinaryRawIO(
                self.get_local_path("openephysbinary/v0.6.x_neuropixels_multiexp_multistream"), load_sync_channel=True
            )
            rawio_no_sync.parse_header()

    def test_missing_folders(self):
        # missing folders should raise an error
        with self.assertWarns(UserWarning):
            rawio = OpenEphysBinaryRawIO(
                self.get_local_path("openephysbinary/v0.6.x_neuropixels_missing_folders"), load_sync_channel=False
            )
            rawio.parse_header()

    def test_multiple_ttl_events_parsing(self):
        rawio = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"), load_sync_channel=False
        )
        rawio.parse_header()
        rawio.header = rawio.header
        # Testing co
        # This is the TTL events from the NI Board channel
        ttl_events = rawio._evt_streams[0][0][1]
        assert "rising" in ttl_events.keys()
        assert "labels" in ttl_events.keys()
        assert "durations" in ttl_events.keys()
        assert "timestamps" in ttl_events.keys()

        # Check that durations of different event streams are correctly parsed:
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "1"], 0.5, atol=0.001)
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "6"], 0.025, atol=0.001)
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "7"], 0.016666, atol=0.001)

    def test_separating_stream_for_non_neural_data(self):
        rawio = OpenEphysBinaryRawIO(
            self.get_local_path("openephysbinary/neural_and_non_neural_data_mixed"), load_sync_channel=False
        )
        rawio.parse_header()
        # Check that the non-neural data stream is correctly separated
        assert len(rawio.header["signal_streams"]["name"]) == 2
        assert rawio.header["signal_streams"]["name"].tolist() == ["Rhythm_FPGA-100.0", "Rhythm_FPGA-100.0_ADC"]


if __name__ == "__main__":
    unittest.main()
