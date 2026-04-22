"""
Tests of neo.rawio.spikeglxrawio
"""

import unittest

import pytest

from neo.rawio.spikeglxrawio import SpikeGLXRawIO, _build_signals_info_dict
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO
import numpy as np


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
        # NP-ultra
        "spikeglx/np_ultra_stub",
        # Filename changed by the user, multi-dock
        "spikeglx/multi_probe_multi_dock_multi_shank_filename_without_info",
        # CatGT
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-A",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-B",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-C",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-D",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-E",
        "spikeglx/multi_trigger_multi_gate/CatGT/Supercat-A",
        # One Box"
        "spikeglx/onebox/run_with_only_adc",
    ]

    def test_loading_only_one_probe_in_multi_probe_scenario(self):
        from pathlib import Path

        local_path_multi_probe_path = Path(
            self.get_local_path("spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI0")
        )
        gate_folder_path = local_path_multi_probe_path / "5-19-2022-CI0_g0"
        probe_folder_path = gate_folder_path / "5-19-2022-CI0_g0_imec1"

        rawio = SpikeGLXRawIO(probe_folder_path)
        rawio.parse_header()

        expected_stream_names = ["imec1.ap", "imec1.lf", "imec1.ap-SYNC", "imec1.lf-SYNC"]
        actual_stream_names = rawio.header["signal_streams"]["name"].tolist()
        assert (
            actual_stream_names == expected_stream_names
        ), f"Expected {expected_stream_names}, but got {actual_stream_names}"

    def test_with_location(self):
        rawio = SpikeGLXRawIO(self.get_local_path("spikeglx/Noise4Sam_g0"), load_channel_location=True)
        rawio.parse_header()
        # one of the stream have channel location
        have_location = []
        for sig_anotations in rawio.raw_annotations["blocks"][0]["segments"][0]["signals"]:
            have_location.append("channel_location_0" in sig_anotations["__array_annotations__"])
        assert any(have_location)

    def test_sync(self):
        # The sync trace is always split off into its own -SYNC stream; the parent
        # AP stream has 384 channels (384 neural channels, SY0 excluded).
        rawio = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_with_sync"))
        rawio.parse_header()
        stream_index = list(rawio.header["signal_streams"]["name"]).index("imec0.ap")

        chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 384

    def test_subset_with_sync(self):
        # Channel-subset recording with SY: 121 saved channels total, 120 neural plus
        # one SY that is split into the -SYNC stream, leaving 120 in the parent.
        rawio = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_subset_with_sync"))
        rawio.parse_header()
        stream_index = list(rawio.header["signal_streams"]["name"]).index("imec0.ap")

        chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 120

    def test_nidq_digital_channel(self):
        rawio_digital = SpikeGLXRawIO(self.get_local_path("spikeglx/DigitalChannelTest_g0"))
        rawio_digital.parse_header()
        # This data should have 8 event channels
        assert np.shape(rawio_digital.header["event_channels"])[0] == 8

        # Channel 0 in this data will have sync pulses at 1 Hz, let's confirm that
        all_events = rawio_digital.get_event_timestamps(0, 0, 0)
        on_events = np.where(all_events[2] == "XD0 ON")
        on_ts = all_events[0][on_events]
        on_ts_scaled = rawio_digital.rescale_event_timestamp(on_ts)
        on_diff = np.diff(on_ts_scaled)
        atol = 0.001
        assert np.allclose(on_diff, 1, atol=atol)

    def test_sync_channel_as_separate_stream(self):
        """Sync trace is always exposed as its own -SYNC stream."""
        rawio = SpikeGLXRawIO(self.get_local_path("spikeglx/NP2_with_sync"))
        rawio.parse_header()

        stream_names = rawio.header["signal_streams"]["name"].tolist()
        sync_streams = [name for name in stream_names if "SYNC" in name]
        assert len(sync_streams) > 0, "No -SYNC stream found"

    def test_t_start_reading(self):
        """Test that t_start values are correctly read for all streams and segments."""

        # Expected t_start values for each stream and segment
        expected_t_starts = {
            "imec0.ap": {0: 15.319535472007237, 1: 15.339535431281986, 2: 21.284723325294053, 3: 21.3047232845688},
            "imec1.ap": {0: 15.319554693264516, 1: 15.339521518106308, 2: 21.284735282142822, 3: 21.304702106984614},
            "imec0.lf": {0: 15.3191688060872, 1: 15.339168765361949, 2: 21.284356659374016, 3: 21.304356618648765},
            "imec1.lf": {0: 15.319321358082725, 1: 15.339321516521915, 2: 21.284568614155827, 3: 21.30456877259502},
        }

        # Initialize the RawIO
        rawio = SpikeGLXRawIO(self.get_local_path("spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI4"))
        rawio.parse_header()

        # Get list of stream names
        stream_names = rawio.header["signal_streams"]["name"]

        # Test t_start for each stream and segment
        for stream_name, expected_values in expected_t_starts.items():
            # Get stream index
            stream_index = list(stream_names).index(stream_name)

            # Check each segment
            for seg_index, expected_t_start in expected_values.items():
                actual_t_start = rawio.get_signal_t_start(block_index=0, seg_index=seg_index, stream_index=stream_index)

                # Use numpy.testing for proper float comparison
                np.testing.assert_allclose(
                    actual_t_start,
                    expected_t_start,
                    rtol=1e-9,
                    atol=1e-9,
                    err_msg=f"Mismatch in t_start for stream '{stream_name}', segment {seg_index}",
                )


def test_build_signals_info_dict_collision_raises_value_error():
    info_a = {"seg_index": 0, "stream_name": "imec0.ap", "meta_file": "/x/first.meta"}
    info_b = {"seg_index": 0, "stream_name": "imec0.ap", "meta_file": "/x/second.meta"}

    expected_message = (
        "Two SpikeGLX file pairs resolve to the same stream 'imec0.ap' in segment 0:\n"
        "  1) /x/first.meta\n"
        "  2) /x/second.meta\n"
        "This can happen if:\n"
        "  - Files were renamed on disk. Stream names come from the 'fileName' field "
        "inside the .meta, not the filename on disk.\n"
        "  - Recordings from different sessions are in the same folder with the same "
        "gate/trigger numbers.\n"
        "  - Duplicate copies exist in subfolders (the reader scans recursively).\n"
        "  - A third-party tool rewrote the .meta file with an incorrect 'fileName' "
        "(for example, LF meta pointing to the AP binary)."
    )

    with pytest.raises(ValueError) as exc_info:
        _build_signals_info_dict([info_a, info_b])

    assert str(exc_info.value) == expected_message


if __name__ == "__main__":
    unittest.main()
