"""
Tests of neo.rawio.examplerawio

Note for dev:
if you write a new RawIO class your need to put some file
to be tested at g-node portal, Ask neuralensemble list for that.
The file need to be small.

Then you have to copy/paste/renamed the TestExampleRawIO
class and a full test will be done to test if the new coded IO
is compliant with the RawIO API.

If you have problems, do not hesitate to ask help github (prefered)
of neuralensemble list.

Note that same mechanism is used a neo.io API so files are tested
several time with neo.rawio (numpy buffer) and neo.io (neo object tree).
See neo.test.iotest.*


Author: Samuel Garcia

"""

import logging
import tempfile
import unittest

from pathlib import Path, PureWindowsPath

from neo.rawio.alphaomegarawio import AlphaOmegaRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


logging.getLogger().setLevel(logging.INFO)


class TestAlphaOmegaRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = AlphaOmegaRawIO

    entities_to_download = [
        "alphaomega",
    ]

    entities_to_test = [
        "alphaomega/mpx_map_version4",
    ]

    def setUp(self):
        super().setUp()
        self.logger = logging.getLogger("neo.rawio.alphaomegarawio.AlphaOmegaRawIO")

    def test_explore_folder_all_mpx(self):
        """We just check that we index all *.lsx files and that all *.mpx files
        are referenced somewhere. We should maybe check that *.lsx files are
        correctly read but it seems like just duplicating source code."""
        path = Path(self.get_local_path("alphaomega/mpx_map_version4"))
        reader = AlphaOmegaRawIO(dirname=path)
        reader._explore_folder()
        all_mpx = sorted(list(path.glob("*.mpx")))
        self.assertSequenceEqual(all_mpx, reader._mpx_files)

    def test_explore_lsx(self):
        """We only load files referenced in lsx file"""
        lsx_file = "i211119-0002.lsx"
        path = Path(self.get_local_path("alphaomega/mpx_map_version4"))
        reader = AlphaOmegaRawIO(dirname=path, lsx_files=[lsx_file])
        reader._explore_folder()
        other_mpx = set(path.glob("*.mpx"))
        with open(path / lsx_file) as f:
            for line in f:
                mpx_file = path / PureWindowsPath(line.strip()).name
                other_mpx.discard(mpx_file)
                with self.subTest(mpx_file=mpx_file):
                    self.assertIn(mpx_file, reader._mpx_files)
        for f in other_mpx:
            with self.subTest(other_mpx=f):
                self.assertNotIn(f, reader._mpx_files)

    def test_explore_no_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # just create a temporary folder that is removed
            pass
        with self.assertRaisesRegex(ValueError, "is not a folder"):
            reader = AlphaOmegaRawIO(dirname=tmpdir)

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertLogs(logger=self.logger, level="ERROR") as cm:
                reader = AlphaOmegaRawIO(dirname=tmpdir)
        self.assertIn("Found no AlphaOmega *.mpx files in", cm.output[0])

    def test_read_file_datablocks(self):
        """Superficial test that check it returns all types of channels"""
        path = Path(self.get_local_path("alphaomega/mpx_map_version4"))
        reader = AlphaOmegaRawIO(dirname=path)
        first_mpx = list(path.glob("*.mpx"))[0]
        (
            metadata,
            continuous_analog_channels,
            segmented_analog_channels,
            digital_channels,
            channel_type,
            stream_data_channels,
            ports,
            events,
            unknown_blocks,  # empty by default
        ) = reader._read_file_datablocks(first_mpx, prune_channels=False)
        self.assertIn("application_version", metadata)
        self.assertIn("application_name", metadata)
        self.assertIn("record_date", metadata)
        self.assertIn("start_time", metadata)
        self.assertIn("stop_time", metadata)
        self.assertIn("erase_count", metadata)
        self.assertIn("map_version", metadata)
        self.assertIn("resource_version", metadata)
        self.assertIn("max_sample_rate", metadata)

        for channel_id, channel in continuous_analog_channels.items():
            with self.subTest(channel_id=channel_id):
                self.assertEqual(channel_type[channel_id], "continuous_analog")
                self.assertIn("spike_color", channel)
                self.assertIn("bit_resolution", channel)
                self.assertIn("sample_rate", channel)
                self.assertIn("spike_count", channel)
                self.assertIn("mode_spike", channel)
                self.assertIn("duration", channel)
                self.assertIn("gain", channel)
                self.assertIn("name", channel)
                self.assertIn("positions", channel)

        for channel_id, channel in segmented_analog_channels.items():
            with self.subTest(channel_id=channel_id):
                self.assertEqual(channel_type[channel_id], "segmented_analog")
                self.assertIn("spike_color", channel)
                self.assertIn("bit_resolution", channel)
                self.assertIn("sample_rate", channel)
                self.assertIn("spike_count", channel)
                self.assertIn("mode_spike", channel)
                self.assertIn("pre_trig_duration", channel)
                self.assertIn("post_trig_duration", channel)
                self.assertIn("level_value", channel)
                self.assertIn("trg_mode", channel)
                self.assertIn("automatic_level_base_rms", channel)
                self.assertIn("gain", channel)
                self.assertIn("name", channel)
                self.assertIn("positions", channel)

        for channel_id, channel in digital_channels.items():
            with self.subTest(channel_id=channel_id):
                self.assertEqual(channel_type[channel_id], "digital")
                self.assertIn("spike_color", channel)
                self.assertIn("sample_rate", channel)
                self.assertIn("save_trigger", channel)
                self.assertIn("duration", channel)
                self.assertIn("prev_status", channel)
                self.assertIn("name", channel)
                self.assertIn("samples", channel)

        for channel_id, stream in stream_data_channels.items():
            with self.subTest(channel_id=channel_id):
                self.assertIn("sample_rate", stream)
                self.assertIn("name", stream)

        for port_id, port in ports.items():
            with self.subTest(port_id=port_id):
                self.assertIn("board_number", port)
                self.assertIn("sample_rate", port)
                self.assertIn("prev_value", port)
                self.assertIn("name", port)
                self.assertIn("samples", port)

        self.assertIsInstance(events, list)

        self.assertFalse(unknown_blocks)  # unknown blocks are ignored by default

    def test_read_unknown_datablocks(self):
        path = Path(self.get_local_path("alphaomega/mpx_map_version4"))
        reader = AlphaOmegaRawIO(dirname=path)
        reader._ignore_unknown_datablocks = False
        first_mpx = list(path.glob("*.mpx"))[0]
        *_, unknown_blocks = reader._read_file_datablocks(first_mpx, prune_channels=False)
        self.assertTrue(unknown_blocks)

    def test_read_file_datablocks_prune(self):
        """Check that pruning keep only channels with recorded data"""
        path = Path(self.get_local_path("alphaomega/mpx_map_version4"))
        reader = AlphaOmegaRawIO(dirname=path)
        first_mpx = list(path.glob("*.mpx"))[0]
        (
            metadata,
            continuous_analog_channels,
            segmented_analog_channels,
            digital_channels,
            channel_type,
            stream_data_channels,
            ports,
            events,
            _,  # ignore unknown_blocks
        ) = reader._read_file_datablocks(first_mpx, prune_channels=True)

        for channel_id, channel in continuous_analog_channels.items():
            with self.subTest(channel_id=channel_id):
                self.assertTrue(channel["positions"])

        for channel_id, channel in segmented_analog_channels.items():
            with self.subTest(channel_id=channel_id):
                self.assertTrue(channel["positions"])

        for channel_id, channel in digital_channels.items():
            with self.subTest(channel_id=channel_id):
                self.assertTrue(channel["samples"])

        for port_id, port in ports.items():
            with self.subTest(port_id=port_id):
                self.assertTrue(port["samples"])

    def test_read_file_datablocks_no_prune(self):
        """Check that we keep empty channels when pruning is False"""
        path = Path(self.get_local_path("alphaomega/mpx_map_version4"))
        reader = AlphaOmegaRawIO(dirname=path)
        mpx_file = Path(
            self.get_local_path("alphaomega/mpx_map_version4/mapfile0054.mpx")
        )
        (
            metadata,
            continuous_analog_channels,
            segmented_analog_channels,
            digital_channels,
            channel_type,
            stream_data_channels,
            ports,
            events,
            _,  # ignore unknown_blocks
        ) = reader._read_file_datablocks(mpx_file, prune_channels=False)

        self.assertFalse(
            all(c["positions"] for c in continuous_analog_channels.values())
        )
        self.assertFalse(
            all(c["positions"] for c in segmented_analog_channels.values())
        )
        self.assertFalse(all(c["samples"] for c in digital_channels.values()))
        self.assertFalse(all(c["samples"] for c in ports.values()))

    def test_correct_number_of_blocks_and_segments(self):
        """We just check that when we read test data we get what we expect"""
        reader = AlphaOmegaRawIO(
            dirname=self.get_local_path("alphaomega/mpx_map_version4")
        )
        reader.parse_header()
        nb_blocks = 1
        self.assertEqual(reader.block_count(), nb_blocks)
        nb_segments = [3]
        for block_index in range(nb_blocks):
            with self.subTest(block_index=block_index):
                self.assertEqual(
                    reader.segment_count(block_index), nb_segments[block_index]
                )

        nb_streams = 5
        self.assertEqual(reader.signal_streams_count(), nb_streams)
        nb_channels = [3, 8, 4, 4, 4]
        for stream_index in range(nb_streams):
            with self.subTest(stream_index=stream_index):
                self.assertEqual(
                    reader.signal_channels_count(stream_index),
                    nb_channels[stream_index],
                )

        nb_event_channels = 5
        self.assertEqual(reader.event_channels_count(), nb_event_channels)
        nb_events = [
            [
                [160, 0, 0, 0, 0],
                [0, 2, 2, 2, 2],
                [0, 0, 0, 0, 0],
            ],
        ]
        for block_index in range(nb_blocks):
            for seg_index in range(nb_segments[block_index]):
                for event_index in range(nb_event_channels):
                    with self.subTest(
                        block_index=block_index,
                        segment_index=seg_index,
                        event_index=event_index,
                    ):
                        self.assertEqual(
                            reader.event_count(block_index, seg_index, event_index),
                            nb_events[block_index][seg_index][event_index],
                        )


if __name__ == "__main__":
    unittest.main()
