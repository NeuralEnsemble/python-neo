import unittest
import tempfile
import warnings
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from neo.io import MaxwellIO
from neo.rawio.maxwellrawio import MaxwellRawIO, auto_install_maxwell_hdf5_compression_plugin
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestMaxwellRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):

    rawioclass = MaxwellRawIO
    entities_to_download = ["maxwell"]
    entities_to_test = files_to_test = [
        "maxwell/MaxOne_data/Record/000011/data.raw.h5",
        "maxwell/MaxTwo_data/Network/000028/data.raw.h5",
    ]

    def setUp(self):
        auto_install_maxwell_hdf5_compression_plugin(force_download=False)


@unittest.skipIf(h5py is None, "requires h5py")
class TestMaxwellRawIOSynthetic(unittest.TestCase):
    def _check_mapping(self, mapping, routed_ids, channel_ids, electrode_ids, rows, legacy=False):
        samples = np.array([[10 * (i + 1) + j for j in range(4)] for i in range(len(routed_ids))], dtype="uint16")
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "recording.h5"
            with h5py.File(filename, "w") as file:
                file["version"] = [b"20160704" if legacy else b"20190530"]
                if legacy:
                    settings = file.create_group("settings")
                    file["sig"] = samples
                    mapping_group = file
                else:
                    recording = file.create_group("wells/well000/rec0000")
                    settings = recording.create_group("settings")
                    mapping_group = settings
                    routed = recording.create_group("groups/routed")
                    routed["channels"] = routed_ids
                    routed["raw"] = samples
                settings["sampling"] = [20000.0]
                settings["lsb"] = [1e-6]
                mapping_group["mapping"] = np.array(mapping, dtype=[("channel", "i4"), ("electrode", "i4")])

            reader = MaxwellRawIO(filename=str(filename))
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reader.parse_header()
                channels = reader.header["signal_channels"]
                self.assertEqual(channels["id"].tolist(), [str(channel) for channel in channel_ids])
                self.assertEqual(
                    channels["name"].tolist(),
                    [f"ch{channel} elec{electrode}" for channel, electrode in zip(channel_ids, electrode_ids)],
                )
                duplicated_ids = {
                    channel
                    for channel, _ in mapping
                    if channel >= 0 and sum(other == channel for other, _ in mapping) > 1
                }
                if duplicated_ids.intersection(routed_ids) and not legacy:
                    self.assertTrue(any("multiple electrode mappings" in str(w.message) for w in caught))
                if not channel_ids:
                    self.assertEqual(reader.signal_streams_count(), 0)
                    self.assertEqual(reader.header["signal_buffers"].size, 0)
                expected = samples[rows, :].T
                if channel_ids:
                    np.testing.assert_array_equal(reader.get_analogsignal_chunk(), expected)
                    np.testing.assert_array_equal(reader.get_analogsignal_chunk(i_start=1, i_stop=3), expected[1:3])
                    np.testing.assert_array_equal(
                        reader.get_analogsignal_chunk(channel_indexes=slice(0, 1)), expected[:, :1]
                    )
                    for i, channel in enumerate(channel_ids):
                        np.testing.assert_array_equal(
                            reader.get_analogsignal_chunk(channel_ids=[str(channel)]),
                            expected[:, i : i + 1],
                        )
                    if len(channel_ids) > 1:
                        np.testing.assert_array_equal(
                            reader.get_analogsignal_chunk(channel_indexes=[len(channel_ids) - 1, 0]),
                            expected[:, [-1, 0]],
                        )
            finally:
                reader.h5_file.close()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                io = MaxwellIO(filename=str(filename))
            try:
                segment = io.read_segment()
                if channel_ids:
                    np.testing.assert_array_equal(segment.analogsignals[0].magnitude, expected)
                else:
                    self.assertEqual(len(segment.analogsignals), 0)
            finally:
                io.h5_file.close()

    def test_routed_channel_mapping(self):
        cases = [
            ([(2, 20), (5, 50)], [2, 5], [2, 5], [20, 50], [0, 1]),
            ([(2, 20), (5, 50)], [2, 8, 5], [2, 5], [20, 50], [0, 2]),
            ([(2, 20), (5, 50)], [8, 5, 2], [5, 2], [50, 20], [1, 2]),
            ([(8, 80)], [2, 5], [], [], []),
            ([(2, 20), (2, 21), (5, 50)], [2, 5], [5], [50], [1]),
            ([(2, 20), (2, 21), (5, 50)], [5, 2], [5], [50], [0]),
            ([(-1, 99), (2, 20), (2, 21), (5, 50), (8, 80)], [2, 5], [5], [50], [1]),
            ([(8, 80), (5, 50), (5, 51), (2, 20), (-1, 99)], [5, 2], [2], [20], [1]),
            ([(2, 20), (5, 20)], [5, 2], [5, 2], [20, 20], [0, 1]),
            ([(2, 20), (2, 21), (5, 50), (5, 51)], [2, 5], [], [], []),
            ([(2, 20), (5, 50), (-1, 99)], [-1, 5, 2], [5, 2], [50, 20], [1, 2]),
            ([(2, 20), (5, 50), (8, 80), (8, 81)], [5, 2], [5, 2], [50, 20], [0, 1]),
            (
                [
                    (399, 25732),
                    (431, 25732),
                    (399, 13364),
                    (431, 13364),
                    (9, 90),
                    (7, 70),
                    (5, 50),
                ],
                [9, 399, 7, 431, 5],
                [9, 7, 5],
                [90, 70, 50],
                [0, 2, 4],
            ),
        ]
        for mapping, routed_ids, channel_ids, electrode_ids, rows in cases:
            with self.subTest(mapping=mapping, routed_ids=routed_ids):
                self._check_mapping(mapping, routed_ids, channel_ids, electrode_ids, rows)

    def test_legacy_channel_mapping(self):
        for mapping in [[(0, 20), (1, 50)], [(-1, 99), (0, 20), (1, 50)]]:
            with self.subTest(mapping=mapping):
                self._check_mapping(mapping, [0, 1], [0, 1], [20, 50], [0, 1], legacy=True)


if __name__ == "__main__":
    unittest.main()
