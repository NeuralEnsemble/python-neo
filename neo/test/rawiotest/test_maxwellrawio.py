import unittest
import tempfile
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

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
    def _check_mapping(self, mapping, channel_ids, electrode_ids, legacy=False):
        samples = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype="uint16")
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
                    routed["channels"] = channel_ids
                    routed["raw"] = samples
                settings["sampling"] = [20000.0]
                settings["lsb"] = [1e-6]
                mapping_group["mapping"] = np.array(mapping, dtype=[("channel", "i4"), ("electrode", "i4")])

            reader = MaxwellRawIO(filename=str(filename))
            try:
                reader.parse_header()
                channels = reader.header["signal_channels"]
                self.assertEqual(channels["id"].tolist(), [str(channel) for channel in channel_ids])
                self.assertEqual(
                    channels["name"].tolist(),
                    [f"ch{channel} elec{electrode}" for channel, electrode in zip(channel_ids, electrode_ids)],
                )
                np.testing.assert_array_equal(reader.get_analogsignal_chunk(), samples.T)
                np.testing.assert_array_equal(
                    reader.get_analogsignal_chunk(channel_ids=[str(channel_ids[1])]), samples[1:, :].T
                )
            finally:
                reader.h5_file.close()

    def test_routed_channel_mapping(self):
        cases = [
            ([(2, 20), (5, 50)], [2, 5], [20, 50]),
            ([(2, 20), (2, 21), (5, 50)], [2, 5], [20, 50]),
            ([(2, 20), (2, 21), (5, 50)], [5, 2], [50, 20]),
            ([(-1, 99), (2, 20), (2, 21), (5, 50), (8, 80)], [2, 5], [20, 50]),
            ([(8, 80), (5, 50), (5, 51), (2, 20), (-1, 99)], [5, 2], [50, 20]),
        ]
        for mapping, channel_ids, electrode_ids in cases:
            with self.subTest(mapping=mapping, channel_ids=channel_ids):
                self._check_mapping(mapping, channel_ids, electrode_ids)

    def test_legacy_channel_mapping(self):
        for mapping in [[(0, 20), (1, 50)], [(-1, 99), (0, 20), (1, 50)]]:
            with self.subTest(mapping=mapping):
                self._check_mapping(mapping, [0, 1], [20, 50], legacy=True)


if __name__ == "__main__":
    unittest.main()
