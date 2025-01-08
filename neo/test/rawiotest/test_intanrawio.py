import unittest
import numpy as np
from pathlib import Path

from neo.rawio.intanrawio import IntanRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestIntanRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = IntanRawIO
    entities_to_download = ["intan"]
    entities_to_test = [
        "intan/intan_rhs_test_1.rhs",  # Format header-attached
        "intan/intan_rhd_test_1.rhd",  # Format header-attached
        "intan/rhs_fpc_multistim_240514_082243/rhs_fpc_multistim_240514_082243.rhs",  # Format header-attached newer version
        "intan/intan_fpc_test_231117_052630/info.rhd",  # Format one-file-per-channel
        "intan/intan_fps_test_231117_052500/info.rhd",  # Format one file per signal
        "intan/intan_fpc_rhs_test_240329_091637/info.rhs",  # Format one-file-per-channel
        "intan/intan_fps_rhs_test_240329_091536/info.rhs",  # Format one-file-per-signal
        "intan/rhd_fpc_multistim_240514_082044/info.rhd",  # Multiple digital channels one-file-per-channel rhd
    ]

    def test_annotations(self):

        intan_reader = IntanRawIO(filename=self.get_local_path("intan/intan_rhd_test_1.rhd"))
        intan_reader.parse_header()

        raw_annotations = intan_reader.raw_annotations
        annotations = raw_annotations["blocks"][0]["segments"][0]  # Intan is mono segment
        signal_annotations = annotations["signals"][0]  # As in the other exmaples, annotaions are duplicated

        # Scalar annotations
        exepcted_annotations = {
            "intan_version": "1.5",
            "desired_impedance_test_frequency": 1000.0,
            "desired_upper_bandwidth": 7500.0,
            "note1": "",
            "notch_filter_mode": 1,
            "notch_filter": False,
            "nb_signal_group": 7,
            "dsp_enabled": 1,
            "actual_impedance_test_frequency": 1000.0,
            "desired_lower_bandwidth": 0.1,
            "note3": "",
            "actual_dsp_cutoff_frequency": 1.165828,
            "desired_dsp_cutoff_frequency": 1.0,
            "actual_lower_bandwidth": 0.0945291,
            "eval_board_mode": 0,
            "note2": "",
            "num_temp_sensor_channels": 0,
        }

        for key in exepcted_annotations:
            if isinstance(exepcted_annotations[key], float):
                self.assertAlmostEqual(signal_annotations[key], exepcted_annotations[key], places=2)
            else:
                self.assertEqual(signal_annotations[key], exepcted_annotations[key])

        # Array annotations
        signal_array_annotations = signal_annotations["__array_annotations__"]
        np.testing.assert_array_equal(signal_array_annotations["native_order"][:10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(
            signal_array_annotations["spike_scope_digital_edge_polarity"][:10], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
        np.testing.assert_array_equal(signal_array_annotations["board_stream_num"][:10], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_correct_reading_one_file_per_channel(self):
        "Issue: https://github.com/NeuralEnsemble/python-neo/issues/1599"
        # Test reading of one-file-per-channel format file. The channels should match the raw files
        file_path = Path(self.get_local_path("intan/intan_fpc_test_231117_052630/info.rhd"))
        intan_reader = IntanRawIO(filename=file_path)
        intan_reader.parse_header()

        # This should be the folder where the files of all the channels are stored
        folder_path = file_path.parent

        # The paths are named as amp-A-000.dat, amp-A-001.dat, amp-A-002.dat, ...
        amplifier_file_paths = [path for path in folder_path.iterdir() if "amp" in path.name]
        channel_names = [path.name[4:-4] for path in amplifier_file_paths]

        for channel_name, amplifier_file_path in zip(channel_names, amplifier_file_paths):
            data_raw = np.fromfile(amplifier_file_path, dtype=np.int16).squeeze()
            data_from_neo = intan_reader.get_analogsignal_chunk(channel_ids=[channel_name], stream_index=0).squeeze()
            np.testing.assert_allclose(data_raw, data_from_neo)


if __name__ == "__main__":
    unittest.main()
