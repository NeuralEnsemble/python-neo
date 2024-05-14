import unittest

from neo.rawio.intanrawio import IntanRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestIntanRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = IntanRawIO
    entities_to_download = ["intan"]
    entities_to_test = [
        "intan/intan_rhs_test_1.rhs",
        "intan/intan_rhd_test_1.rhd",
        "intan/intan_fpc_test_231117_052630/info.rhd",
        "intan/intan_fps_test_231117_052500/info.rhd",
        "intan/intan_fpc_rhs_test_240329_091637/info.rhs",
        "intan/intan_fps_rhs_test_240329_091536/info.rhs",
    ]


    def test_annotations(self):
        
        intan_reader = IntanRawIO(filename=self.get_local_path("intan/intan_rhd_test_1.rhd"))
        intan_reader.parse_header()
        
        raw_annotations = intan_reader.raw_annotations
        annotations = raw_annotations["blocks"][0]["segments"][0]  # Intan is mono segment
        signal_annotations = annotations["signals"][0]  # As in the other exmaples, annotaions are duplicated
        
        
        exepcted_annotations =  {'intan_version': '1.5', 'desired_impedance_test_frequency': 1000.0, 'desired_upper_bandwidth': 7500.0, 'note1': '', 'notch_filter_mode': 1, 'notch_filter': False, 'nb_signal_group': 7, 
                            'dsp_enabled': 1, 'actual_impedance_test_frequency': 1000.0, 'desired_lower_bandwidth': 0.1, 'note3': '', 'actual_dsp_cutoff_frequency': 1.165828, 
                            'desired_dsp_cutoff_frequency': 1.0, 'actual_lower_bandwidth': 0.0945291, 'eval_board_mode': 0, 'note2': '', 'num_temp_sensor_channels': 0}
        
        for key in exepcted_annotations:
            if isinstance(exepcted_annotations[key], float):
                self.assertAlmostEqual(signal_annotations[key], exepcted_annotations[key], places=2)
            else:
                self.assertEqual(signal_annotations[key], exepcted_annotations[key])

   
if __name__ == "__main__":
    unittest.main()
