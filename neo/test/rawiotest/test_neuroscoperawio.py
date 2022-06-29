import unittest
import logging
from pathlib import Path

from neo.rawio.neuroscoperawio import NeuroScopeRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO
from neo.test.rawiotest import rawio_compliance as compliance
from neo.utils.datasets import get_local_testing_data_folder


class TestNeuroScopeRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = NeuroScopeRawIO
    entities_to_download = [
        'neuroscope'
    ]
    entities_to_test = [
        'neuroscope/test1/test1',
        'neuroscope/test1/test1.dat',
        'neuroscope/dataset_1/YutaMouse42-151117.eeg'
    ]

    def test_binary_argument_with_non_canonical_xml_file(self):

        local_test_dir = get_local_testing_data_folder()
        filename = local_test_dir / "neuroscope/test2/recording.xml"
        binary_file = local_test_dir / "neuroscope/test2/signal1.dat"
        reader = NeuroScopeRawIO(filename=filename, binary_file=binary_file)

        msg = 'Before parser_header() no header information should be present'
        assert reader.header is None, msg

        reader.parse_header()

        # After the file resolution test that the right file is being loaded
        assert reader.data_file_path == binary_file
        assert reader.xml_file_path == filename

        txt = reader.__repr__()
        msg = 'After parser_header() nb_block should be known'
        assert 'nb_block' in txt, msg
        txt = reader._repr_annotations()

        # launch a series of test compliance
        compliance.header_is_total(reader)
        compliance.count_element(reader)
        compliance.read_analogsignals(reader)
        compliance.read_spike_times(reader)
        compliance.read_spike_waveforms(reader)
        compliance.read_events(reader)
        compliance.has_annotations(reader)



if __name__ == "__main__":
    unittest.main()
