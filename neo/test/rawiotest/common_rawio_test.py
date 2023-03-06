'''
Common tests for RawIOs:

It is copy/paste from neo/test/iotests/common_io_test.py

The code should be shared for common parts.


The public URL is in url_for_tests.

To deposite new testing files,  please create a account at
gin.g-node.org and upload files at NeuralEnsemble/ephy_testing_data
data repo.


'''

__test__ = False

import logging
import unittest

from neo.utils.datasets import (download_dataset,
    get_local_testing_data_folder, default_testing_repo)

from neo.test.rawiotest.tools import can_use_network

from neo.test.rawiotest import rawio_compliance as compliance

try:
    import datalad
    HAVE_DATALAD = True
except:
    HAVE_DATALAD = False

# url_for_tests = "https://portal.g-node.org/neo/" #This is the old place
repo_for_test = default_testing_repo


class BaseTestRawIO:
    '''
    This class make common tests for all IOs.

    Basically download files from G-node portal.
    And test the IO is working.

    '''
    # ~ __test__ = False

    # all IO test need to modify this:
    rawioclass = None  # the IOclass to be tested

    entities_to_test = []  # list of files to test compliances
    entities_to_download = []  # when files are at gin

    # allow environment to tell avoid using network
    use_network = can_use_network()

    local_test_dir = get_local_testing_data_folder()

    def setUp(self):
        '''
        Set up the test fixture.  This is run for every test
        '''
        self.shortname = self.rawioclass.__name__.lower().replace('rawio', '')

        if HAVE_DATALAD:
            for remote_path in self.entities_to_download:
                download_dataset(repo=repo_for_test, remote_path=remote_path)
        else:
            raise unittest.SkipTest("Requires datalad download of data from the web")

    def get_local_base_folder(self):
        return get_local_testing_data_folder()

    def get_local_path(self, sub_path):
        root_local_path = self.get_local_base_folder()
        local_path = root_local_path / sub_path
        # TODO later : remove the str when all IOs handle the pathlib.Path objects
        local_path = str(local_path)
        return local_path

    def test_read_all(self):
        # Read all file in self.entities_to_test
        if not HAVE_DATALAD:
            return

        for entity_name in self.entities_to_test:
            # entity_name = self.get_filename_path(entity_name)
            # local path is a folder or a file
            local_path = self.get_local_path(entity_name)

            if self.rawioclass.rawmode.endswith('-file'):
                reader = self.rawioclass(filename=local_path)
            elif self.rawioclass.rawmode.endswith('-dir'):
                reader = self.rawioclass(dirname=local_path)

            txt = reader.__repr__()
            assert 'nb_block' not in txt, 'Before parser_header() nb_block should be NOT known'

            reader.parse_header()

            txt = reader.__repr__()
            assert 'nb_block' in txt, 'After parser_header() nb_block should be known'
            # print(txt)

            #
            txt = reader._repr_annotations()
            # reader.print_annotations()

            # lanch a series of test compliance
            compliance.header_is_total(reader)
            compliance.count_element(reader)
            compliance.read_analogsignals(reader)
            compliance.read_spike_times(reader)
            compliance.read_spike_waveforms(reader)
            compliance.read_events(reader)
            compliance.has_annotations(reader)

            # basic benchmark
            level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.INFO)
            compliance.benchmark_speed_read_signals(reader)
            logging.getLogger().setLevel(level)
