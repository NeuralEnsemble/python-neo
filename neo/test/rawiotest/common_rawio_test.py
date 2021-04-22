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

import os
import logging
import unittest

#~ from neo.test.rawiotest.tools import (can_use_network, make_all_directories,
                                   #~ download_test_file, create_local_temp_dir)
from neo.utils import download_dataset, get_local_testing_data_folder

from neo.test.rawiotest.tools import can_use_network

from neo.test.rawiotest import rawio_compliance as compliance


# url_for_tests = "https://portal.g-node.org/neo/" #This is the old place
#~ url_for_tests = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/"
repo_for_test = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

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

    local_test_dir = None

    def setUp(self):
        '''
        Set up the test fixture.  This is run for every test
        '''
        self.shortname = self.rawioclass.__name__.lower().replace('rawio', '')
        
        for remote_path in self.entities_to_download:
            download_dataset(repo=repo_for_test, remote_path=remote_path)

        #~ self.create_local_dir_if_not_exists()
        #~ self.download_test_files_if_not_present()

    #~ def create_local_dir_if_not_exists(self):
        #~ '''
        #~ Create a local directory to store testing files and return it.

        #~ The directory path is also written to self.local_test_dir
        #~ '''
        #~ self.local_test_dir = create_local_temp_dir(self.shortname)
        #~ return self.local_test_dir

    #~ def download_test_files_if_not_present(self):
        #~ '''
        #~ Download %s file at G-node for testing
        #~ url_for_tests is global at beginning of this file.

        #~ ''' % self.rawioclass.__name__

        #~ if not self.use_network:
            #~ raise unittest.SkipTest("Requires download of data from the web")

        #~ url = url_for_tests + self.shortname
        #~ try:
            #~ make_all_directories(self.files_to_download, self.local_test_dir)
            #~ download_test_file(self.files_to_download,
                               #~ self.local_test_dir, url)
        #~ except OSError as exc:
            #~ raise unittest.SkipTest(exc)

    #~ download_test_files_if_not_present.__test__ = False

    #~ def cleanup_file(self, path):
        #~ '''
        #~ Remove test files or directories safely.
        #~ '''
        #~ cleanup_test_file(self.rawioclass, path, directory=self.local_test_dir)

    #~ def get_filename_path(self, filename):
        #~ '''
        #~ Get the path to a filename in the current temporary file directory
        #~ '''
        #~ return os.path.join(self.local_test_dir, filename)
    
    def get_local_base_folder(self):
        return get_local_testing_data_folder()
        
    def get_local_path(self, sub_path):
        root_local_path = self.get_local_base_folder()
        local_path = root_local_path / sub_path
        # TODO later : remove the str when all IOs handle the Path stuff
        local_path = str(local_path)
        return local_path
    
    def get_filename_path(self, filename):
        # keep for backward compatibility
        # will be removed soon
        classname = self.__class__.__name__
        classname = classname.replace('Test', '').replace('RawIO', '').lower()
        root_local_path = self.get_local_base_folder()
        local_path = root_local_path / classname / filename
        # TODO later : remove the str when all IOs handle the Path stuff
        local_path = str(local_path)
        print('get_filename_path will be removed (use get_local_path() instead)', self.__class__.__name__, local_path)
        return local_path

    def test_read_all(self):
        # Read all file in self.entities_to_test

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
            # ~ print(txt)

            #
            txt = reader._repr_annotations()
            # ~ reader.print_annotations()

            # ~ sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
            # ~ i_start=None, i_stop=None, channel_indexes=[1])
            # ~ import matplotlib.pyplot as plt
            # ~ fig, ax = plt.subplots()
            # ~ ax.plot(sigs[:, 0])
            # ~ plt.show()

            # ~ nb_unit = reader.spike_channels_count()
            # ~ for unit_index in range(nb_unit):
            # ~ wfs = reader.spike_raw_waveforms(block_index=0, seg_index=0,
            # ~ unit_index=unit_index)
            # ~ if wfs is not None:
            # ~ import matplotlib.pyplot as plt
            # ~ fig, ax = plt.subplots()
            # ~ ax.plot(wfs[:, 0, :50].T)
            # ~ plt.show()

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
