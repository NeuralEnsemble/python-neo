# -*- coding: utf-8 -*-
'''
Common tests for RawIOs:

It is copy/paste from neo/test/iotests/common_io_test.py

The code should be shared for common parts.


The public URL is in url_for_tests.

The private url for writing is
ssh://gate.g-node.org/groups/neo/io_test_files/

'''

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

__test__ = False

url_for_tests = "https://portal.g-node.org/neo/"

import os
import logging
import unittest

from neo.rawio.tests.tools import (can_use_network, make_all_directories,
        download_test_file, create_local_temp_dir)

from neo.rawio.tests import rawio_compliance as compliance

class BaseTestRawIO(object):
    '''
    This class make common tests for all IOs.
    
    Basically download files from G-node portal.
    And test the IO is working.

    '''
    #~ __test__ = False

    # all IO test need to modify this:
    rawioclass = None  # the IOclass to be tested

    files_to_test = []  # list of files to test compliances
    files_to_download = []  # when files are at G-Node

    # allow environment to tell avoid using network
    use_network = can_use_network()

    local_test_dir = None

    def setUp(self):
        '''
        Set up the test fixture.  This is run for every test
        '''
        self.shortname = self.rawioclass.__name__.lower().replace('rawio', '')
        self.create_local_dir_if_not_exists()
        self.download_test_files_if_not_present()
    
    def create_local_dir_if_not_exists(self):
        '''
        Create a local directory to store testing files and return it.

        The directory path is also written to self.local_test_dir
        '''
        self.local_test_dir = create_local_temp_dir(self.shortname)
        return self.local_test_dir

    def download_test_files_if_not_present(self):
        '''
        Download %s file at G-node for testing
        url_for_tests is global at beginning of this file.

        ''' % self.rawioclass.__name__
        
        if not self.use_network:
            raise unittest.SkipTest("Requires download of data from the web")

        url = url_for_tests+self.shortname
        try:
            make_all_directories(self.files_to_download, self.local_test_dir)
            download_test_file(self.files_to_download,
                               self.local_test_dir, url)
        except IOError as exc:
            raise unittest.SkipTest(exc)
    download_test_files_if_not_present.__test__ = False

    def cleanup_file(self, path):
        '''
        Remove test files or directories safely.
        '''
        cleanup_test_file(self.rawioclass, path, directory=self.local_test_dir)

    def get_filename_path(self, filename):
        '''
        Get the path to a filename in the current temporary file directory
        '''
        return os.path.join(self.local_test_dir, filename)
    
    def test_read_all(self):
        #Read all file in self.entities_to_test 
        
        
        for entity_name in self.entities_to_test:
            entity_name = self.get_filename_path(entity_name)
            
            if self.rawioclass.rawmode.endswith('-file'):
                reader = self.rawioclass(filename=entity_name)
            elif self.rawioclass.rawmode.endswith('-dir'):
                reader = self.rawioclass(dirname=entity_name)
            
            txt = reader.__repr__()
            assert 'nb_block' not in txt, 'Before parser_header() nb_block should be NOT known'
            
            reader.parse_header()
            
            txt = reader.__repr__()
            assert 'nb_block' in txt, 'After parser_header() nb_block should be known'
            #~ print(txt)
            
            #
            txt = reader._repr_annotations()
            #~ reader.print_annotations()
            
            #~ sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            #~ i_start=None, i_stop=None, channel_indexes=[1])
            #~ import matplotlib.pyplot as plt
            #~ fig, ax = plt.subplots()
            #~ ax.plot(sigs[:, 0])
            #~ plt.show()
            
            #~ nb_unit = reader.unit_channels_count()
            #~ for unit_index in range(nb_unit):
                #~ wfs = reader.spike_raw_waveforms(block_index=0, seg_index=0,
                                #~ unit_index=unit_index)
                #~ if wfs is not None:
                    #~ import matplotlib.pyplot as plt
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(wfs[:, 0, :50].T)
                    #~ plt.show()
            
            #lanch a series of test compliance
            compliance.header_is_total(reader)
            compliance.count_element(reader)
            compliance.read_analogsignals(reader)
            compliance.read_spike_times(reader)
            compliance.read_spike_waveforms(reader)
            compliance.read_events(reader)
            compliance.has_annotations(reader)
            
            
            
            
            
            #basic benchmark
            level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.INFO)
            compliance.benchmark_speed_read_signals(reader)
            logging.getLogger().setLevel(level)
            
            
            
            

