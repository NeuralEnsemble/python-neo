# encoding: utf-8
"""
Common tests for IOs:
 * check presence of all necessary attr
 * check types
 * write/read consistency  


"""
__test__ = False #

url_for_tests =  "https://portal.g-node.org/neo/"

import os
import urllib
import logging

import neo
from neo.test.tools import assert_arrays_almost_equal, assert_arrays_equal, assert_same_sub_schema, assert_neo_object_is_compliant

from neo.test.io.generate_datasets import generate_from_supported_objects


class BaseTestIO(object):
    # subclasses must define self.ioclass attribute

    def test_write_then_read(self):
        """
        Test for IO that are able to write and read:
          1 - Generate a full schema with supported objects.
          2 - Write to a file
          3 - Read from the file
          4 - Check the hierachy
          5 - Check data
        
        Work only for IO for Block and Segment for the higher object (main cases).
        """
        higher = self.ioclass.supported_objects[0]
        if not(higher in self.ioclass.readable_objects and higher in self.ioclass.writeable_objects):
            return
        if not(higher == neo.Block or higher == neo.Segment):
            return
        
        # when io need external knowldge for writting or read such as sampling_rate
        # the test is too much complex too design genericaly
        if len(self.ioclass.read_params[higher]) != 0 : return
        
        if self.ioclass.mode == 'file':
            filename = 'test_io_'+self.ioclass.__name__
            if len(self.ioclass.extensions)>=1:
                filename += '.'+self.ioclass.extensions[0]
            writer = self.ioclass(filename = filename)
            reader = self.ioclass(filename = filename)
        elif self.ioclass.mode == 'dir':
            dirname = 'test_io_'+self.ioclass.__name__
            writer = self.ioclass(dirname = dirname)
            reader = self.ioclass(dirname = dirname)
        else:
            return
        
        ob = generate_from_supported_objects(self.ioclass.supported_objects)
        if higher == neo.Block:
            writer.write_block(ob)
            ob2 = reader.read_block()
        elif higher == neo.Segment:
            writer.write_segment(ob)
            ob2 = reader.read_segment()
        
        assert_same_sub_schema(ob, ob2)

    def test_read_then_write(self):
        """
        Test for IO that are able to read and write:
         1 - Read a file
         2 Write object set in another file
         3 Compare the 2 files hash
         
        """
        pass
        #TODO
    
    def test_download_test_files_if_not_present(self ):
        """
        Download file at G-node for testing
        url_for_tests is global at beginning of this file.
        
        """
        if not hasattr(self,'files_to_download'):
            self.files_to_download = self.files_to_test
        
        shortname = self.ioclass.__name__.lower().strip('io')
        localdir = os.path.dirname(__file__)+'/files_for_tests'
        if not os.path.exists(localdir):
            os.mkdir(localdir)
        localdir = localdir +'/'+ shortname
        if not os.path.exists(localdir):
            os.mkdir(localdir)
        
        url = url_for_tests+shortname
        for filename in self.files_to_download:
            localfile =  localdir+'/'+filename
            distantfile = url+'/'+filename
            if not os.path.exists(localfile):
                logging.info('Downloading %s here %s' % (distantfile, localfile))
                urllib.urlretrieve(distantfile, localfile)
        
        self.local_test_dir = localdir
        
    def test_assert_readed_neo_object_is_compliant(self):
        self.test_download_test_files_if_not_present()
        
        for filename in self.files_to_test:
            
            filename = os.path.join(self.local_test_dir, filename)
            if self.ioclass.mode == 'file':
                r = self.ioclass(filename = filename)
            elif self.ioclass.mode == 'dir':
                # TODO in TDT
                r = self.ioclass(dirname = filename)
            else:
                continue
            
            ob = r.read()
            assert_neo_object_is_compliant(ob)





    
