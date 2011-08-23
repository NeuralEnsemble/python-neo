# encoding: utf-8
"""
Common tests for IOs:
 * check presence of all necessary attr
 * check types
 * write/read consistency  


"""
url_for_tests =  "https://portal.g-node.org/neo/"


import os
import urllib

import neo
from neo.test.tools import assert_arrays_almost_equal, assert_arrays_equal, assert_same_sub_schema
#~ from neo import description
from neo.test.io.generate_datasets import generate_from_supported_objects



def test_write_then_read(ioclass):
    """
    Test for IO that are able to write and read:
      1 - Generate a full schema with supported objects.
      2 - Write to a file
      3 - Read from the file
      4 - Check the hierachy
      5 - Check data
    
    Work only for IO for Block and Segment for the higher object (main cases).
    """
    higher = ioclass.supported_objects[0]
    if not(higher in ioclass.readable_objects and higher in ioclass.writeable_objects):
        return
    if not(higher == neo.Block or higher == neo.Segment):
        return
    
    filename = 'test_io_'+ioclass.name
    if len(ioclass.extensions)>=1:
        filename += '.'+ioclass.extensions[0]
    writer = ioclass(filename = filename)
    reader = ioclass(filename = filename)
    
    ob = generate_from_supported_objects(ioclass.supported_objects)
    if higher == neo.Block:
        writer.write_block(ob)
        ob2 = reader.read_block()
    elif higher == neo.Segment:
        writer.write_segment(ob)
        ob2 = reader.read_segment()
    
    assert_same_sub_schema(ob, ob2)

def test_read_then_write(ioclass):
    """
    Test for IO that are able to read and write:
     1 - Read a file
     2 Write object set in another file
     3 Compare the 2 files hash
     
    """
    pass
    


def download_test_files_if_not_present(ioclass, filenames ):
    shortname = ioclass.__name__.lower().strip('io')
    localdir = os.path.dirname(__file__)+'/files_for_tests'
    if not os.path.exists(localdir):
        os.mkdir(localdir)
    localdir = localdir +'/'+ shortname
    if not os.path.exists(localdir):
        os.mkdir(localdir)
    
    url = url_for_tests+shortname
    for filename in filenames:
        localfile =  localdir+'/'+filename
        distantfile = url+'/'+filename
        if not os.path.exists(localfile):
            print 'Downloading ', distantfile, 'here ', localfile
            urllib.urlretrieve(distantfile, localfile)
    
    return localdir
    




    
    
