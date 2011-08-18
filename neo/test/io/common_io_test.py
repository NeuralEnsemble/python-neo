# encoding: utf-8
"""
Common tests for IOs:
 * check presence of all necessary attr
 * check types
 * write/read consistency  


"""
import os

import neo
from neo.test.tools import assert_arrays_almost_equal, assert_arrays_equal, assert_same_sub_schema
#~ from neo import description
from neo.test.io.generate_datasets import generate_from_supported_objects



def test_write_them_read(ioclass):
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



#~ def test_the_same_schema(ob1, ob2):
    #~ """
    #~ Test if ob1 and ob2 has the same sub schema.
    #~ Explore all one_to_many_relationship
    #~ """
    #~ if type(ob1) != type(ob2):
        #~ return False
    #~ _type = type(ob1)
    #~ classname =description.name_by_class[_type]
    
    #~ if classname not in description.one_to_many_reslationship:
        #~ return
    #~ for child in description.one_to_many_reslationship[classname]:
        #~ if not hasattr(ob1, '_'+child+'s'):
            #~ assert not hasattr(ob2, '_'+child+'s'), '%s 2 do have %s but not %s 1'%(classname, child, classname)
            #~ continue
        #~ else:
            #~ assert hasattr(ob2, '_'+child+'s'), '%s 1 have %s but not %s 2'%(classname, child, classname)
        
        #~ sub1 = getattr(ob1, '_'+child+'s')
        #~ sub2 = getattr(ob2, '_'+child+'s')
        #~ assert len(sub1) == len(sub2), 'theses two %s have not the same %s number'%(classname, child)
        #~ for i in range(len(getattr(ob1, '_'+child+'s'))):
            #~ test_the_same_schema(sub1[i], sub2[i])
            
    
    #~ # many_to_many_reslationship is not tested because of infinite recursive loops
    
    
    
    
    



    
    
