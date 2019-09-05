# -*- coding: utf-8 -*-
"""
Common tools that are useful for neo.io object tests
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import logging
import os
import shutil
import tempfile

from neo.core import Block, Segment
from neo.test.generate_datasets import generate_from_supported_objects


def close_object_safe(obj):
    """
    Close an object safely, ignoring errors

    For some io types, like HDF5IO, the file should be closed before being
    opened again in a test.  Call this after the test is done to make sure
    the file is closed.
    """
    try:
        obj.close()
    except:
        pass


def cleanup_test_file(mode, path, directory=None):
    """
    Remove test files or directories safely.  mode is the mode of the io class,
    either 'file' or 'directory'.  It can also be an io class object, or any
    other object with a 'mode' attribute.  If that is the case, use the
    'mode' attribute from the object.

    If directory is not None and path is not an absolute path already,
    use the file from the given directory.
    """
    if directory is not None and not os.path.isabs(path):
        path = os.path.join(directory, path)
    if hasattr(mode, 'mode'):
        mode = mode.mode
    if mode == 'file':
        if os.path.exists(path):
            os.remove(path)
    elif mode == 'dir':
        if os.path.exists(path):
            shutil.rmtree(path)


def get_test_file_full_path(ioclass, filename=None,
                            directory=None, clean=False):
    """
    Get the full path for a file of the given filename.

    If filename is None, create a filename.

    If filename is a list, get the full path for each item in the list.

    If return_path is True, also return the full path to the file.

    If directory is not None and path is not an absolute path already,
    use the file from the given directory.

    If return_path is True, return the full path of the file along with
    the io object.  return reader, path.  Default is False.

    If clean is True, try to delete existing versions of the file
    before creating the io object.  Default is False.
    """
    # create a filename if none is provided
    if filename is None:
        filename = 'Generated0_%s' % ioclass.__name__
        if (ioclass.mode == 'file' and len(ioclass.extensions) >= 1):
            filename += '.' + ioclass.extensions[0]
    elif not hasattr(filename, 'lower'):
        return [get_test_file_full_path(ioclass, filename=fname,
                                        directory=directory, clean=clean) for
                fname in filename]

    # if a directory is provided add it
    if directory is not None and not os.path.isabs(filename):
        filename = os.path.join(directory, filename)

    if clean:
        cleanup_test_file(ioclass, filename)

    return filename


# prevent this being treated as a test if imported into a test file
get_test_file_full_path.__test__ = False


def create_generic_io_object(ioclass, filename=None, directory=None,
                             return_path=False, clean=False):
    """
    Create an io object in a generic way that can work with both
    file-based and directory-based io objects

    If filename is None, create a filename.

    If return_path is True, also return the full path to the file.

    If directory is not None and path is not an absolute path already,
    use the file from the given directory.

    If return_path is True, return the full path of the file along with
    the io object.  return reader, path.  Default is False.

    If clean is True, try to delete existing versions of the file
    before creating the io object.  Default is False.
    """
    filename = get_test_file_full_path(ioclass, filename=filename,
                                       directory=directory, clean=clean)
    try:
        # actually create the object
        if ioclass.mode == 'file':
            ioobj = ioclass(filename=filename)
        elif ioclass.mode == 'dir':
            ioobj = ioclass(dirname=filename)
        else:
            ioobj = None
    except:
        print(filename)
        raise

    # return the full path if requested, otherwise don't
    if return_path:
        return ioobj, filename
    return ioobj


def iter_generic_io_objects(ioclass, filenames, directory=None,
                            return_path=False, clean=False):
    """
    Return an iterable over the io objects created from a list of filenames.

    The objects are automatically cleaned up afterwards.

    If directory is not None and path is not an absolute path already,
    use the file from the given directory.

    If return_path is True, yield the full path of the file along with
    the io object.  yield reader, path.  Default is False.

    If clean is True, try to delete existing versions of the file
    before creating the io object.  Default is False.
    """
    for filename in filenames:
        ioobj, path = create_generic_io_object(ioclass, filename=filename,
                                               directory=directory,
                                               return_path=True,
                                               clean=clean)

        if ioobj is None:
            continue
        if return_path:
            yield ioobj, path
        else:
            yield ioobj
        close_object_safe(ioobj)


def create_generic_reader(ioobj, target=None, readall=False):
    """
    Create a function that can read the target object from a file.

    If target is None, use the first supported_objects from ioobj
    If target is False, use the 'read' method.
    If target is the Block or Segment class, use read_block or read_segment,
    respectively.
    If target is a string, use 'read_'+target.

    If readall is True, use the read_all_ method instead of the read_ method.
    Default is False.
    """
    if target is None:
        target = ioobj.supported_objects[0].__name__

    if target == Block:
        if readall:
            return ioobj.read_all_blocks
        return ioobj.read_block
    elif target == Segment:
        if readall:
            return ioobj.read_all_segments
        return ioobj.read_segment
    elif not target:
        if readall:
            raise ValueError('readall cannot be True if target is False')
        return ioobj.read
    elif hasattr(target, 'lower'):
        if readall:
            return getattr(ioobj, 'read_all_%ss' % target.lower())
        return getattr(ioobj, 'read_%s' % target.lower())


def iter_generic_readers(ioclass, filenames, directory=None, target=None,
                         return_path=False, return_ioobj=False,
                         clean=False, readall=False):
    """
    Iterate over functions that can read the target object from a list of
    filenames.

    If target is None, use the first supported_objects from ioobj
    If target is False, use the 'read' method.
    If target is the Block or Segment class, use read_block or read_segment,
    respectively.
    If target is a string, use 'read_'+target.

    If directory is not None and path is not an absolute path already,
    use the file from the given directory.

    If return_path is True, return the full path of the file along with
    the reader object.  return reader, path.

    If return_ioobj is True, return the io object as well as the reader.
    return reader, ioobj.  Default is False.

    If both return_path and return_ioobj is True,
    return reader, path, ioobj.  Default is False.

    If clean is True, try to delete existing versions of the file
    before creating the io object.  Default is False.

    If readall is True, use the read_all_ method instead of the read_ method.
    Default is False.
    """
    for ioobj, path in iter_generic_io_objects(ioclass=ioclass,
                                               filenames=filenames,
                                               directory=directory,
                                               return_path=True,
                                               clean=clean):
        res = create_generic_reader(ioobj, target=target, readall=readall)
        if not return_path and not return_ioobj:
            yield res
        else:
            res = (res,)

        if return_path:
            res = res + (path,)
        if return_ioobj:
            res = res + (ioobj,)
        yield res


def create_generic_writer(ioobj, target=None):
    """
    Create a function that can write the target object to a file using the
    neo io object ioobj.

    If target is None, use the first supported_objects from ioobj
    If target is False, use the 'write' method.
    If target is the Block or Segment class, use write_block or write_segment,
    respectively.
    If target is a string, use 'write_'+target.
    """
    if target is None:
        target = ioobj.supported_objects[0].__name__

    if target == Block:
        return ioobj.write_block
    elif target == Segment:
        return ioobj.write_segment
    elif not target:
        return ioobj.write
    elif hasattr(target, 'lower'):
        return getattr(ioobj, 'write_' + target.lower())


def read_generic(ioobj, target=None, lazy=False, readall=False,
                 return_reader=False):
    """
    Read the target object from a file using the given neo io object ioobj.

    If target is None, use the first supported_objects from ioobj
    If target is False, use the 'write' method.
    If target is the Block or Segment class, use write_block or write_segment,
    respectively.
    If target is a string, use 'write_'+target.

    The lazy parameters is passed to the reader.  Defaults is True.

    If readall is True, use the read_all_ method instead of the read_ method.
    Default is False.

    If return_reader is True, yield the io reader function as well as the
    object. yield obj, reader.  Default is False.
    """
    obj_reader = create_generic_reader(ioobj, target=target, readall=readall)
    obj = obj_reader(lazy=lazy)
    if return_reader:
        return obj, obj_reader
    return obj


def iter_read_objects(ioclass, filenames, directory=None, target=None,
                      return_path=False, return_ioobj=False,
                      return_reader=False, clean=False, readall=False,
                      lazy=False):
    """
    Iterate over objects read from a list of filenames.

    If target is None, use the first supported_objects from ioobj
    If target is False, use the 'read' method.
    If target is the Block or Segment class, use read_block or read_segment,
    respectively.
    If target is a string, use 'read_'+target.

    If directory is not None and path is not an absolute path already,
    use the file from the given directory.

    If return_path is True, yield the full path of the file along with
    the object.  yield obj, path.

    If return_ioobj is True, yield the io object as well as the object.
    yield obj, ioobj.  Default is False.

    If return_reader is True, yield the io reader function as well as the
    object. yield obj, reader.  Default is False.

    If some combination of return_path, return_ioobj, and return_reader
    is True, they are yielded in the order: obj, path, ioobj, reader.

    If clean is True, try to delete existing versions of the file
    before creating the io object.  Default is False.

    The lazy parameter is passed to the reader.  Defaults is True.

    If readall is True, use the read_all_ method instead of the read_ method.
    Default is False.
    """
    for obj_reader, path, ioobj in iter_generic_readers(ioclass, filenames,
                                                        directory=directory,
                                                        target=target,
                                                        return_path=True,
                                                        return_ioobj=True,
                                                        clean=clean,
                                                        readall=readall):
        obj = obj_reader(lazy=lazy)
        if not return_path and not return_ioobj and not return_reader:
            yield obj
        else:
            obj = (obj,)

        if return_path:
            obj = obj + (path,)
        if return_ioobj:
            obj = obj + (ioobj,)
        if return_reader:
            obj = obj + (obj_reader,)
        yield obj


def write_generic(ioobj, target=None, obj=None, return_writer=False):
    """
    Write the target object to a file using the given neo io object ioobj.

    If target is None, use the first supported_objects from ioobj
    If target is False, use the 'write' method.
    If target is the Block or Segment class, use write_block or write_segment,
    respectively.
    If target is a string, use 'write_'+target.

    obj is the object to write.  If obj is None, an object is created
    automatically for the io class.

    If return_writer is True, yield the io writer function as well as the
    object. yield obj, writer.  Default is False.
    """
    if obj is None:
        supported_objects = ioobj.supported_objects
        obj = generate_from_supported_objects(supported_objects)
    obj_writer = create_generic_writer(ioobj, target=target)
    obj_writer(obj)
    if return_writer:
        return obj, obj_writer
    return obj
