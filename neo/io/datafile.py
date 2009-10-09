# -*- coding: utf-8 -*-

"""
NeuroTools.io
==================

A collection of functions to handle all the inputs/outputs of the NeuroTools.signals
file, used by the loaders.

Classes
-------

FileHandler        - abstract class which should be overriden, managing how a file will load/write
                     its data
"""


from NeuroTools import check_dependency

import os, numpy


class BaseFile(object):
    """
    Class to handle all the file read/write methods for the key objects of the
    signals class, i.e SpikeList and AnalogSignalList. Could be extented
    
    This is an abstract class that will be implemented for each format (txt, pickle, hdf5)
    The key methods of the class are:
        write(object)              - Write an object to a file
        read_spikes(params)        - Read a SpikeList file with some params
        read_analogs(type, params) - Read an AnalogSignalList of type `type` with some params
    
    Inputs:
        filename - the file name for reading/writing data
    
    If you want to implement your own file format, you just have to create an object that will 
    inherit from this FileHandler class and implement the previous functions. See io.py for more
    details
    """
    
    def __init__(self, filename):
        self.filename = filename
        self.fileobj  = open(self.filename, 'r', DEFAULT_BUFFER_SIZE)
        self.metadata = {}
    
    def __str__(self):
        return "%s (%s)" % (self.__class__.__name__, self.filename)
    
    def __del__(self):
        self.close()

    def close(self):
        self.fileobj.close()       
    
    def write(self, object):
        """
        Write the object to the file. 
        
        Examples:
            >> handler.write(SpikeListObject)
            >> handler.write(VmListObject)
        """
        return _abstract_method(self)
    
    def read_spikes(self, params):
        """
        Read a SpikeList object from a file and return the SpikeList object, created from the File and
        from the additional params that may have been provided
        
        Examples:
            >> params = {'id_list' : range(100), 't_stop' : 1000}
            >> handler.read_spikes(params)
                SpikeList Object (with params taken into account)
        """
        return _abstract_method(self)
    
    def read_analogs(self, type, params):
        """
        Read an AnalogSignalList object from a file and return the AnalogSignalList object of type 
        `type`, created from the File and from the additional params that may have been provided
        
        `type` can be in ["vm", "current", "conductance"]
        
        Examples:
            >> params = {'id_list' : range(100), 't_stop' : 1000}
            >> handler.read_analogs("vm", params)
                VmList Object (with params taken into account)
            >> handler.read_analogs("current", params)
                CurrentList Object (with params taken into account)
        """
        return _abstract_method(self)