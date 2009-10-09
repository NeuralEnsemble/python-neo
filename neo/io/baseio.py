# -*- coding: utf-8 -*-

"""
NeuroTools.io
==================

A collection of functions to handle all the inputs/outputs of the NeuroTools.signals
file, used by the loaders.

Classes
-------

BaseFile        - abstract class which should be overriden, managing how a file will load/write
                     its data
"""

class BaseFile(object):
	"""
	Generic class to handle all the file read/write methods for the key objects of the
    core class. This template is file reading/writing oriented but it can also handle data 
    from/to a database like TDT sytem tanks or SQLite files.
	
    This is an abstract class that will be implemented for each format
    The key methods of the class are:
        write(object)                - Write an object to a file
        read_spikes(params)          - Read Spike object from file with some params
        read_analogs(type, params)   - Read AnalogSignal object from file with some params
        read_spiketrains(params)     - Read SpikeTrain object from file with some params
        read_spiketrainlists(params) - Read SpikeTrainList object from file with some params
        read_epochs(type, params)    - Read Epoch object from file with some params
        read_events(params)          - Read Event object from file with some params
        read_blocks(type, params)    - Read Block object from file with some params
        
	Each object is able to declare what can be accessed or written
	
	the object types can be one of the class defined in neo.core :
		Block (with all segments, AnalogSignals, SpikeTrains, ...)
		Segment (with all  AnalogSignals, SpikeTrains, Events, Epoch, ...)
		SpikeTrain
		SpikeTrainList
		AnalogSignal
		AnalogSignalList
		Neuron
	
	Each IOclass implementation can also add attributs (fields) freely to all object.
	Each IOclass should come with tipics files exemple.
    
    Inputs:
        filename - the file name for reading/writing data
    
    If you want to implement your own file format, you just have to create an object that will 
    inherit from this FileHandler class and implement the previous functions. See io.py for more
    details
    """
	
	is_readable        = False
	is_writable        = False	
	is_object_readable = False
	is_object_writable = False
	has_header         = False	
	is_streameable     = False
	read_params        = {}
    write_params       = {}   
	level              = None
    nfiles             = 0        
	
	def __init__(self , filename = None , **kargs ) :
		pass
	
	def read(self, **kargs ):
		"""
        bulk read the file at the highest level possible
 		"""
		pass
		
	def write(self, **kargs):
		"""
        bulk write the file at the highest level possible
		"""
		pass
	
    ######## All individual read methods #######################
    
	def read_spikes(self, params):
        """
        Read Spikes objects from a file
        
        Examples:
        """
        return _abstract_method(self)
    
    def read_analogs(self, params):
        """
        Read AnalogSignal objects from a file
        
        Examples:

        """
        return _abstract_method(self)
    
    def read_spiketrains(self, params):
        """
        Read SpikeTrains objects from a file
        
        Examples:

        """
        return _abstract_method(self)
    
    def read_events(self, params):
        """
        Read Events objects from a file
        
        Examples:

        """
        return _abstract_method(self)
    
    def read_blocks(self, params):
        """
        Read Events objects from a file
        
        Examples:

        """
        return _abstract_method(self)
        
    def read_epochs(self, params):
        """
        Read Epochs objects from a file
        
        Examples:

        """
        return _abstract_method(self)
     
    def read_spiketrainlists(self, params):
        """
        Read SpikeTrainList objects from a file
        
        Examples:

        """
        return _abstract_method(self)

    def read_header(self):
        """
        Read metadata/header from a file
        
        Examples:

        """
        return _abstract_method(self)

######## All individual write methods #######################

	def write_spikes(self, params):
        """
        Write Spikes objects from a file
        
        Examples:
        """
        return _abstract_method(self)
    
    def write_analogs(self, params):
        """
        Write AnalogSignal objects from a file
        
        Examples:

        """
        return _abstract_method(self)
    
    def write_spiketrains(self, params):
        """
        Write SpikeTrains objects from a file
        
        Examples:

        """
        return _abstract_method(self)
    
    def write_events(self, params):
        """
        Write Events objects from a file
        
        Examples:

        """
        return _abstract_method(self)
    
    def write_blocks(self, params):
        """
        Write Events objects from a file
        
        Examples:

        """
        return _abstract_method(self)
        
    def write_epochs(self, params):
        """
        Write Epochs objects from a file
        
        Examples:

        """
        return _abstract_method(self)
     
    def write_spiketrainlists(self, params):
        """
        Write SpikeTrainList objects from a file
        
        Examples:

        """
        return _abstract_method(self)
		
	def write_header(self):
		"""
        Write metadata/header from a file
        
        Examples:

        """
        return _abstract_method(self)
	