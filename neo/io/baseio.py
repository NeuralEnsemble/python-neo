 # -*- coding: utf-8 -*-
 
 
"""
Class base for IO modules.
This is a template.


 
"""





class BaseIO(object):
	"""
	Classe template to handle Input/Output for different file formats.
	
	This template is file reading/writing orented but it can also handle data from/to a database
	like TDT sytem tanks or SQLite files.
	
	Each object is able to declare what can be readed or writen
	
	
	the type can be one of the class defined in neo.core :
		Block (with all segments, AnalogSignals, SpikeTrains, ...)
		Segment (with all  AnalogSignals, SpikeTrains, Events, Epoch, ...)
		SpikeTrain
		SpikeTrainList
		AnalogSignal
		AnalogSignalList
		Neuron
	
	
	Each IOclass implementation can also add attributs (fields) freely to all object.
		
	
	Each IOclass should come with tipics files exemple.
	
	
	"""
	
	is_readable = False
	is_writable = False
	
	is_subelement_readable = False
	is_subelement_writable = False
	
	
	has_header = False
	
	is_streameable = False
	
	type = None
	
	read_params = { }
	write_params = { }
	
	nfiles = 0
	
	def __init__(self , filename = None , **kargs ) :
		pass
		
		
	
	def read(self, **kargs ):
		"""
		read all
		"""
		pass
		
	def write(self, **kargs):
		"""
		read all
		"""
		pass
	
	def read_spike(self):
		pass

	def read_analogsignal(self):
		pass

	def write_spike(self):
		pass

	def write_analogsignal(self):
		pass


	def read_header():
		pass
		
	def write_header():
		pass
	
	def get_extra_data_of( type )
	