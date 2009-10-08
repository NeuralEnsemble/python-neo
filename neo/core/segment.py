# -*- coding: utf-8 -*-



class Segment(object):
	"""
	
	
	**Definition**
	Container gathering discrete or continous data acquired during the same time lapse.
	In short, a :class:`Segment` may contain :class:`AnalogSignal`, :class:`SpikeTrain`, :class:`Event` and :class:`Epoch` 
	that share the same time base.
	
	**Usage**
	
	
	**Example**
	
	seg = Segment()
	
	seg.get_analogsignals()
	
	seg.get_spiketrains()
	
	seg.get_events()
	
	seg.get_epochs()
	
	"""
	
	def __init__(self, *arg , **karg):
		pass
	
	def get_segments(self):
		"""
		Return  :calss:`AnalogSignalList`.
		"""
		pass
		
	def get_spiketrains(self):
		"""
		Return :class:`SpikeTrainList`.
		"""
		pass
		
	def get_events(self):
		"""
		Return a list of :class:`Event`.
		"""
		pass
	
	def get_epochs(self):
		"""
		Return a list of :class:`Epoch`.
		"""
		pass	




