# -*- coding: utf-8 -*-



class Block(object):
	"""
	Top level container for data.
	
	**Definition**
	Main container gathering all the data discrete or continous for a given setup.
	It can be view as a list of :class:`Segment`.
	
	A block is not necessary a homogeneous recorging contrary to :class:`Segment`
	
	**Usage**
	
	
	**Example**
	
	
	bl = Block( segments = [seg1 , seg2 , seg3] )
	
	bl.get_segments()
	
	
	"""
	
	def __init__(self, *arg , **karg):
		pass
