# -*- coding: utf-8 -*-

from epoch import Epoch

definition = """

"""

class Event(Epoch):
    definition = """An :class:`Event` is an object to represent ponctual time event.
    Useful for managing trigger, stimulus, ..."""
    
    __doc__ = """
    Object to represent ponctual time event.

    **Definition**"""+definition+"""
    

    with arguments:
    
    ``time`` The time of the Event
    
    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):
        Epoch.__init__(self, *arg, **karg)
        self.duration = 0
        if 'time' in karg.keys():
            self.time = karg['time']



