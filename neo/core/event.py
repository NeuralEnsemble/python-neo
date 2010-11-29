# -*- coding: utf-8 -*-

from epoch import Epoch

definition = """

"""

class Event(Epoch):
    definition = """An :class:`Event` is an object to represent a point in time.
    Useful for managing triggers, stimuli, ..."""
    
    __doc__ = """
    Object to represent an event.

    **Definition**
    %s
    

    with arguments:
    
    ``time`` The time of the Event
    
    **Usage**

    **Example**

    """ % definition
    
    def __init__(self, *arg, **karg):
        Epoch.__init__(self, *arg, **karg)
        self.duration = 0
        if 'time' in karg.keys():
            self.time = karg['time']



