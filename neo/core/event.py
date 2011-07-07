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
    # Add event to segment
    e = Event(time=3.4, label='Flash')
    seg._events.append(e)
    print seg.get_events()[0].label
    """ % definition
    
    def __init__(self, *arg, **karg):
        """Initializes as Epoch but with duration zero."""
        Epoch.__init__(self, *arg, **karg)
        self.duration = 0
        if 'time' in karg.keys():
            self.time = karg['time']



