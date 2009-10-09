# -*- coding: utf-8 -*-

class Event(Epoch):
     
    """
    Object to represent ponctual time event.

    **Definition**
    An :class:`Event` is a discrete evenement arriving at time t. It inherits from the
    :class:`Epoch` object, and can be viewed as an :class:`Epoch` without any duration

    with arguments:
    
    ``time`` The time of the Event
    
    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):        
        Epoch.__init__(self, arg, kwarg)
        self.duration = 0