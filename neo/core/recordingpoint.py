# -*- coding: utf-8 -*-

class RecordingPoint(object):     
    """
    Define a particular position for recordings

    **Definition**
    A :class:`RecordingPoint` is a physical location identifying the recorded data. It can
    for example be the position of the Eletrode.

    with arguments:
    
    ``id``

    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):
        self._analogsignals = [ ]
        if 'analogsignals' in karg :
            self._analogsignals = analogsignals