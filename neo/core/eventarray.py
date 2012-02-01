from neo.core.baseneo import BaseNeo

import numpy as np
import quantities as pq

class EventArray(BaseNeo):
    """
    Array of events. Introduced for performance reasons.
    An :class:`EventArray` is prefered to a list of :class:`Event` objects.
    
    *Usage*:
        TODO
    
    *Required attributes/properties*:
        :times: (quantity array 1D)
        :labels: (numpy.array 1D dtype='S')
    
    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:            
    
    """
    def __init__(self, times=np.array([])*pq.s, labels=np.array([], dtype='S'),
                 name=None, description=None, file_origin=None,
                 **annotations):
        """Initialize a new EventArray."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        self.times = times
        self.labels = labels
        
        self.segment = None

    def __repr__(self):
        return "<EventArray: %s>" % ", ".join('%s@%s' % item for item in zip(self.labels, self.times))
