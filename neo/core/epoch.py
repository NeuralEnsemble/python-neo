from neo.core.baseneo import BaseNeo
import quantities as pq

class Epoch(BaseNeo):
    """
    Similar to :class:`Event` but with a duration.
    Useful for describing a period, the state of a subject, ...
    
    *Usage*:
        TODO
    
    *Required attributes/properties*:
        :time: (quantity)
        :duration: (quantity)
        :label: (str)
    
    Recommended attributes/properties:
        :name:
        :description:
        :file_origin:
    """
    
    def __init__(self, time, duration, label, name=None, description=None,
                 file_origin=None, **annotations):
        """Initialize a new Epoch."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        self.time = time
        self.duration = duration
        self.label = label
        
        self.segment = None

