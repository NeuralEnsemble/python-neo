class BaseNeo(object):
    """
    Bunch: do the same in the other sense
    Working now:
      myneoobject.hi = 2
      >>>myneoobject._annotations['hi'] = 2
    Have to work in the future:
      myneoobject._annotations['hi'] = 3
      >>>myneoobject.hi = 3
    
    # I (SAM) like the idea of creating an instance with any keyword:
    myneoobject( mywife = 'therese')
    
      
      
    """

    def __init__(self, *args, **kwargs):
        """This is the base constructor for all neo objects.
        
        For definitions of keyword arguments, see the class documentation
        for the specific object, ie `neo.AnalogSignal`.
        """
        self.__dict__['_annotations'] = kwargs
        #self.__dict__['_annotations'] = { }

    def __getattr__(self, k):
        if hasattr(self, '_annotations'):
            if k in self._annotations.keys():
                return self._annotations[k]
        return self.__dict__[k]

    # The __setattr__ method does problems with properties of the inherited objects

    def annotate(self, **annotations):
        """
        Add annotations (non-standardized metadata) to a Neo object.
        
        Example:
        
        >>> obj.annotate(key1=value1, key2=value2)
        >>> obj.key2
        value2
        """
        self._annotations.update(annotations)
