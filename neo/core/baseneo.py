class BaseNeo(object):
    """
    Bunch: do the same in the other sense
    Working now:
      myneoobject.hi = 2
      >>>myneoobject._annotations['hi'] = 2
    Have to work in the future:
      myneoobject._annotations['hi'] = 3
      >>>myneoobject.hi = 3
    """

    def __init__(self, *args, **kwargs):
        self.__dict__['_annotations'] = {}

    # not working and break properties of inherited objects
    """def __getattr__(self, k):
        if k in self.__dict__.keys():
            return super(BaseNeo, self).__getattr__(k)
        elif hasattr(self, '_annotations'):
            if k in self._annotations.keys():
                return self._annotations[k]
        else: 
            raise AttributeError(str(k) + ' is not available in this object')

    def __setattr__(self, k, v):
        if hasattr(self, '_annotations'):
            if k in self._annotations.keys():
                self._annotations[k] = v
        else:
            super(BaseNeo, self).__setattr__(k, v)

    def __delattr__(self, k):
        if k in self._annotations.keys():
            del self._annotations[k]
        else:
            super(BaseNeo, self).__delattr__(k)
    """


