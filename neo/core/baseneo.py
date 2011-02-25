class BaseNeo(object):
    """
    Bunch: do the same in the other sense
    Working now:
      myneoobject.hi = 2
      >>>myneoobject._anotations['hi'] = 2
    Have to work in the future:
      myneoobject._anotations['hi'] = 3
      >>>myneoobject.hi = 3
    """
    def __new__(cls):
        cls._anotations = {}
        return cls

    def __setattr__(self, k, v):
        if not k in self.__dict__.keys():
            # raise Exception / Warning ?
            self._anotations[k] = v
        self.__dict__[k] = v

    def __delattr__(self, k):
        if k in self._anotations.keys():
            # raise Exception / Warning ? 
            del self._anotations[k]
        del self.__dict__[k]



