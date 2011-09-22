class BaseNeo(object):
    """This is the base class from which all Neo objects inherit.
    
    This class implements support for universally recommended arguments,
    and also sets up the `annotations` dict for additional arguments.
    
    The following "universal" methods are available:
        __init__ : Grabs the universally recommended arguments `name`,
            `file_origin`, `description` and stores as attributes.
            
            Also stores every additional argument (that is, every argument
            that is not handled by BaseNeo or the child class), and puts
            in the dict `annotations`.
        
        __getattr__ : Provides shortcut access by argument name to keys
            in `annotations`.
        
        annotate(**kwargs) : Updates `annotations` with keyword/value pairs.
    
    Each child class should: 
        0) call BaseNeo.__init__(self, **kwargs) with all non-recommended
            (additional) arguments.
        1) process its necessary arguments in its __new__ or __init__ method
        2) process its recommended arguments in its __new__ or __init__ method
    
    Non-keyword arguments should only be used for necessary arguments.
    
    The necessary and recommended arguments for each child class (Neo object)
    are specified in ../description.py and the documentation for the child.
    """

    def __init__(self, name=None, file_origin=None, description=None, **kwargs):
        """This is the base constructor for all Neo objects.
        
        Stores universally recommended attributes and creates `annotations`
        from additional arguments not processed by BaseNeo or the child class.
        """
        # create `annotations` for additional arguments
        # This funny syntax is to avoid recursion loops with __getattr__
        self.__dict__['annotations'] = kwargs
        
        # these attributes are recommended for all objects.
        self.name = name
        self.description = description
        self.file_origin = file_origin

    def __getattr__(self, k):
        if hasattr(self, 'annotations'):
            if k in self.annotations.keys():
                return self.annotations[k]
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
        self.annotations.update(annotations)
