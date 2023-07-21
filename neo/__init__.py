'''
Neo is a package for representing electrophysiology data in Python,
together with support for reading a wide range of neurophysiology file formats
'''
import importlib.metadata
# this need to be at the begining because some sub module will need the version
__version__ = importlib.metadata.version("neo")

import logging

logging_handler = logging.StreamHandler()

from neo.core import *
from neo.io import *

