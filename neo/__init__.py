'''
Neo is a package for representing electrophysiology data in Python,
together with support for reading a wide range of neurophysiology file formats
'''
import logging

logging_handler = logging.StreamHandler()

from neo.core import *
from neo.io import *

import importlib.metadata
__version__ = importlib.metadata.version("neo")
