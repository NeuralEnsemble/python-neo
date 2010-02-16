# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import NeuroshareTdt2IO
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class NeuroshareSpike2IOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        pass


if __name__ == "__main__":
    unittest.main()


