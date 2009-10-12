import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))
from neo.core import Neuron

class NeuronTest(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testIdCreate(self):
        neuron = Neuron()
        assert neuron.id == 0
        
    def testForceIdCreate(self):
        neuron = Neuron(id=5)
        assert neuron.id == 5
        
if __name__ == "__main__":
    unittest.main()