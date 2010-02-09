"""
===========
Spike2 Library
===========
For reading data from CED's Spike2 Son files into the NeuroTools enviroment.
The data is read from the CED files using sonpy.

Jens Kremkow
INCM-CNRS, Marseille, France
ALUF, Freiburg, Germany
2008


sonpy is written by:

Antonio Gonzalez
Department of Neuroscience
Karolinska Institutet
Antonio.Gonzalez@cantab.net

http://www.neuro.ki.se/broberger/

###########################################

Usage:

single_channel = load('filename',channels=[2])
dict_of_all_channels = load('filename')

"""

from sonpy import son

__all__ = ['spike2channels']
