"""
IgorProIO Demo
===========================

"""

import os
from urllib.request import urlretrieve
import zipfile
import matplotlib.pyplot as plt
from neo.io import get_io


# Downloaded from Human Brain Project Collaboratory
# Digital Reconstruction of Neocortical Microcircuitry (nmc-portal)
# http://microcircuits.epfl.ch/#/animal/8ecde7d1-b2d2-11e4-b949-6003088da632
datafile_url = "https://microcircuits.epfl.ch/data/released_data/B95.zip"
filename_zip = "B95.zip"
filename = "grouped_ephys/B95/B95_Ch0_IDRest_107.ibw"
urlretrieve(datafile_url, filename_zip)

zip_ref = zipfile.ZipFile(filename_zip)  # create zipfile object
zip_ref.extract(path=".", member=filename)  # extract file to dir
zip_ref.close()


reader = get_io(filename)
signal = reader.read_analogsignal()
plt.plot(signal.times, signal)
plt.xlabel(signal.sampling_period.dimensionality)
plt.ylabel(signal.dimensionality)
