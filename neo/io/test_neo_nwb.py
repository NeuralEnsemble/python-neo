import nwbio
from nwbio import *

filename = "/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb"

io = nwbio.NWBIO(filename)
#io = pynwb.NWBHDF5IO(filename, mode='r') # Open a file with NWBHDF5IO
#container = io.read() # Define the file as a NWBFile object
#print("container = ", container)

#io.__init__("/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb")

# Test the entire file
io.read_block()

# Tests the different functions
#io._handle_general_group(block='') 
#io._handle_epochs_group(block='') 
#io._handle_acquisition_group(False, block='')
#io._handle_stimulus_group(False, block='')
#io._handle_processing_group(block='')
#io._handle_analysis_group(block='')

#io._handle_timeseries('index_000', True, 1)

#get_units(container.data)

