from pathlib import Path
from neo.rawio.neuroscoperawio import NeuroScopeRawIO

# The files can be found here https://buzsakilab.nyumc.org/datasets/TingleyD/DT1/DT1_rLS_20150723_1584um/
data_path = Path("/home/heberto/ephy_testing_data/")
file_path =  data_path / "neuroscope" / "dataset_1" / "YutaMouse42-151117.eeg"
filename  = str(file_path)

neo_object = NeuroScopeRawIO(filename=filename)
neo_object.parse_header()