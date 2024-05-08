"""
Converting to NWB using Neo
===========================

"""

from urllib.request import urlretrieve
from urllib.parse import quote
from neo.io import get_io


dataset_url = "https://object.cscs.ch/v1/AUTH_63ea6845b1d34ad7a43c8158d9572867/hbp-d000017_PatchClamp-GranuleCells_pub"

folder = "GrC_Subject15_180116"

# filenames = ["180116_0004 IV -70.abf", "180116_0005 CC step.abf", "180116_0006 EPSP.abf"]
filenames = ["180116_0005 CC step.abf"]

for filename in filenames:
    datafile_url = f"{dataset_url}/{folder}/{quote(filename)}"
    local_file = urlretrieve(datafile_url, filename)


reader = get_io("180116_0005 CC step.abf")
data = reader.read()

global_metadata = {
    "session_start_time": data[0].rec_datetime,
    "identifier": data[0].file_origin,
    "session_id": "180116_0005",
    "institution": "University of Pavia",
    "lab": "D'Angelo Lab",
    "related_publications": "https://doi.org/10.1038/s42003-020-0953-x",
}

# data[0].annotate(**global_metadata)

signal_metadata = {
    "nwb_group": "acquisition",
    "nwb_neurodata_type": ("pynwb.icephys", "PatchClampSeries"),
    "nwb_electrode": {
        "name": "patch clamp electrode",
        "description": "The patch-clamp pipettes were pulled from borosilicate glass capillaries "
        "(Hilgenberg, Malsfeld, Germany) and filled with intracellular solution "
        "(K-gluconate based solution)",
        "device": {"name": "patch clamp electrode"},
    },
    "nwb:gain": 1.0,
}

for segment in data[0].segments:
    signal = segment.analogsignals[0]
    signal.annotate(**signal_metadata)

from neo.io import NWBIO

writer = NWBIO("GrC_Subject15_180116.nwb", mode="w", **global_metadata)
writer.write(data)
