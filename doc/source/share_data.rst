===========================================
Sharing neuroscience data in an open format
===========================================

.. FAIR, advantages of open formats

.. data from other formats, or from simulations


NIX
===



Neurodata Without Borders (NWB)
===============================

`Neurodata Without Borders`_ (NWB:N) is an open standard file format for neurophysiology.

Neo's :class:`~neo.io.NWBIO` module can read NWB 2.0-format files, and maps their structure
onto Neo objects and annotations.
Neo support for NWB is a work-in-progress, it does not currently support NWB extensions for example.
If you encounter a problem reading an NWB file with Neo, please make a `bug report`_ (see :doc:`bug_reports`).

:class:`~neo.io.NWBIO` can also write to NWB 2.0 format.
Since NWB has a more complex structure than Neo's basic :class:`Block` - :class:`Segment` hierarchy,
and NWB requires fairly extensive metadata, it is recommended to annotate the Neo objects with special,
NWB-specific annotations, to ensure data and metadata are correctly placed within the NWB file.

As an example, let's use a public dataset, "`Whole cell patch-clamp recordings of cerebellar granule cells`_",
contributed to the EBRAINS_ repository by Marialuisa Tognolina from the laboratory of Egidio D'Angelo at the University of Pavia.

As we can see from the dataset description,

    *This dataset provides a characterization of the intrinsic excitability and synaptic properties of the cerebellar granule cells.
    Whole-cell patch-clamp recordings were performed on acute parasagittal cerebellar slices obtained from juvenile Wistar rats (p18-p24).
    Passive granule cells parameters were extracted in voltage-clamp mode by analyzing current relaxation induced by step voltage changes (IV protocol).
    Granule cells intrinsic excitability was investigated in current-clamp mode by injecting 2 s current steps (CC step protocol).
    Synaptic transmission properties were investigated in current clamp mode by an electrical stimulation of the mossy fibers bundle (5 pulses at 50 Hz, EPSP protocol).*

The dataset contains recordings from multiple subjects. For this example. let's download the data for Subject 15. You can download them by hand, by selecting each file then selecting "Download file", or run the following code:


.. ipython::

    In [1]: from urllib.request import urlretrieve

    In [2]: from urllib.parse import quote

    In [2]: dataset_url = "https://object.cscs.ch/v1/AUTH_63ea6845b1d34ad7a43c8158d9572867/hbp-d000017_PatchClamp-GranuleCells_pub"

    In [3]: folder = "GrC_Subject15_180116"

    In [4]: filenames = ["180116_0004 IV -70.abf", "180116_0005 CC step.abf", "180116_0006 EPSP.abf"]

    In [5]: for filename in filenames:
       ...:     datafile_url = f"{dataset_url}/{folder}/{quote(filename)}"
       ...:     local_file = urlretrieve(datafile_url, filename)

Let's start with the current-clamp data.

.. ipython::

    In [6]: from neo.io import get_io

    In [7]: reader = get_io("180116_0005 CC step.abf")

    In [8]: data = reader.read()

    In [9]: data
    Out[9]:

We can see that the file contains a single :class:`Block`, containing 15 :class:`Segments`,
and each segment contains one :class:`AnalogSignal` with a single channel, and an :class:`Event`.

.. note: the events are essentially empty

To quickly take a look at the data, let's plot it:

.. ipython::

    In [10]: import matplotlib.pyplot as plt

    In [11]: fig = plt.figure(figsize=(10, 5))

    In [12]: for segment in data[0].segments:
       ....:     signal = segment.analogsignals[0]
       ....:     plt.plot(signal.times, signal)

    In [13]: plt.xlabel(f"Time ({signal.times.units.dimensionality.string})")

    In [14]: plt.ylabel(f"Voltage ({signal.units.dimensionality.string})")

    In [15]: plt.savefig("source/nwb_example_cc_step.png")

.. image:: nwb_example_cc_step.png

NWB files can contain a lot of metadata. File-level metadata should be attached to the Neo :class:`Block`.
Here we take metadata from the dataset description on the EBRAINS search portal:

.. ipython::

    In [16]: global_metadata = {
       ....:     "session_start_time": data[0].rec_datetime,
       ....:     "identifier": data[0].file_origin,
       ....:     "session_id": "180116_0005",
       ....:     "institution": "University of Pavia",
       ....:     "lab": "D'Angelo Lab",
       ....:     "related_publications": "https://doi.org/10.1038/s42003-020-0953-x"
       ....: }

    In [17]: data[0].annotate(**global_metadata)

The location of data stored in an NWB file depends on the source of the data, e.g. whether they are stimuli,
intracellular electrophysiology recordings, extracellular electrophysiology recordings, behavioural measuremenets, etc.
For this, we need to annotate all data objects with special metadata, identified by keys starting with "``nwb_``":

.. ipython::

    In [18]: signal_metadata = {
       ....:     "nwb_group": "acquisition",
       ....:     "nwb_neurodata_type": "icephys.PatchClampSeries",
       ....:     "nwb_electrode": {
       ....:         "device": {
       ....:            "name": "patch clamp electrode"
       ....:         }
       ....:     }
       ....: }

    In [19]: for segment in data[0].segments:
       ....:     signal = segment.analogsignals[0]
       ....:     signal.annotate(**signal_metadata)

Now that we have annotated our dataset, we can write it to an NWB file:

.. ipython::

    In [20]: from neo.io import NWBIO

    In [21]: writer = NWBIO("GrC_Subject15_180116.nwb", mode="w")

    In [22]: writer.write(data[0])



.. _`Neurodata Without Borders`: https://www.nwb.org
.. _`bug report`: https://github.com/NeuralEnsemble/python-neo/issues/new
.. _`Whole cell patch-clamp recordings of cerebellar granule cells`: https://doi.org/10.25493/CHJG-7QC
.. _EBRAINS: https://ebrains.eu/services/data-and-knowledge/
