from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.elanrawio import ElanRawIO


class ElanIO(ElanRawIO, BaseFromRaw):
    """
    Class for reading data from Elan.

    Elan is software for studying time-frequency maps of EEG data.

    Elan is developed in Lyon, France, at INSERM U821

    https://elan.lyon.inserm.fr

    Args:
        filename (string) :
            Full path to the .eeg file
        entfile (string) :
            Full path to the .ent file (optional). If None, the path to the ent
            file is inferred from the filename by adding the ".ent" extension
            to it
        posfile (string) :
            Full path to the .pos file (optional). If None, the path to the pos
            file is inferred from the filename by adding the ".pos" extension
            to it
    """
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename, entfile=None, posfile=None):
        ElanRawIO.__init__(self, filename=filename, entfile=entfile,
                           posfile=posfile)
        BaseFromRaw.__init__(self, filename)
