from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.elanrawio import ElanRawIO


class ElanIO(ElanRawIO, BaseFromRaw):
    """
    Class for reading data from Elan.

    Elan is software for studying time-frequency maps of EEG data.

    Elan is developed in Lyon, France, at INSERM U821

    https://elan.lyon.inserm.fr
    """
    _prefered_signal_group_mode = 'group-by-same-units'
    _default_group_mode_have_change_in_0_9 = True

    def __init__(self, filename):
        ElanRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
