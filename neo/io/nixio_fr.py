from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.nixrawio import NIXRawIO

# This class subjects to limitations when there are multiple asymmetric blocks


class NixIO(NIXRawIO, BaseFromRaw):

    name = 'NIX IO'

    _prefered_signal_group_mode = 'group-by-same-units'
    _prefered_units_group_mode = 'split-all'

    def __init__(self, filename):
        NIXRawIO.__init__(self, filename)
        BaseFromRaw.__init__(self, filename)

    def read_block(self, block_index=0, lazy=False, signal_group_mode=None,
                   units_group_mode=None, load_waveforms=False):
        bl = super(NixIO, self).read_block(block_index, lazy,
                                           signal_group_mode,
                                           units_group_mode,
                                           load_waveforms)
        for chx in bl.channel_indexes:
            if "nix_name" in chx.annotations:
                nixname = chx.annotations["nix_name"]
                chx.annotations["nix_name"] = nixname[0]
        return bl

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.header = None
        self.file.close()
