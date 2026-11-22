"""3RScan pretraining dataset wrapper."""

from ..build import DATASET_REGISTRY
from .base import ScanBase
import pdb


@DATASET_REGISTRY.register()
class RScanSpatialRefer(ScanBase):
    """3RScan spatial-refer dataset used for pretraining."""

    def __init__(self, cfg, split):
        """Load 3RScan language data and scan metadata."""
        super(RScanSpatialRefer, self).__init__(cfg, split)
        self.base_dir = cfg.data.rscan_base#'dataset/3RScan'
        #pdb.set_trace()
        ##self.__class__.__name__='RScanSpatialRefer',split='pretrain'
        split_cfg = cfg.data.get(self.__class__.__name__).get(split)#split_cfg={'sources': ['3rscan_view_cap']}
        #pdb.set_trace()
        
        all_scan_ids = self._load_split(self.split) #self.split='pretrain',len(all_scan_ids)=1204
        #pdb.set_trace()
        #len(self.lang_data)=38528, len(self.ground_lang_data)=38528, len(self.scan_ids)=1204
        self.lang_data, self.ground_lang_data, self.scan_ids = self._load_lang(split_cfg, all_scan_ids)##self.lang_data和self.ground_lang_data 完全相同
        #pdb.set_trace()
        
        self.scan_data = self._load_scan_pretrain(self.lang_data, self.ground_lang_data)
        #pdb.set_trace()

        print(f"Loading 3RScan {split}-set scans")
        print(f"Finish loading 3RScan {split}-set scans")

    def __len__(self):
        """Return the number of scene-level samples."""
        return len(self.scan_data)

    def __getitem__(self, index):
        """Return one pretraining sample."""
        return self._getitem_refer(index)
