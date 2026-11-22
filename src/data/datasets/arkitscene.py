"""ARKitScenes pretraining dataset wrapper."""

from ..build import DATASET_REGISTRY
from .base import ScanBase
import pdb


@DATASET_REGISTRY.register()
class ARKitSceneSpatialRefer(ScanBase):
    """ARKitScenes spatial-refer dataset used for pretraining."""

    def __init__(self, cfg, split):
        """Load ARKitScenes language data and scan metadata."""
        super(ARKitSceneSpatialRefer, self).__init__(cfg, split)
        #pdb.set_trace()
        
        self.base_dir = cfg.data.arkitscene_base#'dataset/Arkitscenes'

        split_cfg = cfg.data.get(self.__class__.__name__).get(split)#{'sources': ['arkitscenes_view_cap']}
        #pdb.set_trace()
        
        
        all_scan_ids = self._load_split(self.split)#self.split='pretrain'
        #pdb.set_trace()

        self.lang_data, self.ground_lang_data, self.scan_ids = self._load_lang(split_cfg, all_scan_ids)
        #pdb.set_trace()
        
        self.scan_data = self._load_scan_pretrain(self.lang_data, self.ground_lang_data)

        print(f"Loading ARKitScene {split}-set scans")
        print(f"Finish loading ARKitScene {split}-set scans")

    def __len__(self):
        """Return the number of scene-level samples."""
        return len(self.scan_data)

    def __getitem__(self, index):
        """Return one pretraining sample."""
        return self._getitem_refer(index)
