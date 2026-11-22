"""Base dataset logic shared by scene-level data loaders."""

import copy
import json
import os

import jsonlines
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..data_utils import load_safetensor_from_hf
import pdb

class ScanBase(Dataset):
    """Base dataset for scene-level pretraining data."""

    def __init__(self, cfg, split):
        """Store common dataset config and split state."""
        self.cfg = cfg
        self.split = split #split='pretrain'
        self.debug = cfg.debug.flag #False
        self.debug_size = cfg.debug.debug_size #20
        #pdb.set_trace()

        assert self.split in ['pretrain', 'train', 'val', 'test']
        if self.split == 'test':
            self.split = 'val'

        self.use_scene_cap = getattr(cfg.data.args, 'use_scene_cap', False) #self.use_scene_cap=True

    def _load_split(self, split): #split='pretrain'
        """Load the split file and return sorted scan ids."""
        if 'scannet' in self.__class__.__name__.lower():#'scannetspatialrefer'
            #pdb.set_trace() #self.base_dir='dataset/ScanNet'
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_' + split + '.txt')
            #split_file='dataset/ScanNet/annotations/splits/scannetv2_pretrain.txt'
        else:
            #pdb.set_trace()
            split_file = os.path.join(self.base_dir, 'annotations/splits/' + split + '_split.txt')#'dataset/3RScan/annotations/splits/pretrain_split.txt'
        #split_file='dataset/ScanNet/annotations/splits/scannetv2_pretrain.txt'
        scan_ids = {x.strip() for x in open(split_file, 'r', encoding='utf-8')}##这里都是scene的id, 例如scene0191_00
        return sorted(scan_ids)  #len(sorted(scan_ids))=1201

    def _load_scan_pretrain(self, data, ground_lang_data):#len(data)=38296,len(ground_lang_data)=38296
        """Group per-view language entries into per-scene pretrain samples."""
        #pdb.set_trace()
        scans = []
        scan_id_to_idx = {}

        for i, item in enumerate(data):
            #pdb.set_trace()
            scan_id = item['scan_id']##'scene0000_00'
            
            
            refer_item = ground_lang_data[i]
            #pdb.set_trace()
            
            
            if scan_id not in scan_id_to_idx:
                scan_entry = {
                    'scan_id': scan_id,
                    'safetensors_path': item['safetensors_path'],
                    'sentence': [],
                    'refer_sentence': [],
                }
                #pdb.set_trace()
                
                
                scans.append(scan_entry)
                scan_id_to_idx[scan_id] = len(scans) - 1

            scan = scans[scan_id_to_idx[scan_id]]
            scan['sentence'].append(item['utterance'][:5])
            if len(refer_item['utterance']) > 0:
                scan['refer_sentence'].append(refer_item['utterance'])
            else:
                scan['refer_sentence'].append(item['utterance'][:5])
        return scans

    def _load_lang(self, cfg, scan_ids):##cfg={'sources': ['scannet_view_cap', 'ssg_ref_total_by_view_full']},scan_ids--里全是scene的id
        """Load language annotations for the requested scan ids."""
        #pdb.set_trace()
        caption_source = cfg.sources #['scannet_view_cap', 'ssg_ref_total_by_view_full']
        lang_data = []
        ground_lang_data = []
        valid_scan_ids = set()
        
        #pdb.set_trace()
        if self.use_scene_cap: #True
            scene_cap_file = os.path.join(self.base_dir, 'annotations/scene_cap.json')#self.base_dir='dataset/ScanNet'
            #pdb.set_trace() ##scene_cap.json中每个场景都有3个平行描述，第1个caption,通常更像自然语言描述/美化版场景描述;第 2个caption更像简化的场景图转文本;第 3个caption通常更像更完整的场景图关系描述;
            #scene_cap_file='dataset/ScanNet/annotations/scene_cap.json'
            
            if not os.path.exists(scene_cap_file):
                self.scene_caps = {}
                #pdb.set_trace()
            else:
                with open(scene_cap_file, 'r', encoding='utf-8') as f:
                    self.scene_caps = json.load(f) ##len(self.scene_caps)=1510
        else:
            self.scene_caps = None

        for anno_type in caption_source:
            #pdb.set_trace()
            
            
            if anno_type == 'scannet_view_cap':
                anno_file = os.path.join(self.base_dir, 'annotations/per_view_captions_scannet_v1.jsonl')
                #pdb.set_trace()##anno_file='dataset/ScanNet/annotations/per_view_captions_scannet_v1.jsonl'
                
                
                with jsonlines.open(anno_file, 'r') as handle:
                    #pdb.set_trace()
                    
                    for item in handle:##item是per_view_captions_scannet_v1.jsonl中元素{}内的
                        #pdb.set_trace()
                        
                        if item['scan_id'] in scan_ids:
                            lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])#item['scan_id']='scene0000_00'
                            
                            
                            
            elif anno_type == '3rscan_view_cap':
                #pdb.set_trace()
                anno_file = os.path.join(self.base_dir, 'annotations/per_view_captions_3rscan_v1.jsonl')
                with jsonlines.open(anno_file, 'r') as handle:
                    for item in handle:
                        if item['scan_id'] in scan_ids:
                            lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])
            elif anno_type == 'arkitscenes_view_cap':
                #pdb.set_trace()
                
                anno_file = os.path.join(self.base_dir, 'annotations/per_view_captions_arkitscenes_v1.jsonl')
                #pdb.set_trace()
                
                
                with jsonlines.open(anno_file, 'r') as handle:
                    #pdb.set_trace()
                    
                    for item in handle:
                        #pdb.set_trace()
                        
                        
                        if item['scan_id'] in scan_ids:
                            lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])
            elif 'ssg_ref' in anno_type: ##当anno_type='ssg_ref_total_by_view_full'
                refer_file = os.path.join('dataset/refer', f'{anno_type}.jsonl')#refer_file='dataset/refer/ssg_ref_total_by_view_full.jsonl'
                #pdb.set_trace()
                
                with jsonlines.open(refer_file, 'r') as handle:
                    #pdb.set_trace()
                    
                    for item in handle:
                        #pdb.set_trace()
                        if item['scan_id'] in scan_ids:
                            ground_lang_data.append(item)
                            valid_scan_ids.add(item['scan_id'])

        if len(ground_lang_data) == 0:
            for item in lang_data:
                ground_lang_data.append(copy.deepcopy(item))
        #pdb.set_trace()
        ##len(lang_data)=38296, len(ground_lang_data)=38296, len(sorted(valid_scan_ids))=1201
        return lang_data, ground_lang_data, sorted(valid_scan_ids)

    def _getitem_refer(self, index):
        """Return one pretraining sample with views and language."""
        if self.cfg.mode != 'pretrain':
            raise RuntimeError('Spatial refer downstream mode was removed because it is unused in this repo.')

        item = self.scan_data[index]
        #pdb.set_trace()
        
        scan_id = item['scan_id']#scan_id='48458202'
        scene_tensor = load_safetensor_from_hf(
            repo_id='MatchLab/ScenePoint',
            filename=item['safetensors_path'],
        )##scene_tensor里只有两个元素，'color_images'和'point_map'，scene_tensor['color_images'].shape=torch.Size([32, 192, 256, 3]),scene_tensor['point_map'].shape=torch.Size([32, 224, 224, 3])
        #pdb.set_trace()


        ####保存图片###从这里开始_znn#########
        save_dir='/data2/znn/CODE/UniScene3D/Test_scene_tensor_znn/'
        from pathlib import Path
        from PIL import Image
        save_dir = Path('/data2/znn/CODE/UniScene3D/Test_scene_tensor_znn/')

        scan_save_dir = save_dir / scan_id
        color_save_dir = scan_save_dir / 'color_images'
        point_save_dir = scan_save_dir / 'point_map'

        color_save_dir.mkdir(parents=True, exist_ok=True)
        point_save_dir.mkdir(parents=True, exist_ok=True)

        def save_tensor_images(tensor, out_dir, normalize=False):
            tensor = tensor.detach().cpu()
            for i, img in enumerate(tensor):
                out_path = out_dir / f'{i}.png'
                if out_path.exists():
                    continue
                img = img.numpy()
                if normalize:
                    img_min = img.min()
                    img_max = img.max()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                    else:
                        img = np.zeros_like(img)

                if img.dtype != np.uint8:
                    if img.max() <= 1.0 and img.min() >= 0.0:
                        img = img * 255.0
                    img = img.clip(0, 255).astype(np.uint8)

                Image.fromarray(img).save(out_path)


        save_tensor_images(scene_tensor['color_images'], color_save_dir, normalize=False)
        save_tensor_images(scene_tensor['point_map'], point_save_dir, normalize=True)
        ####保存图片###到这里结束_znn#########

                


        out = {
            'point_map': scene_tensor['point_map'],
            'sentence': item['sentence'],
            'refer_sentence': item['refer_sentence'],
            'images': F.interpolate(
                scene_tensor['color_images'].permute(0, 3, 1, 2),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            ),
            'scan_id': scan_id,
        }

        if self.use_scene_cap:#self.use_scene_cap=True
            #pdb.set_trace()
            
            scene_caps = self.scene_caps.get(scan_id)#scan_id='48458202', scene_caps={''captions'':[]}
            if scene_caps is not None:
                #pdb.set_trace()
                captions = scene_caps['captions']#有几个平行的captions
                scene_cap = captions[np.random.choice(len(captions))]#随机选了一个captions
            else:
                scene_cap = 'This is a scene.'
            out['scene_cap'] = scene_cap

        return out
