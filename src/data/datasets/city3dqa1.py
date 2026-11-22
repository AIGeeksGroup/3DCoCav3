"""City3DQA1 pretraining dataset built from rendered City-3DQA views."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class City3DQA1SpatialRefer(Dataset):
    """Load rendered City-3DQA views for UniScene3D pretraining."""

    def __init__(self, cfg, split):
        if split != "pretrain":
            raise ValueError("City3DQA1SpatialRefer is only intended for pretrain split.")

        self.cfg = cfg
        self.split = split
        data_cfg = cfg.data.City3DQA1SpatialRefer
        self.base_dir = Path(data_cfg.base_dir)
        self.cities = list(data_cfg.get("cities", ["Lihu", "Qingdao", "Longhua"]))
        self.image_size = int(data_cfg.get("image_size", 224))
        self.language = str(data_cfg.get("language", "en"))
        self.scene_cap_max_views = int(data_cfg.get("scene_cap_max_views", 4))
        self.text_dir_name = str(data_cfg.get("text_dir_name", "RGB_txt3"))
        self.allowed_scene_ids = self._load_qa_split_scene_filter(data_cfg)
        self.samples = self._discover_samples()

    def _load_qa_split_scene_filter(self, data_cfg):
        filter_cfg = data_cfg.get("qa_split_filter", None)
        if not filter_cfg or not filter_cfg.get("enabled", False):
            return None

        qa_path = (
            Path(filter_cfg.qa_dir)
            / str(filter_cfg.get("mode_name", "urban_mode"))
            / f"{filter_cfg.get('split', 'train')}.json"
        )
        with qa_path.open("r", encoding="utf-8") as f:
            qa_data = json.load(f)

        scene_ids = {item["scene_id"] for item in qa_data}
        print(
            f"[City3DQA1] Loaded {len(scene_ids)} QA split scenes from "
            f"{qa_path}"
        )
        return scene_ids

    def _discover_samples(self):
        samples = []
        for city in self.cities:
            city_dir = self.base_dir / city
            if not city_dir.exists():
                print(f"[City3DQA1] Skip missing city directory: {city_dir}")
                continue

            for area_dir in sorted(p for p in city_dir.iterdir() if p.is_dir()):
                scene_id = f"{city}_{area_dir.name}"
                if self.allowed_scene_ids is not None and scene_id not in self.allowed_scene_ids:
                    continue

                cameras_file = area_dir / "cameras.json"
                point_meta_file = area_dir / "pointmap_meta.json"
                rgb_dir = area_dir / "RGB"
                pointmap_dir = area_dir / "Pointmap"
                txt_dir = area_dir / self.text_dir_name
                valid_dir = area_dir / "ValidMask"

                required = [cameras_file, point_meta_file, rgb_dir, pointmap_dir, txt_dir, valid_dir]
                if not all(path.exists() for path in required):
                    print(f"[City3DQA1] Skip incomplete area: {area_dir}")
                    continue

                with open(cameras_file, "r", encoding="utf-8") as f:
                    cameras = json.load(f)
                with open(point_meta_file, "r", encoding="utf-8") as f:
                    point_meta = json.load(f)

                views = []
                for cam in sorted(cameras, key=lambda item: int(item["view_id"])):
                    file_name = cam["file_name"]
                    rgb_path = rgb_dir / file_name
                    pm_path = pointmap_dir / file_name
                    valid_path = valid_dir / file_name
                    if not (rgb_path.exists() and pm_path.exists() and valid_path.exists()):
                        continue

                    captions = self._load_view_captions(txt_dir, file_name)
                    if not captions:
                        captions = [f"A rendered urban scene view from {city} {area_dir.name}."]

                    views.append(
                        {
                            "view_id": int(cam["view_id"]),
                            "file_name": file_name,
                            "rgb_path": rgb_path,
                            "pointmap_path": pm_path,
                            "valid_path": valid_path,
                            "captions": captions,
                        }
                    )

                if views:
                    samples.append(
                        {
                            "scan_id": f"City3DQA1_{scene_id}",
                            "city": city,
                            "area": area_dir.name,
                            "views": views,
                            "point_meta": point_meta,
                            "scene_cap": self._build_scene_caption(views, city, area_dir.name),
                        }
                    )

        print(
            f"[City3DQA1] Loaded {len(samples)} areas from "
            f"{self.base_dir} using cities: {', '.join(self.cities)}"
        )
        return samples

    def _load_view_captions(self, txt_dir, file_name):
        stem = Path(file_name).stem
        candidates = []
        suffixes = [self.language]
        if self.language != "en":
            suffixes.append("en")
        if self.language != "zh":
            suffixes.append("zh")

        for suffix in suffixes:
            txt_path = txt_dir / f"{stem}.{suffix}.txt"
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()
                if text:
                    candidates.append(text)

        return candidates

    def _build_scene_caption(self, views, city, area):
        captions = []
        for view in views[: self.scene_cap_max_views]:
            if view["captions"]:
                captions.append(view["captions"][0])
        if captions:
            return " ".join(captions)
        return f"An urban 3D scene from City3DQA1 {city} {area}."

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_pointmap(self, pointmap_path, valid_path, point_meta):
        image = Image.open(pointmap_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        encoded = np.asarray(image, dtype=np.float32) / 255.0

        xyz_min = np.asarray(point_meta["xyz_min"], dtype=np.float32)
        xyz_max = np.asarray(point_meta["xyz_max"], dtype=np.float32)
        pointmap = encoded * (xyz_max - xyz_min) + xyz_min

        valid = Image.open(valid_path).convert("L")
        valid = valid.resize((self.image_size, self.image_size), Image.NEAREST)
        valid_mask = np.asarray(valid) > 0
        pointmap[~valid_mask] = 0.0
        return torch.from_numpy(pointmap).contiguous()

    def __getitem__(self, index):
        item = self.samples[index]
        rgb_views = []
        pointmap_views = []
        sentences = []
        refer_sentences = []

        for view in item["views"]:
            rgb_views.append(self._load_rgb(view["rgb_path"]))
            pointmap_views.append(
                self._load_pointmap(view["pointmap_path"], view["valid_path"], item["point_meta"])
            )
            sentences.append(view["captions"])
            refer_sentences.append(view["captions"])

        images = torch.stack(rgb_views, dim=0)
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images.float(), size=(224, 224), mode="bilinear", align_corners=False).to(torch.uint8)

        return {
            "point_map": torch.stack(pointmap_views, dim=0),
            "images": images,
            "sentence": sentences,
            "refer_sentence": refer_sentences,
            "scene_cap": item["scene_cap"],
            "scan_id": item["scan_id"],
        }

@DATASET_REGISTRY.register()
class City3DQA1SpatialReferPT(City3DQA1SpatialRefer):
    """Load cached City-3DQA PT views while keeping captions from rendered folders."""

    def __init__(self, cfg, split):
        if split != "pretrain":
            raise ValueError("City3DQA1SpatialReferPT is only intended for pretrain split.")

        self.cfg = cfg
        self.split = split
        data_cfg = cfg.data.City3DQA1SpatialReferPT
        self.base_dir = Path(data_cfg.base_dir)
        self.text_base_dir = Path(data_cfg.text_base_dir)
        self.cities = list(data_cfg.get("cities", ["Lihu", "Qingdao", "Longhua"]))
        self.image_size = int(data_cfg.get("image_size", 224))
        self.language = str(data_cfg.get("language", "en"))
        self.scene_cap_max_views = int(data_cfg.get("scene_cap_max_views", 4))
        self.text_dir_name = str(data_cfg.get("text_dir_name", "RGB_txt4"))
        self.allowed_scene_ids = self._load_qa_split_scene_filter(data_cfg)
        self.samples = self._discover_samples()

    def _discover_samples(self):
        samples = []
        for city in self.cities:
            cache_city_dir = self.base_dir / city
            text_city_dir = self.text_base_dir / city
            if not cache_city_dir.exists():
                print(f"[City3DQA1-PT] Skip missing cache city directory: {cache_city_dir}")
                continue
            if not text_city_dir.exists():
                print(f"[City3DQA1-PT] Skip missing text city directory: {text_city_dir}")
                continue

            for cache_path in sorted(cache_city_dir.glob("*.pt")):
                area = cache_path.stem
                scene_id = f"{city}_{area}"
                if self.allowed_scene_ids is not None and scene_id not in self.allowed_scene_ids:
                    continue

                txt_dir = text_city_dir / area / self.text_dir_name
                if not txt_dir.exists():
                    print(f"[City3DQA1-PT] Skip missing text directory: {txt_dir}")
                    continue

                samples.append(
                    {
                        "scan_id": f"City3DQA1_{scene_id}",
                        "scene_id": scene_id,
                        "city": city,
                        "area": area,
                        "cache_path": cache_path,
                        "txt_dir": txt_dir,
                    }
                )

        print(
            f"[City3DQA1-PT] Loaded {len(samples)} cached areas from "
            f"{self.base_dir} using text root {self.text_base_dir}"
        )
        return samples

    def _load_cache(self, cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        images = cache["images"]
        point_maps = cache["point_map"]

        if images.ndim != 4:
            raise ValueError(f"Expected cached images with 4 dims in {cache_path}, got {images.shape}.")
        if point_maps.ndim != 4:
            raise ValueError(f"Expected cached point_map with 4 dims in {cache_path}, got {point_maps.shape}.")

        if images.shape[-1] == 3 and images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        if point_maps.shape[1] == 3:
            point_maps = point_maps.permute(0, 2, 3, 1).contiguous()
        elif point_maps.shape[-1] != 3:
            raise ValueError(f"Expected cached point_map channels in dim 1 or -1 in {cache_path}, got {point_maps.shape}.")

        return images.contiguous(), point_maps.contiguous(), list(cache.get("file_names", []))

    def __getitem__(self, index):
        item = self.samples[index]
        images, point_maps, file_names = self._load_cache(item["cache_path"])
        if len(file_names) != images.shape[0]:
            file_names = [f"view_{view_idx}.png" for view_idx in range(images.shape[0])]

        sentences = []
        refer_sentences = []
        views = []
        for file_name in file_names:
            captions = self._load_view_captions(item["txt_dir"], file_name)
            if not captions:
                captions = [f"A rendered urban scene view from {item['city']} {item['area']}."]
            sentences.append(captions)
            refer_sentences.append(captions)
            views.append({"captions": captions})

        return {
            "point_map": point_maps,
            "images": images,
            "sentence": sentences,
            "refer_sentence": refer_sentences,
            "scene_cap": self._build_scene_caption(views, item["city"], item["area"]),
            "scan_id": item["scan_id"],
        }
