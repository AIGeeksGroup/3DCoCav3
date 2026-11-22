"""City3DQA1 QA/caption datasets for MSG scene-caption training.

This module is intentionally separate from the original datasets.  It loads
the individual scene captions stored in the scene's ``sce_txt/*.txt`` files.
"""

import json
import itertools
import math
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..build import DATASET_REGISTRY


_PSEUDO_CAPTIONS_PER_SCENE = 5
_PSEUDO_VIEWS_PER_CAPTION = 4
_MAXIMIN_VIEW_SET_CACHE = {}
_VIEW_NAME_RE = re.compile(
    r"el_(?P<elevation>-?\d+(?:\.\d+)?)_view(?P<azimuth>-?\d+(?:\.\d+)?)"
)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scene_caption_candidates(render_base_dir, scene_id):
    """Read every text file under one scene's ``sce_txt`` directory.

    Only files whose suffix is ``.txt`` are read.  Some of these text files
    contain five captions written as a JSON-like string list; each list item
    is kept as a separate candidate.  The items are deliberately not joined.
    Files with a ``.json`` suffix are never scanned.
    """
    city, area = str(scene_id).split("_", 1)
    caption_dir = Path(render_base_dir) / city / area / "sce_txt"
    if not caption_dir.is_dir():
        return []

    candidates = []
    for path in sorted(caption_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            # A single .txt file may contain several independent captions.
            # Keep every caption as its own candidate instead of concatenating
            # the five sentences into one long scene description.
            for item in parsed:
                candidate = str(item).strip()
                if candidate:
                    candidates.append(candidate)
            continue
        elif isinstance(parsed, str) and parsed.strip():
            text = parsed.strip()
        if text:
            candidates.append(text)
    return candidates


def choose_scene_caption(render_base_dir, scene_id, training):
    candidates = load_scene_caption_candidates(render_base_dir, scene_id)
    if not candidates:
        return "", False
    return (random.choice(candidates) if training else candidates[0]), True


def _view_direction(file_name):
    """Return a unit viewing direction parsed from City3DQA view names."""
    match = _VIEW_NAME_RE.search(Path(str(file_name)).stem)
    if match is None:
        return None
    elevation = math.radians(float(match.group("elevation")))
    azimuth = math.radians(float(match.group("azimuth")))
    horizontal = math.cos(elevation)
    return (
        horizontal * math.cos(azimuth),
        horizontal * math.sin(azimuth),
        math.sin(elevation),
    )


def _maximin_view_sets(file_names, num_views=_PSEUDO_VIEWS_PER_CAPTION):
    """Find view sets whose minimum pairwise angular separation is maximal.

    The returned sets have a fixed maximin spacing.  Randomness is applied
    only when choosing among equally optimal sets, so every pseudo caption is
    different while the four-view coverage rule remains fixed.
    """
    names = tuple(str(name) for name in file_names)
    cache_key = (names, int(num_views))
    if cache_key in _MAXIMIN_VIEW_SET_CACHE:
        return _MAXIMIN_VIEW_SET_CACHE[cache_key]

    directions = [_view_direction(name) for name in names]
    valid = [index for index, direction in enumerate(directions) if direction is not None]
    if len(valid) < num_views:
        result = [tuple(valid)] if valid else []
        _MAXIMIN_VIEW_SET_CACHE[cache_key] = result
        return result

    distances = {}
    for left, right in itertools.combinations(valid, 2):
        distances[(left, right)] = math.sqrt(
            sum(
                (directions[left][axis] - directions[right][axis]) ** 2
                for axis in range(3)
            )
        )

    best_score = -1.0
    best_sets = []
    for candidate in itertools.combinations(valid, num_views):
        score = min(
            distances[tuple(sorted(pair))]
            for pair in itertools.combinations(candidate, 2)
        )
        if score > best_score + 1e-10:
            best_score = score
            best_sets = [candidate]
        elif abs(score - best_score) <= 1e-10:
            best_sets.append(candidate)

    _MAXIMIN_VIEW_SET_CACHE[cache_key] = best_sets
    return best_sets


def _unique_scene_ids(qa_dir, mode_name, split_names):
    scene_ids = []
    seen = set()
    for split_name in split_names:
        path = Path(qa_dir) / mode_name / f"{split_name}.json"
        for item in _read_json(path):
            scene_id = str(item["scene_id"])
            if scene_id not in seen:
                seen.add(scene_id)
                scene_ids.append(scene_id)
    return scene_ids


@DATASET_REGISTRY.register()
class City3DQA1MSGQAUrbanPT(Dataset):
    """QA rows; training uses the union of train.json and val.json."""

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        data_cfg = cfg.data.City3DQA1QAUrbanCoCaDecoderPT
        self.qa_dir = Path(data_cfg.qa_dir)
        self.mode_name = str(data_cfg.get("mode_name", "urban_mode"))
        self.cache_base_dir = Path(data_cfg.cache_base_dir)
        self.render_base_dir = Path(
            cfg.data.City3DQA1CaptionUrbanCoCaDecoderPT.render_base_dir
        )
        self.scene_cache_size = int(data_cfg.get("scene_cache_size", 4))
        self.image_size = int(data_cfg.get("image_size", 224))
        self.num_views = int(data_cfg.get("num_views", cfg.get("num_views", 50)))
        self.use_unanswer = bool(data_cfg.get(split, {}).get("use_unanswer", True))
        self.unanswerable_text = str(data_cfg.get("unanswerable_text", "unknown"))
        self._scene_cache = {}
        self._scene_cache_order = []
        self.lang_data = self._load_lang()

    def _load_lang(self):
        splits = ["train", "val"] if self.split == "train" else [self.split]
        rows = []
        seen = set()
        for split_name in splits:
            path = self.qa_dir / self.mode_name / f"{split_name}.json"
            for item in _read_json(path):
                key = str(item.get("question_id", f"{item['scene_id']}::{len(rows)}"))
                if key in seen:
                    continue
                seen.add(key)
                if item.get("answers") or self.use_unanswer:
                    rows.append(item)
        print(
            f"[City3DQA1MSGQAUrbanPT] split={self.split} source={splits} rows={len(rows)}"
        )
        return rows

    def _discover_cache(self, scene_id):
        city, area = scene_id.split("_", 1)
        return self.cache_base_dir / city / f"{area}.pt"

    def _load_scene_cache(self, scene_id):
        if scene_id in self._scene_cache:
            self._scene_cache_order.remove(scene_id)
            self._scene_cache_order.append(scene_id)
            return self._scene_cache[scene_id]
        path = self._discover_cache(scene_id)
        scene = torch.load(path, map_location="cpu")
        self._scene_cache[scene_id] = scene
        self._scene_cache_order.append(scene_id)
        while len(self._scene_cache_order) > self.scene_cache_size:
            old = self._scene_cache_order.pop(0)
            self._scene_cache.pop(old, None)
        return scene

    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        item = self.lang_data[index]
        scene_id = str(item["scene_id"])
        scene = self._load_scene_cache(scene_id)
        answers = list(item.get("answers", []))
        target = random.choice(answers) if self.split == "train" and answers else (
            answers[0] if answers else self.unanswerable_text
        )
        scene_caption, scene_caption_valid = choose_scene_caption(
            self.render_base_dir, scene_id, training=self.split == "train"
        )
        return {
            "task_name": "city3dqa1_msg_qa_urban_pt",
            "sentence": str(item["question"]),
            "scan_id": scene_id,
            "answers": "[answer_seq]".join(answers),
            "target_text": target,
            "point_map": scene["point_map"].contiguous().clone(),
            "images": scene["images"].contiguous().clone(),
            "view_ids": scene["view_ids"].contiguous().clone(),
            "file_names": list(scene["file_names"]),
            "data_idx": str(item.get("question_id", index)),
            "question_type": torch.tensor(
                0 if str(item.get("hop", "")).lower().startswith("single") else 1,
                dtype=torch.long,
            ),
            "scene_caption": scene_caption,
            "scene_caption_valid": torch.tensor(scene_caption_valid, dtype=torch.bool),
        }


@DATASET_REGISTRY.register()
class City3DQA1MSGCaptionUrbanPT(Dataset):
    """MSG caption samples plus on-demand multi-view pseudo captions."""

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        caption_cfg = cfg.data.City3DQA1CaptionUrbanCoCaDecoderPT
        self.qa_dir = Path(caption_cfg.qa_dir)
        self.mode_name = str(caption_cfg.get("mode_name", "urban_mode"))
        self.cache_base_dir = Path(caption_cfg.cache_base_dir)
        self.render_base_dir = Path(caption_cfg.render_base_dir)
        self.scene_cache_size = int(caption_cfg.get("scene_cache_size", 4))
        self.pseudo_captions_per_scene = int(
            caption_cfg.get("pseudo_captions_per_scene", _PSEUDO_CAPTIONS_PER_SCENE)
        )
        self.pseudo_views_per_caption = int(
            caption_cfg.get("pseudo_views_per_caption", _PSEUDO_VIEWS_PER_CAPTION)
        )
        self.msg_pseudo_ratio = float(caption_cfg.get("msg_pseudo_ratio", 3.0))
        self._scene_cache = {}
        self._scene_cache_order = []
        self._view_caption_cache = {}
        split_names = ["train", "val"] if split == "train" else [split]
        candidates = _unique_scene_ids(self.qa_dir, self.mode_name, split_names)
        self.scene_ids = [
            scene_id for scene_id in candidates if self._cache_path(scene_id).is_file()
        ]
        self.samples = []
        for scene_id in self.scene_ids:
            scene_candidates = load_scene_caption_candidates(
                self.render_base_dir, scene_id
            )
            if scene_candidates:
                if self.split == "train":
                    # Five pseudo candidates are generated on demand.  Three
                    # copies of the MSG candidate pool give the usual 3:1
                    # MSG:pseudo ratio when a scene has five MSG captions.
                    # For scenes with a different number of MSG captions,
                    # increase the integer repeat count to keep the ratio at
                    # least 3:1.
                    msg_repeats = max(
                        1,
                        math.ceil(
                            self.msg_pseudo_ratio * self.pseudo_captions_per_scene
                            / len(scene_candidates)
                        ),
                    )
                    for _ in range(msg_repeats):
                        self.samples.extend(
                            (scene_id, caption_index, caption, "msg")
                            for caption_index, caption in enumerate(scene_candidates)
                        )
                    self.samples.extend(
                        (scene_id, pseudo_index, None, "pseudo")
                        for pseudo_index in range(self.pseudo_captions_per_scene)
                    )
                else:
                    self.samples.extend(
                        (scene_id, caption_index, caption, "msg")
                        for caption_index, caption in enumerate(scene_candidates)
                    )
                continue
            self.samples.append((scene_id, None, None, "msg"))
        print(
            f"[City3DQA1MSGCaptionUrbanPT] split={split} source={split_names} "
            f"scenes={len(self.scene_ids)} samples={len(self.samples)}"
        )

    def _cache_path(self, scene_id):
        city, area = scene_id.split("_", 1)
        return self.cache_base_dir / city / f"{area}.pt"

    def _load_scene_cache(self, scene_id):
        if scene_id in self._scene_cache:
            self._scene_cache_order.remove(scene_id)
            self._scene_cache_order.append(scene_id)
            return self._scene_cache[scene_id]
        scene = torch.load(self._cache_path(scene_id), map_location="cpu")
        self._scene_cache[scene_id] = scene
        self._scene_cache_order.append(scene_id)
        while len(self._scene_cache_order) > self.scene_cache_size:
            old = self._scene_cache_order.pop(0)
            self._scene_cache.pop(old, None)
        return scene

    def _load_view_caption_records(self, scene_id, file_names):
        if scene_id in self._view_caption_cache:
            return self._view_caption_cache[scene_id]

        city, area = scene_id.split("_", 1)
        caption_dir = self.render_base_dir / city / area / "RGB_txt5"
        records = []
        for file_name in file_names:
            stem = Path(str(file_name)).stem
            text = ""
            for suffix in ("en", "zh"):
                path = caption_dir / f"{stem}.{suffix}.txt"
                if path.is_file():
                    text = path.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                    if text:
                        break
            if text:
                records.append((str(file_name), text))
        self._view_caption_cache[scene_id] = records
        return records

    def _build_pseudo_view_caption(self, scene_id, file_names):
        records = self._load_view_caption_records(scene_id, file_names)
        if len(records) < self.pseudo_views_per_caption:
            return ""
        view_sets = _maximin_view_sets(
            [file_name for file_name, _ in records],
            self.pseudo_views_per_caption,
        )
        if not view_sets:
            return ""
        selected = random.choice(view_sets)
        observations = [
            f"Observation from view {rank}: {records[index][1]}"
            for rank, index in enumerate(selected, start=1)
        ]
        return "\n".join(observations)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene_id, caption_index, caption_override, sample_kind = self.samples[index]
        scene = self._load_scene_cache(scene_id)
        if sample_kind == "pseudo":
            caption = self._build_pseudo_view_caption(
                scene_id, scene["file_names"]
            )
            valid = bool(caption)
            if not valid:
                # Keep the sample usable if a scene has fewer than four valid
                # RGB_txt5 English/Chinese files.
                caption, valid = choose_scene_caption(
                    self.render_base_dir, scene_id, training=True
                )
        elif caption_override is None:
            caption, valid = choose_scene_caption(
                self.render_base_dir, scene_id, training=self.split == "train"
            )
        else:
            caption, valid = caption_override, True
        data_idx = scene_id
        if caption_index is not None:
            suffix = "pseudo_view_caption" if sample_kind == "pseudo" else "scene_caption"
            data_idx = f"{scene_id}::{suffix}_{caption_index}"
        if not valid:
            caption = "An urban City3DQA scene."
        return {
            "task_name": "city3dqa1_msg_caption_urban_pt",
            "scan_id": scene_id,
            "caption_text": caption,
            "target_text": caption,
            "point_map": scene["point_map"].contiguous().clone(),
            "images": scene["images"].contiguous().clone(),
            "view_ids": scene["view_ids"].contiguous().clone(),
            "file_names": list(scene["file_names"]),
            "data_idx": data_idx,
            "scene_caption": caption,
            "scene_caption_valid": torch.tensor(valid, dtype=torch.bool),
        }
