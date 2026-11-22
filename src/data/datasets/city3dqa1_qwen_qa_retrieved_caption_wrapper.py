"""Qwen QA-only wrapper using the reference rule-based caption retrieval."""

import re
from pathlib import Path

import torch

from .city3dqa1_msg_caption_qa import load_scene_caption_candidates
from .city3dqa1_qwen_cached_feature_wrapper import (
    City3DQA1QwenCachedFeatureMixedWrapper,
)
from .dataset_wrapper import DATASETWRAPPER_REGISTRY


_ALIASES = {
    "car": ("car", "cars", "auto", "automobile", "motorcar", "vehicle"),
    "bridge": ("bridge", "overpass", "span", "way to connect two separate areas"),
    "ordinary building": ("ordinary building", "block", "construction"),
    "commercial building": (
        "commercial building", "commercial center", "commercial complex",
        "business center", "marketplace", "retail space", "providing retail space",
        "hospitality services", "accommodating businesses and organizations",
    ),
    "residential building": (
        "residential building", "residence", "residential complex", "dwelling",
        "habitation", "home", "living space", "providing a living space for individuals",
        "comfort and privacy to the residents",
    ),
    "office building": (
        "office building", "office complex", "office space", "workplace", "work zones",
        "individual work", "areas for individual work", "employees to network and socialize",
    ),
    "cultural center": (
        "cultural center", "cultural building", "cultural hub", "art gallery",
        "heritage and traditions", "preserving heritage", "learning and education",
    ),
    "transportation building": (
        "transportation building", "transport facility", "transport terminal", "transit hub",
        "station building", "transport", "travel", "tickets for travel",
        "connect two separate areas",
    ),
    "municipal building": (
        "municipal building", "government building", "city hall", "public administration building",
        "government regulations", "municipal regulations", "public hearings", "polling station",
        "local government assemblies",
    ),
    "temporary building": (
        "temporary building", "temporary structure", "temporary housing", "temporary lodging",
        "prefab structure", "short-term housing", "temporary residency",
        "temporary residency options", "arrange for temporary lodging",
    ),
    "boat": ("boat", "craft", "river vehicle", "vessel", "watercraft"),
    "terrain": ("terrain", "land", "ground"),
}
_DIRECTION_ALIASES = {
    "front-left": ("front-left", "front left", "frontleft"),
    "front-right": ("front-right", "front right", "frontright"),
    "back-left": ("back-left", "back left", "backleft", "rear-left", "rear left", "rearleft"),
    "back-right": ("back-right", "back right", "backright", "rear-right", "rear right", "rearright"),
    "front": ("front", "ahead", "forward"),
    "back": ("back", "rear", "behind"),
    "left": ("left",),
    "right": ("right",),
}
_INVERSE_DIRECTION = {
    "front": "back", "back": "front", "left": "right", "right": "left",
    "front-left": "back-right", "back-right": "front-left",
    "front-right": "back-left", "back-left": "front-right",
}
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "there", "any", "what", "which", "where",
    "in", "on", "at", "of", "to", "from", "and", "or", "do", "does", "can",
    "it", "for", "someone", "this", "scene", "area", "object", "located", "exist",
}


def _matches(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", text.lower()) is not None


def _entities(text):
    text = str(text).lower()
    found = set()
    for category, aliases in _ALIASES.items():
        if any(_matches(text, alias) for alias in aliases):
            found.add(category)
    specific = {
        "commercial building", "residential building", "office building", "cultural center",
        "transportation building", "municipal building", "temporary building",
    }
    if not found.intersection(specific) and _matches(text, "building"):
        found.add("ordinary building")
    return found


def _directions(text):
    text = str(text).lower().replace("-", " ")
    found, compound_found = set(), set()
    for direction, aliases in _DIRECTION_ALIASES.items():
        for alias in aliases:
            if _matches(text, alias.replace("-", " ")):
                (compound_found if "-" in direction else found).add(direction)
                break
    for compound, parts in {
        "front-left": {"front", "left"}, "front-right": {"front", "right"},
        "back-left": {"back", "left"}, "back-right": {"back", "right"},
    }.items():
        if compound in compound_found:
            found.difference_update(parts)
    return found | compound_found


def _family(text):
    text = str(text).lower()
    if re.search(r"\b(closer|nearest|closest|nearer|farther|further|quicker|distance|near|far)\b", text):
        return "distance"
    if _directions(text):
        return "direction"
    if re.search(r"\b(how many|number of|count|instances of)\b", text):
        return "count"
    if re.search(r"\b(is there|are there|does .* exist|do .* exist|any)\b", text):
        return "existence"
    return "general"


def _supports_distance(question, caption):
    question, caption = str(question).lower(), str(caption).lower()
    comparative = re.search(
        r"\b(closer|nearest|closest|nearer|farther|further|quicker|quickest)\b", question
    ) is not None
    if comparative:
        return re.search(
            r"\b(closer|nearest|closest|nearer|farther|further|distance between)\b", caption
        ) is not None
    return "distance" in caption or re.search(
        r"\b\d+(?:\.\d+)?\s*(?:xy\s*)?(?:units|away|meters|metres)\b", caption
    ) is not None


def _tokens(text):
    return {
        token for token in re.findall(r"[a-z]+", str(text).lower())
        if len(token) > 2 and token not in _STOP_WORDS
    }


def _directional_roles(text):
    """Return nearest canonical category mentions on either side of a direction."""
    text = str(text).lower()
    spans = []
    for category, aliases in _ALIASES.items():
        for alias in aliases:
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text):
                spans.append((match.start(), match.end(), category))
    spans.sort()
    direction_matches = []
    for direction, aliases in _DIRECTION_ALIASES.items():
        for alias in aliases:
            pattern = re.escape(alias).replace(r"\ ", r"[ -]")
            direction_matches.extend(
                (match.start(), match.end(), direction)
                for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", text)
            )
    if not direction_matches:
        return None
    position, _, direction = min(direction_matches, key=lambda row: row[0])
    before = [row for row in spans if row[1] <= position]
    after = [row for row in spans if row[0] >= position]
    if not before or not after:
        return None
    return before[-1][2], after[0][2], direction


def _question_roles(question):
    roles = _directional_roles(question)
    if roles is None:
        return None
    target, reference, direction = roles
    return target, reference, direction


def _supports_direction(question, caption):
    q_directions = _directions(question)
    c_roles, q_roles = _directional_roles(caption), _question_roles(question)
    if not q_directions:
        return False
    if q_roles and c_roles:
        q_target, q_reference, q_direction = q_roles
        c_target, c_reference, c_direction = c_roles
        if q_target == c_target and q_reference == c_reference:
            return q_direction == c_direction
        if q_target == c_reference and q_reference == c_target:
            return _INVERSE_DIRECTION.get(q_direction) == c_direction
        return False
    q_entities = _entities(question)
    if c_roles and q_entities:
        c_target, c_reference, c_direction = c_roles
        if c_reference in q_entities and c_direction in q_directions:
            return True
        if c_target in q_entities and _INVERSE_DIRECTION.get(c_direction) in q_directions:
            return True
    return bool(q_directions & _directions(caption))


def _score(question, caption):
    q_entities, c_entities = _entities(question), _entities(caption)
    q_family, c_family = _family(question), _family(caption)
    q_terms, c_terms = _tokens(question), _tokens(caption)
    score = 2.0 * len(q_terms & c_terms) / max(len(q_terms), 1)
    if q_entities:
        score += 3.0 * len(q_entities & c_entities) / len(q_entities)
        if q_entities <= c_entities:
            score += 1.0
    if q_family != "general":
        score += 3.0 if q_family == c_family else -2.0
    if q_family == "direction":
        q_directions, c_directions = _directions(question), _directions(caption)
        if _supports_direction(question, caption):
            score += 4.0
        elif q_directions and c_directions:
            score -= 2.0
        q_roles, c_roles = _question_roles(question), _directional_roles(caption)
        if q_roles and c_roles:
            q_target, q_reference, q_direction = q_roles
            c_target, c_reference, c_direction = c_roles
            if q_reference == c_reference and (q_target == c_target or q_target not in q_entities):
                score += 5.0 if q_direction == c_direction else -4.0
            elif q_reference == c_target and q_target == c_reference:
                score += 5.0 if _INVERSE_DIRECTION.get(q_direction) == c_direction else -4.0
    if q_family == "distance" and q_entities and q_entities <= c_entities:
        score += 3.0 if _supports_distance(question, caption) else -3.0
    return score


def retrieve_caption(question, candidates):
    """Pick the top rule-scored caption, or return empty when evidence is insufficient."""
    if not candidates:
        return ""
    ranked = sorted(
        ((_score(question, text), index, text) for index, text in enumerate(candidates)),
        key=lambda row: (-row[0], row[1]),
    )
    best_score, _, best_caption = ranked[0]
    family = _family(question)
    if family in {"direction", "distance"}:
        if _family(best_caption) != family:
            return ""
        if _entities(question) and not _entities(question) <= _entities(best_caption):
            return ""
        if family == "direction" and not _supports_direction(question, best_caption):
            return ""
        if family == "distance" and not _supports_distance(question, best_caption):
            return ""
    elif best_score <= 0:
        return ""
    return best_caption


def _encode_qa_with_caption(tokenizer, question, caption, answer,
                            question_max_length, caption_max_length, answer_max_length):
    question_ids = tokenizer.encode(str(question), add_special_tokens=False)
    caption_ids = tokenizer.encode(str(caption), add_special_tokens=False)
    question = tokenizer.decode(question_ids[:question_max_length], skip_special_tokens=True)
    caption = tokenizer.decode(caption_ids[:caption_max_length], skip_special_tokens=True)
    if caption:
        user_content = (
            "Use the scene caption as additional context when answering the question.\n"
            f"Scene caption: {caption}\nQuestion: {question}"
        )
    else:
        user_content = f"Answer the question about the 3D scene.\nQuestion: {question}"
    messages = [{"role": "user", "content": user_content}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    complete_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": str(answer)}],
        tokenize=True, add_generation_prompt=False, return_dict=False
    )
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Qwen chat-template prompt is not a prefix of the complete QA sequence.")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not complete_ids or complete_ids[-1] != eos_token_id):
        complete_ids.append(eos_token_id)
    complete_ids = complete_ids[:len(prompt_ids) + answer_max_length + 1]
    ids = torch.tensor(complete_ids, dtype=torch.long)
    decoder_input_ids = ids[:-1].contiguous()
    labels = ids[1:].clone()
    labels[:max(0, len(prompt_ids) - 1)] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    return decoder_input_ids, labels, torch.ones_like(decoder_input_ids), torch.tensor(prompt_ids, dtype=torch.long)


@DATASETWRAPPER_REGISTRY.register()
class City3DQA1QwenQARetrievedCaptionWrapper(City3DQA1QwenCachedFeatureMixedWrapper):
    """Use the same question-aware caption retrieval for train, val and test."""

    def __init__(self, cfg, dataset, split="train"):
        super().__init__(cfg, dataset, split=split)
        caption_cfg = cfg.data.City3DQA1CaptionUrbanCoCaDecoderPT
        self.render_base_dir = Path(caption_cfg.render_base_dir)
        self.caption_max_length = int(caption_cfg.get("qa_condition_max_length", 128))
        self.question_max_length = int(
            cfg.data.City3DQA1QAUrbanCoCaDecoderPT.get("max_question_length", 128)
        )
        self.caption_cache = {}

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if "sentence" not in out:
            raise ValueError("Retrieved-caption wrapper only accepts QA samples.")
        scene_id = str(out["scan_id"])
        if scene_id not in self.caption_cache:
            self.caption_cache[scene_id] = load_scene_caption_candidates(
                self.render_base_dir, scene_id
            )
        caption = retrieve_caption(out["sentence"], self.caption_cache[scene_id])
        decoder_ids, labels, decoder_mask, prompt_ids = _encode_qa_with_caption(
            self.tokenizer,
            out["sentence"],
            caption,
            out.get("target_text", ""),
            self.question_max_length,
            self.caption_max_length,
            self.qa_max_target_length,
        )
        out["scene_caption"] = caption
        out["scene_caption_valid"] = torch.tensor(bool(caption), dtype=torch.bool)
        out["retrieved_caption"] = caption
        out["decoder_input_ids"] = decoder_ids
        out["decoder_labels"] = labels
        out["decoder_attention_mask"] = decoder_mask
        out["generation_input_ids"] = prompt_ids
        out["generation_attention_mask"] = torch.ones_like(prompt_ids)
        return out
