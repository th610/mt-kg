import json
import os
from typing import Any, Dict, List, Tuple

from configs.moviegraphs import data_cfg

DEFAULT_RELATIONS = [
	"Leader-Sub",
	"Colleague",
	"Service",
	"Parent-offs",
	"Sibling",
	"Couple",
	"Friend",
	"Opponent",
]


class MovieGraphsDataset:
	"""Minimal loader skeleton compatible with main.py."""

	def __init__(self, train: bool = True, llm: Any = None, preloaded_path: str = ""):
		self.train = train
		self.llm = llm
		self.preloaded_path = preloaded_path or data_cfg.DATA.PRE_LOADED_PATH
		self.samples: List[Dict[str, Any]] = []
		self.relation_dict: Dict[str, int] = {}
		self._load_samples()

	def _load_samples(self) -> None:
		if not self.preloaded_path or not os.path.exists(self.preloaded_path):
			self._set_default_relations()
			return

		with open(self.preloaded_path, "r", encoding="utf-8") as f:
			raw = json.load(f)

		if not isinstance(raw, list):
			raise ValueError("PRE_LOADED_PATH must contain a JSON list.")

		self.samples = [self._normalize_sample(item) for item in raw]
		self._build_relation_dict_from_labels()

	def _normalize_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
		sample = {
			"video_name": item.get("video_name", "unknown_scene"),
			"background_summaries": item.get("background_summaries", []),
			"interaction_summaries": item.get("interaction_summaries", []),
			"individual_summaries": item.get("individual_summaries", []),
			"queries": item.get("queries", []),
			"labels": item.get("labels", []),
			"end": bool(item.get("end", False)),
			"background_images": item.get("background_images", []),
			"face_images": item.get("face_images", []),
		}

		q_len = len(sample["queries"])
		for key, default_value in [
			("background_summaries", ""),
			("interaction_summaries", ""),
			("individual_summaries", ["", ""]),
			("labels", "unknown"),
			("background_images", ""),
			("face_images", ["", ""]),
		]:
			if len(sample[key]) < q_len:
				sample[key].extend([default_value for _ in range(q_len - len(sample[key]))])

		return sample

	def _set_default_relations(self) -> None:
		self.relation_dict = {name: idx for idx, name in enumerate(DEFAULT_RELATIONS)}

	def _build_relation_dict_from_labels(self) -> None:
		labels = []
		for sample in self.samples:
			for label in sample["labels"]:
				if label not in labels:
					labels.append(label)
		if not labels:
			self._set_default_relations()
			return
		self.relation_dict = {name: idx for idx, name in enumerate(labels)}

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> Tuple[Any, ...]:
		sample = self.samples[index]
		return (
			sample["background_summaries"],
			sample["interaction_summaries"],
			sample["individual_summaries"],
			sample["queries"],
			sample["labels"],
			sample["end"],
			sample["video_name"],
			sample["background_images"],
			sample["face_images"],
		)

	def get_relation_dict(self) -> Dict[str, int]:
		return self.relation_dict

