import json
import os
from typing import Any, Dict, List, Tuple

from configs.hlvu import data_cfg


class HLVUDataset:
	"""Placeholder HLVU loader with the same tuple contract as MovieGraphsDataset."""

	def __init__(self, llm: Any = None, preprocessed_path: str = ""):
		self.llm = llm
		self.preprocessed_path = preprocessed_path or data_cfg.DATA.PRE_PROCESSED_PATH
		self.samples: List[Dict[str, Any]] = []
		self.relation_dict: Dict[str, int] = {}
		self._load_samples()

	def _load_samples(self) -> None:
		if not self.preprocessed_path or not os.path.exists(self.preprocessed_path):
			self.relation_dict = {}
			return
		with open(self.preprocessed_path, "r", encoding="utf-8") as f:
			raw = json.load(f)
		if not isinstance(raw, list):
			raise ValueError("PRE_PROCESSED_PATH must contain a JSON list.")
		self.samples = raw

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> Tuple[Any, ...]:
		item = self.samples[index]
		return (
			item.get("background_summaries", []),
			item.get("interaction_summaries", []),
			item.get("individual_summaries", []),
			item.get("queries", []),
			item.get("labels", []),
			bool(item.get("end", False)),
			item.get("video_name", "unknown_scene"),
			item.get("background_images", []),
			item.get("face_images", []),
		)

	def get_relation_dict(self) -> Dict[str, int]:
		return self.relation_dict

