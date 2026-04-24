#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from configs.moviegraphs import data_cfg
from data.moviegraphs_data import MovieGraphsDataset


def main() -> None:
	with tempfile.TemporaryDirectory() as tmpdir:
		temp_path = Path(tmpdir) / "pre_loaded.json"
		payload = [
			{
				"video_name": "demo_scene",
				"background_summaries": ["people at office"],
				"interaction_summaries": ["discussion near desk"],
				"individual_summaries": [["manager", "intern"]],
				"queries": [["a", "b"]],
				"labels": ["Leader-Sub"],
				"end": True,
				"background_images": [""],
				"face_images": [["", ""]],
			}
		]
		temp_path.write_text(json.dumps(payload), encoding="utf-8")

		data_cfg.DATA.PRE_LOADED_PATH = str(temp_path)
		dataset = MovieGraphsDataset(train=True, llm=None)

		assert len(dataset) == 1, "Dataset length mismatch"
		sample = dataset[0]
		assert len(sample) == 9, "Tuple contract mismatch"
		assert dataset.get_relation_dict(), "Relation dict should not be empty"

	print("[PASS] smoke_test: loader import + tuple contract validated")


if __name__ == "__main__":
	main()


