"""
evaluator.py

정답 여부를 누적하고 accuracy를 계산한다.
- 전체 accuracy
- category별 accuracy (SSU / SSR / SDP 및 세부 category)
- 논문 Table과 직접 비교할 수 있는 형식으로 출력
"""

import json
import logging
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# SIV-Bench TSV 실제 카테고리명 기준 매핑 (8728 샘플 전수 확인)
CATEGORY_TO_TASK = {
    # SSU
    "Environment Perception":        "SSU",
    "Action Recognition":            "SSU",
    "Human Attribute Identification": "SSU",
    "Facial Expression Recognition": "SSU",
    # SSR
    "Relation Inference":            "SSR",
    "Emotion Inference":             "SSR",
    "Intent Inference":              "SSR",
    "Attitude Inference":            "SSR",
    # SDP
    "Counterfactual Prediction":     "SDP",
    "Factual Prediction":            "SDP",
}


class Evaluator:
    def __init__(self):
        # category → {correct, total}
        self._cat: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        self._overall = {"correct": 0, "total": 0}
        # 전체 결과 로그 (나중에 분석용)
        self._records = []

    def add(
        self,
        question_id: str,
        predicted: Optional[str],
        ground_truth: str,
        category: str,
        raw_output: Optional[str] = None,
    ) -> bool:
        """결과 1개를 추가한다.

        Args:
            question_id: e.g. '0001-1'
            predicted: 파싱된 예측 답 ('A'~'N' 또는 None)
            ground_truth: 정답 알파벳
            category: e.g. 'Relation Inference'

        Returns:
            정답 여부 (True/False)
        """
        is_correct = (predicted is not None) and (predicted == ground_truth)

        self._overall["total"] += 1
        self._overall["correct"] += int(is_correct)

        self._cat[category]["total"] += 1
        self._cat[category]["correct"] += int(is_correct)

        self._records.append({
            "question_id": question_id,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "category": category,
            "correct": is_correct,
            "raw_output": raw_output,
        })

        return is_correct

    def get_accuracy(self) -> float:
        """전체 accuracy를 반환한다."""
        if self._overall["total"] == 0:
            return 0.0
        return self._overall["correct"] / self._overall["total"]

    def get_category_accuracy(self) -> Dict[str, float]:
        """category별 accuracy dict를 반환한다."""
        return {
            cat: (v["correct"] / v["total"] if v["total"] > 0 else 0.0)
            for cat, v in self._cat.items()
        }

    def get_task_accuracy(self) -> Dict[str, float]:
        """SSU / SSR / SDP 상위 task별 accuracy를 반환한다."""
        task_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        for cat, v in self._cat.items():
            task = CATEGORY_TO_TASK.get(cat, "Unknown")
            task_counts[task]["correct"] += v["correct"]
            task_counts[task]["total"] += v["total"]
        return {
            task: (v["correct"] / v["total"] if v["total"] > 0 else 0.0)
            for task, v in task_counts.items()
        }

    def print_summary(self):
        """콘솔에 결과 요약을 출력한다."""
        overall = self.get_accuracy()
        task_acc = self.get_task_accuracy()
        cat_acc = self.get_category_accuracy()

        print("\n" + "=" * 60)
        print(f"{'Overall':<30} {overall*100:.1f}%  ({self._overall['correct']}/{self._overall['total']})")
        print("-" * 60)
        for task in ["SSU", "SSR", "SDP"]:
            acc = task_acc.get(task, 0.0)
            print(f"  {task:<28} {acc*100:.1f}%")
        print("-" * 60)
        for cat, acc in sorted(cat_acc.items()):
            task = CATEGORY_TO_TASK.get(cat, "?")
            n = self._cat[cat]
            print(f"    [{task}] {cat:<26} {acc*100:.1f}%  ({n['correct']}/{n['total']})")
        print("=" * 60)

    def save_records(self, path: str):
        """전체 예측 결과를 JSON으로 저장한다."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)
        logger.info(f"결과 저장: {path}")
