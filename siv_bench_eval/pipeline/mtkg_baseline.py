"""
pipeline/mtkg_baseline.py

mtKG baseline 파이프라인:
  비디오 → 16프레임 → [mtKG: 그래프 구축 → 커뮤니티 요약] → o4-mini → 정답 파싱 → evaluator 누적

plain_baseline.py와 동일한 인터페이스를 따른다.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import SIVBenchSample, iter_samples
from frame_sampler import sample_frames_uniform
from answer_parser import parse_answer
from evaluator import Evaluator
from models.o4_mini import O4Mini
from pipeline.mtkg_adapter import (
    O4MiniMtKGBridge,
    build_mtkg_context,
    build_mtkg_qa_prompt,
    DEFAULT_NUM_SEGMENTS,
)

logger = logging.getLogger(__name__)


def run_mtkg_baseline(
    subtitle_condition: str = "w_sub",
    num_frames: int = 16,
    num_segments: int = DEFAULT_NUM_SEGMENTS,
    max_samples: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    local_video_dir: Optional[str] = None,
    result_save_path: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Evaluator:
    """mtKG baseline을 실행하고 Evaluator 객체를 반환한다.

    Args:
        subtitle_condition: 'origin' | 'w_sub' | 'wo_sub'
        num_frames: 균일 샘플링 프레임 수
        num_segments: mtKG 시간 구간 수 (기본 4)
        max_samples: 실행할 최대 QA 수 (None = 전체)
        categories: 평가할 category 이름 목록 (None = 전체)
        cache_dir: HF Hub 캐시 디렉토리
        local_video_dir: 미리 다운로드된 비디오 루트 디렉토리
        result_save_path: 예측 결과 JSON 저장 경로 (None = 저장 안 함)
        api_key: OpenAI API key
        base_url: OpenAI base URL
    """
    model = O4Mini(api_key=api_key, base_url=base_url)
    llm_bridge = O4MiniMtKGBridge(model)
    evaluator = Evaluator()

    start_time = time.time()

    for i, (sample, video_path) in enumerate(
        iter_samples(
            subtitle_condition=subtitle_condition,
            cache_dir=cache_dir,
            local_video_dir=local_video_dir,
            max_samples=max_samples,
            categories=categories,
        )
    ):
        try:
            result = _run_single(sample, video_path, num_frames, num_segments, model, llm_bridge)
            is_correct = evaluator.add(
                question_id=sample.question_id,
                predicted=result["predicted"],
                ground_truth=sample.correct_answer_index,
                category=sample.category,
                raw_output=result["raw_output"],
            )
            elapsed = time.time() - start_time
            acc = evaluator.get_accuracy()
            raw_preview = repr(result["raw_output"][:60]) if result["raw_output"] else "None"
            ctx_len = len(result.get("context", ""))
            logger.info(
                f"[{i+1}] {sample.question_id} | pred={result['predicted']} "
                f"gt={sample.correct_answer_index} | {'O' if is_correct else 'X'} "
                f"| acc={acc*100:.1f}% | {elapsed:.1f}s | ctx={ctx_len}chars | raw={raw_preview}"
            )
        except Exception as e:
            logger.error(f"[{i+1}] {sample.question_id} 처리 실패: {e}")
            evaluator.add(
                question_id=sample.question_id,
                predicted=None,
                ground_truth=sample.correct_answer_index,
                category=sample.category,
                raw_output=str(e),
            )

    evaluator.print_summary()

    if result_save_path:
        evaluator.save_records(result_save_path)

    return evaluator


def _run_single(
    sample: SIVBenchSample,
    video_path: str,
    num_frames: int,
    num_segments: int,
    model: O4Mini,
    llm_bridge: O4MiniMtKGBridge,
) -> dict:
    """단일 샘플에 대해 프레임 추출 → mtKG 그래프 구축 → QA → 파싱."""
    frames_b64 = sample_frames_uniform(video_path, num_frames=num_frames)

    # mtKG 파이프라인: 그래프 구축 → 커뮤니티 요약 context
    context = build_mtkg_context(frames_b64, llm_bridge, num_segments=num_segments)

    # context + 질문 + 선택지로 QA 수행
    messages = build_mtkg_qa_prompt(
        context=context,
        question=sample.question,
        options=sample.options,
    )

    raw_output = model.chat(messages)
    predicted = parse_answer(raw_output)

    return {"raw_output": raw_output, "predicted": predicted, "context": context}
