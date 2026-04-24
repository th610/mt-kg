"""
pipeline/plain_baseline.py

plain baseline 파이프라인:
  비디오 → 16프레임 → o4-mini → 정답 파싱 → evaluator 누적

이 파일이 전체 흐름의 중심이며, mtkg_adapter.py도 같은 인터페이스를 따른다.
"""

import logging
import time
from typing import Optional, Sequence

from dataset import SIVBenchSample, iter_samples
from frame_sampler import sample_frames_uniform
from prompt_builder import build_plain_prompt
from answer_parser import parse_answer
from evaluator import Evaluator
from models.o4_mini import O4Mini

logger = logging.getLogger(__name__)


def run_plain_baseline(
    subtitle_condition: str = "w_sub",
    num_frames: int = 16,
    max_samples: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    local_video_dir: Optional[str] = None,
    result_save_path: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Evaluator:
    """plain baseline을 실행하고 Evaluator 객체를 반환한다.

    Args:
        subtitle_condition: 'origin' | 'w_sub' | 'wo_sub'
        num_frames: 균일 샘플링 프레임 수
        max_samples: 실행할 최대 QA 수 (None = 전체)
        categories: 평가할 category 이름 목록 (None = 전체)
        cache_dir: HF Hub 캐시 디렉토리
        local_video_dir: 미리 다운로드된 비디오 루트 디렉토리
        result_save_path: 예측 결과 JSON 저장 경로 (None = 저장 안 함)
        api_key: OpenAI API key (None이면 환경변수 사용)
        base_url: OpenAI base URL (None이면 환경변수 사용)

    Returns:
        누적된 Evaluator 객체
    """
    model = O4Mini(api_key=api_key, base_url=base_url)
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
            result = _run_single(sample, video_path, num_frames, model)
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
            logger.info(
                f"[{i+1}] {sample.question_id} | pred={result['predicted']} "
                f"gt={sample.correct_answer_index} | {'O' if is_correct else 'X'} "
                f"| acc={acc*100:.1f}% | {elapsed:.1f}s | raw={raw_preview}"
            )
        except Exception as e:
            logger.error(f"[{i+1}] {sample.question_id} 처리 실패: {e}")
            # 실패한 경우 None 예측으로 기록
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
    model: O4Mini,
) -> dict:
    """단일 샘플에 대해 프레임 추출 → 프롬프트 빌드 → API 호출 → 파싱."""
    frames_b64 = sample_frames_uniform(video_path, num_frames=num_frames)

    # TODO: w_sub 조건에서 자막 텍스트를 별도로 추출하려면
    #       subtitle 파일을 파싱해서 여기에 전달해야 함
    #       현재는 자막이 영상에 burn-in 되어 있으므로 frames로 충분
    subtitle_text = None

    messages = build_plain_prompt(
        question=sample.question,
        options=sample.options,
        frames_b64=frames_b64,
        subtitle=subtitle_text,
    )

    raw_output = model.chat(messages)
    predicted = parse_answer(raw_output)

    return {
        "raw_output": raw_output,
        "predicted": predicted,
    }
