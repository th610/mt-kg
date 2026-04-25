"""
pipeline/sage.py

SAGE (Scene-Aware Graph Encoding) 파이프라인:
  Step 1: 비디오 → 16프레임 → GPT-4o-mini → social graph JSON 추출
  Step 2: graph text + 16프레임 → GPT-4o-mini → SSR answer

plain_baseline.py / mtkg_baseline.py와 동일한 외부 인터페이스를 따른다.
"""

import json
import logging
import time
from typing import List, Optional, Sequence

from dataset import SIVBenchSample, iter_samples
from frame_sampler import sample_frames_uniform
from answer_parser import parse_answer
from evaluator import Evaluator
from models.gpt4o_mini import GPT4oMini

logger = logging.getLogger(__name__)

# ── 태스크 매핑 ───────────────────────────────────────────────────────────────
TASK_MAP = {
    "Intent Inference":   "SSR_intent",
    "Relation Inference": "SSR_relation",
    "Emotion Inference":  "SSR_emotion",
    "Attitude Inference": "SSR_attitude",
}

TASK_HINTS = {
    "SSR_intent":   "Focus on WHY the person acts this way. Use role and event context to infer motivation.",
    "SSR_relation": "Focus on the dominant social relationship. Use event type and interaction patterns.",
    "SSR_emotion":  "Focus on the emotional state. Use state_change and character emotion fields.",
    "SSR_attitude": "Focus on the attitude shown toward others. Use interaction type and intensity.",
}

# ── Step 1 프롬프트 ────────────────────────────────────────────────────────────
_GRAPH_EXTRACTION_PROMPT = """\
You are a social scene analyst.
Given video frames, extract the social structure as JSON.

Rules:
- Identify all visible characters (use A, B, C... as IDs)
- Assign ONE primary role per character: aggressor / target / mediator / observer / supporter / initiator / responder
- Identify ONE dominant event type for the entire scene
- Event stage: initiating / escalating / resolving / stable
- State change: describe emotional/behavioral shift from early to late frames (frames 1-8 vs 9-16)
- Interaction intensity: 0.0 to 1.0

Output ONLY valid JSON, no explanation:

{
  "characters": [
    {"id": "A", "role": "<role>", "emotion": "<emotion>"},
    ...
  ],
  "interactions": [
    {"from": "<id>", "to": "<id>", "type": "<conflict|support|ignore|mediate|confront>", "intensity": <0.0-1.0>},
    ...
  ],
  "event": {
    "type": "<event_type>",
    "participants": ["<id>", ...],
    "stage": "<stage>"
  },
  "state_change": {
    "early": "<description of frames 1-8>",
    "late": "<description of frames 9-16>"
  }
}"""


def _build_step1_messages(frames_b64: List[str], subtitle: Optional[str]) -> List[dict]:
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        }
        for b64 in frames_b64
    ]
    prompt = _GRAPH_EXTRACTION_PROMPT
    if subtitle:
        prompt += f"\n\nSubtitle:\n{subtitle}"
    user_content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": user_content}]


def extract_social_graph(
    frames_b64: List[str],
    model: GPT4oMini,
    subtitle: Optional[str] = None,
) -> Optional[dict]:
    """Step 1: 16프레임 → social graph JSON."""
    messages = _build_step1_messages(frames_b64, subtitle)
    raw = model.chat(messages, max_tokens=800, temperature=0.0)

    try:
        text = raw
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"graph JSON 파싱 실패: {repr(raw[:120])}")
        return None


def graph_to_text(graph: Optional[dict]) -> str:
    """추출된 social graph를 LLM 입력용 텍스트로 변환."""
    if graph is None:
        return "[Social Context: extraction failed]"

    lines = ["[Social Context]"]

    event = graph.get("event", {})
    lines.append(f"- Event: {event.get('type', 'unknown')} ({event.get('stage', 'unknown')})")
    participants = ", ".join(event.get("participants", []))
    lines.append(f"  Participants: {participants}")

    for char in graph.get("characters", []):
        lines.append(f"- {char['id']}: role={char.get('role','?')}, emotion={char.get('emotion','?')}")

    for intr in graph.get("interactions", []):
        lines.append(
            f"- {intr['from']}→{intr['to']}: "
            f"{intr.get('type','?')} (intensity={intr.get('intensity','?')})"
        )

    state = graph.get("state_change", {})
    if state:
        lines.append("- State change:")
        lines.append(f"  Early: {state.get('early', '')}")
        lines.append(f"  Late:  {state.get('late', '')}")

    return "\n".join(lines)


def _build_step2_messages(
    frames_b64: List[str],
    graph_text: str,
    question: str,
    options: List[str],
    task_type: str,
) -> List[dict]:
    hint = TASK_HINTS.get(task_type, "Focus on social context to answer.")
    options_text = "\n".join(options)
    prompt = (
        f"You are analyzing a social interaction video.\n\n"
        f"{graph_text}\n\n"
        f"{hint}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer with ONLY the option letter (A, B, C, D, or E):"
    )
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        }
        for b64 in frames_b64
    ]
    user_content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": user_content}]


def _run_single(
    sample: SIVBenchSample,
    video_path: str,
    num_frames: int,
    model: GPT4oMini,
) -> dict:
    """단일 샘플: 프레임 추출 → Step1 → Step2 → 파싱."""
    frames_b64 = sample_frames_uniform(video_path, num_frames=num_frames)
    task_type = TASK_MAP.get(sample.category, "SSR_intent")

    # Step 1: social graph 추출
    graph = extract_social_graph(frames_b64, model, subtitle=None)
    graph_text = graph_to_text(graph)

    # Step 2: QA
    messages = _build_step2_messages(
        frames_b64, graph_text, sample.question, sample.options, task_type
    )
    raw_output = model.chat(messages, max_tokens=10, temperature=0.0)
    predicted = parse_answer(raw_output)

    return {
        "raw_output": raw_output,
        "predicted": predicted,
        "graph": graph,
        "graph_text": graph_text,
    }


def run_sage_pipeline(
    subtitle_condition: str = "origin",
    num_frames: int = 16,
    max_samples: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    local_video_dir: Optional[str] = None,
    result_save_path: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Evaluator:
    """SAGE 파이프라인을 실행하고 Evaluator 객체를 반환한다.

    Args:
        subtitle_condition: 'origin' | 'w_sub' | 'wo_sub'  (스펙: origin 단일 조건)
        num_frames: 균일 샘플링 프레임 수 (스펙: 16)
        max_samples: 실행할 최대 QA 수 (None = 전체)
        categories: 평가할 category 이름 목록 (None = 전체)
        cache_dir: HF Hub 캐시 디렉토리
        local_video_dir: 미리 다운로드된 비디오 루트 디렉토리
        result_save_path: 예측 결과 JSON 저장 경로 (None = 저장 안 함)
        api_key: OpenAI API key
        base_url: OpenAI base URL
    """
    model = GPT4oMini(api_key=api_key, base_url=base_url)
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
            graph_ok = "ok" if result["graph"] is not None else "fail"
            raw_preview = repr(result["raw_output"][:60]) if result["raw_output"] else "None"
            logger.info(
                f"[{i+1}] {sample.question_id} | pred={result['predicted']} "
                f"gt={sample.correct_answer_index} | {'O' if is_correct else 'X'} "
                f"| acc={acc*100:.1f}% | {elapsed:.1f}s | graph={graph_ok} | raw={raw_preview}"
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