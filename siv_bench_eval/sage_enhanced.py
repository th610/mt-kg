"""
sage_enhanced.py

SAGE Enhanced: SocialGPT 프롬프트 패턴을 분석해 보강된 SAGE 파이프라인.

실행:
    python sage_enhanced.py --condition SAGE_enhanced --output_dir results/
    python sage_enhanced.py --condition all --output_dir results/
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_qa_tsv, download_video
from frame_sampler import sample_frames_uniform
from answer_parser import parse_answer
from configs.default import HF_CACHE_DIR, LOCAL_VIDEO_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 태스크 매핑 ────────────────────────────────────────────────────────────────
TASK_MAP = {
    "Intent Inference":              "SSR_intent",
    "Relation Inference":            "SSR_relation",
    "Emotion Inference":             "SSR_emotion",
    "Attitude Inference":            "SSR_attitude",
    "Action Recognition":            "SSU_action",
    "Facial Expression Recognition": "SSU_facial",
    "Human Attribute Identification":"SSU_attribute",
    "Environment Perception":        "SSU_environment",
    "Factual Prediction":            "SDP_factual",
    "Counterfactual Prediction":     "SDP_counterfactual",
}

SSR_CATEGORIES = ["Intent Inference", "Relation Inference", "Emotion Inference", "Attitude Inference"]
SSU_CATEGORIES = ["Action Recognition", "Facial Expression Recognition", "Human Attribute Identification", "Environment Perception"]
SDP_CATEGORIES = ["Factual Prediction", "Counterfactual Prediction"]
ALL_CATEGORIES  = SSR_CATEGORIES + SSU_CATEGORIES + SDP_CATEGORIES

# ── GPT-4o-mini 클라이언트 ─────────────────────────────────────────────────────
def make_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def call_api(client: OpenAI, messages: List[dict], max_tokens: int = 800, temperature: float = 0.0, retries: int = 5) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            logger.warning(f"API 실패 ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"API {retries}회 실패: {last_err}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE ORIGINAL — spec 그대로
# ═══════════════════════════════════════════════════════════════════════════════

_ORIGINAL_GRAPH_PROMPT = """\
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

_ORIGINAL_TASK_HINTS = {
    "SSR_intent":      "Focus on WHY the person acts this way. Use role and event context to infer motivation.",
    "SSR_relation":    "Focus on the dominant social relationship. Use event type and interaction patterns.",
    "SSR_emotion":     "Focus on the emotional state. Use state_change and character emotion fields.",
    "SSR_attitude":    "Focus on the attitude shown toward others. Use interaction type and intensity.",
    "SSU_action":      "Focus on the physical action or activity being performed in the scene.",
    "SSU_facial":      "Focus on the facial expression of the target person.",
    "SSU_attribute":   "Focus on the visible attributes of the person (age, gender, appearance).",
    "SSU_environment": "Focus on the environment and setting visible in the frames.",
    "SDP_factual":     "Based on the current scene dynamics, predict what will most likely happen next.",
    "SDP_counterfactual": "Consider the alternative scenario and reason about the counterfactual outcome.",
}


def _graph_to_text_original(graph: Optional[dict]) -> str:
    if graph is None:
        return "[Social Context: extraction failed]"
    lines = ["[Social Context]"]
    event = graph.get("event", {})
    lines.append(f"- Event: {event.get('type', 'unknown')} ({event.get('stage', 'unknown')})")
    lines.append(f"  Participants: {', '.join(event.get('participants', []))}")
    for char in graph.get("characters", []):
        lines.append(f"- {char['id']}: role={char.get('role','?')}, emotion={char.get('emotion','?')}")
    for intr in graph.get("interactions", []):
        lines.append(f"- {intr['from']}→{intr['to']}: {intr.get('type','?')} (intensity={intr.get('intensity','?')})")
    state = graph.get("state_change", {})
    if state:
        lines.append("- State change:")
        lines.append(f"  Early: {state.get('early', '')}")
        lines.append(f"  Late:  {state.get('late', '')}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE ENHANCED — SocialGPT 패턴 반영
# ═══════════════════════════════════════════════════════════════════════════════

_ENHANCED_GRAPH_PROMPT = """\
You are an expert social scene analyst specializing in human interaction recognition.
Your task: analyze the given video frames and extract the complete social structure as structured JSON.

[RULES]
- Identify ALL visible characters. Use <A>, <B>, <C>... as IDs (consistent symbol system).
- For each character, assign:
  • role (ONE of): aggressor | target | mediator | observer | supporter | initiator | responder
  • emotion: the dominant emotional state visible
  • appearance: estimated age_group (child|teen|adult|elder) and gender (male|female|unknown)
- Identify ONE dominant event type from: argument | greeting | cooperation | confrontation | comfort | negotiation | other
- Event stage: initiating | escalating | resolving | stable
- For each interaction between characters, classify type: conflict | support | ignore | mediate | confront
- Interaction intensity: 0.0 (minimal) to 1.0 (maximal)
- State change: describe the behavioral/emotional shift from frames 1-8 (early) to frames 9-16 (late)

[OUTPUT FORMAT]
Output ONLY valid JSON. No explanation. No markdown.

[EXAMPLE]
{
  "characters": [
    {"id": "A", "role": "initiator", "emotion": "assertive", "appearance": {"age_group": "adult", "gender": "male"}},
    {"id": "B", "role": "responder", "emotion": "hesitant", "appearance": {"age_group": "adult", "gender": "female"}}
  ],
  "interactions": [
    {"from": "A", "to": "B", "type": "confront", "intensity": 0.7}
  ],
  "event": {
    "type": "confrontation",
    "participants": ["A", "B"],
    "stage": "escalating"
  },
  "state_change": {
    "early": "<A> approaches <B> calmly at a desk",
    "late": "<A> leans forward with raised voice, <B> looks away tensely"
  }
}

Now analyze the given frames and output JSON:"""


_ENHANCED_CONTEXT = {
    "SSR_intent": (
        "Intent refers to the underlying motivation or purpose behind a person's action. "
        "To infer intent: (1) identify the person's role in the social event, "
        "(2) consider the event type and stage, "
        "(3) examine how their behavior changed from early to late frames. "
        "Intent is WHY they act, not WHAT they do."
    ),
    "SSR_relation": (
        "Social relation refers to the type of interpersonal relationship between individuals. "
        "Categories: family (parent-child, siblings, grandparent), romantic (lovers/spouses), "
        "professional (colleagues, boss-employee, teacher-student, trainer-trainee), "
        "social (friends, classmates, strangers, acquaintances). "
        "Key signals: interaction type, intensity, roles, and appearance (age gap, gender)."
    ),
    "SSR_emotion": (
        "Emotion refers to the internal affective state of a person at a given moment. "
        "To identify emotion: (1) read the character emotion field in the social context, "
        "(2) track how it changed from early to late frames (state_change), "
        "(3) cross-check with interaction type — e.g., 'conflict' often implies anger/fear. "
        "Focus on the target person specified in the question."
    ),
    "SSR_attitude": (
        "Attitude refers to the evaluative orientation a person shows toward others — "
        "e.g., hostile, friendly, respectful, dismissive, supportive, dominant. "
        "To infer attitude: (1) examine interaction type and intensity, "
        "(2) consider the person's role (aggressor → hostile; supporter → positive), "
        "(3) look at the event stage — escalating often signals negative attitude."
    ),
    "SSU_action": (
        "Action Recognition identifies the physical activity being performed. "
        "Look for body posture, limb movements, and object interactions. "
        "Focus on the dominant ongoing action rather than background details."
    ),
    "SSU_facial": (
        "Facial Expression Recognition identifies the emotional expression shown on a person's face. "
        "Key indicators: eyebrow position, eye openness, mouth shape, cheek tension. "
        "Focus on the face of the target person specified in the question."
    ),
    "SSU_attribute": (
        "Human Attribute Identification recognizes visual properties of a person: "
        "age group (child/teen/adult/elder), gender, clothing, and other appearance cues. "
        "Use the appearance field in social context as a reference."
    ),
    "SSU_environment": (
        "Environment Perception identifies the setting and location of the scene. "
        "Look for background elements: furniture, walls, outdoor/indoor cues, lighting, props. "
        "Consider what activity the space is designed for."
    ),
    "SDP_factual": (
        "Factual Prediction infers the most probable next event given the current scene. "
        "Use the event stage (initiating/escalating/resolving/stable) and state_change trajectory "
        "to extrapolate what action or outcome naturally follows."
    ),
    "SDP_counterfactual": (
        "Counterfactual Prediction reasons about an alternative outcome if a specific condition changed. "
        "Identify the counterfactual premise in the question, then reason about how the social "
        "dynamics (roles, interactions, event stage) would differ under that premise."
    ),
}

_ENHANCED_GUIDANCE = {
    "SSR_intent": (
        "Example reasoning: "
        "If <A> has role=initiator, event=confrontation (escalating), and state_change shows "
        "A approaching B who is avoiding eye contact → A's intent is likely 'to confront or pressure B'."
    ),
    "SSR_relation": (
        "Example reasoning: "
        "If interaction type=support, intensity=0.8, A is adult/female with role=supporter "
        "and B is child/unknown → the relation is likely parent-child rather than colleagues."
    ),
    "SSR_emotion": (
        "Example reasoning: "
        "If character emotion=anxious, state_change.late='B steps back and avoids eye contact', "
        "and interaction type=confront → B's emotion is fear or discomfort, not anger."
    ),
    "SSR_attitude": (
        "Example reasoning: "
        "If A→B interaction type=confront, intensity=0.9, event stage=escalating, "
        "and A's role=aggressor → A's attitude toward B is hostile/dominant."
    ),
    "SSU_action": (
        "Example reasoning: "
        "If state_change.early shows characters standing still and late shows rapid arm movement "
        "with objects being exchanged → the action is likely 'handshake' or 'passing objects'."
    ),
    "SSU_facial": (
        "Example reasoning: "
        "If character emotion=anxious, state_change.late shows the character with raised eyebrows "
        "and wide eyes → the facial expression is likely 'surprised' or 'fearful'."
    ),
    "SSU_attribute": (
        "Example reasoning: "
        "If appearance shows age_group=elder, gender=male, and the character wears formal attire "
        "in a professional setting → the attribute answer involves an older male in formal wear."
    ),
    "SSU_environment": (
        "Example reasoning: "
        "If state_change describes desks, whiteboards, and seated students → "
        "the environment is a classroom or educational setting."
    ),
    "SDP_factual": (
        "Example reasoning: "
        "If event stage=escalating and interaction type=conflict with high intensity → "
        "the most likely next event is a confrontation or separation between the characters."
    ),
    "SDP_counterfactual": (
        "Example reasoning: "
        "If the premise is 'if A had not intervened' and A's role=mediator, event stage=resolving → "
        "without A, the conflict would likely have continued escalating."
    ),
}


def _graph_to_text_enhanced(graph: Optional[dict], task_type: str) -> str:
    """task_type에 따라 관련 정보를 앞으로 배치."""
    if graph is None:
        return "[Social Context: extraction failed]"

    event = graph.get("event", {})
    characters = graph.get("characters", [])
    interactions = graph.get("interactions", [])
    state = graph.get("state_change", {})

    def event_block():
        lines = []
        lines.append(f"- Event: {event.get('type', 'unknown')} ({event.get('stage', 'unknown')})")
        lines.append(f"  Participants: {', '.join(event.get('participants', []))}")
        return lines

    def role_block():
        lines = []
        for char in characters:
            app = char.get("appearance", {})
            age = app.get("age_group", "?")
            gender = app.get("gender", "?")
            lines.append(
                f"- <{char['id']}>: role={char.get('role','?')}, emotion={char.get('emotion','?')}, "
                f"appearance=({age}/{gender})"
            )
        return lines

    def interaction_block():
        lines = []
        for intr in interactions:
            lines.append(
                f"- <{intr['from']}>→<{intr['to']}>: "
                f"{intr.get('type','?')} (intensity={intr.get('intensity','?')})"
            )
        return lines

    def state_block():
        lines = []
        if state:
            lines.append("- State change:")
            lines.append(f"  Early: {state.get('early', '')}")
            lines.append(f"  Late:  {state.get('late', '')}")
        return lines

    # task_type별 우선순서
    order = {
        "SSR_intent":        [event_block, role_block, state_block, interaction_block],
        "SSR_relation":      [interaction_block, role_block, event_block, state_block],
        "SSR_emotion":       [state_block, role_block, interaction_block, event_block],
        "SSR_attitude":      [interaction_block, state_block, event_block, role_block],
        "SSU_action":        [state_block, event_block, role_block, interaction_block],
        "SSU_facial":        [role_block, state_block, interaction_block, event_block],
        "SSU_attribute":     [role_block, event_block, state_block, interaction_block],
        "SSU_environment":   [state_block, event_block, role_block, interaction_block],
        "SDP_factual":       [event_block, state_block, interaction_block, role_block],
        "SDP_counterfactual":[event_block, interaction_block, state_block, role_block],
    }.get(task_type, [event_block, role_block, interaction_block, state_block])

    lines = ["[Social Context]"]
    for block_fn in order:
        lines.extend(block_fn())
    return "\n".join(lines)


def _build_enhanced_qa_messages(
    frames_b64: List[str],
    graph_text: str,
    question: str,
    options: List[str],
    task_type: str,
) -> List[dict]:
    context = _ENHANCED_CONTEXT.get(task_type, "")
    guidance = _ENHANCED_GUIDANCE.get(task_type, "")
    options_text = "\n".join(options)

    prompt = (
        f"[EXPECTATION]\n"
        f"You are an expert in social interaction analysis. "
        f"Using the extracted social context below and the video frames, "
        f"select the single best answer to the multiple-choice question.\n\n"
        f"[CONTEXT — {task_type.replace('_', ' ').upper()}]\n"
        f"{context}\n\n"
        f"[SOCIAL CONTEXT FROM VIDEO]\n"
        f"{graph_text}\n\n"
        f"[GUIDANCE]\n"
        f"{guidance}\n\n"
        f"[QUESTION]\n"
        f"{question}\n\n"
        f"[OPTIONS]\n"
        f"{options_text}\n\n"
        f"Answer with ONLY the single option letter from the choices above:"
    )

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}
        for b64 in frames_b64
    ]
    user_content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": user_content}]


# ═══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_graph_json(raw: str) -> Optional[dict]:
    try:
        text = raw
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return None


def _extract_graph(frames_b64: List[str], client: OpenAI, prompt: str) -> Optional[dict]:
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}
        for b64 in frames_b64
    ]
    user_content.append({"type": "text", "text": prompt})
    raw = call_api(client, [{"role": "user", "content": user_content}], max_tokens=900, temperature=0.0)
    return _parse_graph_json(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# PLAIN baseline (gpt-4o-mini, 프레임만)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_plain(frames_b64: List[str], question: str, options: List[str], client: OpenAI) -> dict:
    options_text = "\n".join(options)
    prompt = (
        f"You are analyzing a social interaction video. "
        f"Answer the following multiple-choice question based on the video frames.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer with ONLY the single option letter from the choices above:"
    )
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}
        for b64 in frames_b64
    ]
    user_content.append({"type": "text", "text": prompt})
    raw = call_api(client, [{"role": "user", "content": user_content}], max_tokens=10, temperature=0.0)
    return {"raw_output": raw, "predicted": parse_answer(raw), "graph": None}


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE ORIGINAL pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sage_original(frames_b64: List[str], question: str, options: List[str], task_type: str, client: OpenAI) -> dict:
    graph = _extract_graph(frames_b64, client, _ORIGINAL_GRAPH_PROMPT)
    graph_text = _graph_to_text_original(graph)

    hint = _ORIGINAL_TASK_HINTS.get(task_type, "Focus on social context to answer.")
    options_text = "\n".join(options)
    prompt = (
        f"You are analyzing a social interaction video.\n\n"
        f"{graph_text}\n\n"
        f"{hint}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer with ONLY the single option letter from the choices above:"
    )
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}
        for b64 in frames_b64
    ]
    user_content.append({"type": "text", "text": prompt})
    raw = call_api(client, [{"role": "user", "content": user_content}], max_tokens=10, temperature=0.0)
    return {"raw_output": raw, "predicted": parse_answer(raw), "graph": graph}


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE ENHANCED pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sage_enhanced(frames_b64: List[str], question: str, options: List[str], task_type: str, client: OpenAI) -> dict:
    graph = _extract_graph(frames_b64, client, _ENHANCED_GRAPH_PROMPT)
    graph_text = _graph_to_text_enhanced(graph, task_type)
    messages = _build_enhanced_qa_messages(frames_b64, graph_text, question, options, task_type)
    raw = call_api(client, messages, max_tokens=10, temperature=0.0)
    return {"raw_output": raw, "predicted": parse_answer(raw), "graph": graph}


# ═══════════════════════════════════════════════════════════════════════════════
# 샘플링: random seed 42, 태스크별 n_per_task개
# ═══════════════════════════════════════════════════════════════════════════════

def sample_subset(n_ssr: int = 50, n_ssu: int = 50, n_sdp: int = 50, seed: int = 42):
    all_samples = load_qa_tsv()
    task_buckets = defaultdict(list)
    for s in all_samples:
        if s.category in ALL_CATEGORIES:
            task_buckets[s.category].append(s)

    rng = random.Random(seed)
    subset = []
    for cat, n in (
        [(c, n_ssr) for c in SSR_CATEGORIES] +
        [(c, n_ssu) for c in SSU_CATEGORIES] +
        [(c, n_sdp) for c in SDP_CATEGORIES]
    ):
        bucket = task_buckets[cat]
        chosen = rng.sample(bucket, min(n, len(bucket)))
        subset.extend(chosen)
    return subset


# ═══════════════════════════════════════════════════════════════════════════════
# 실험 실행
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(condition: str, n_ssr: int = 50, n_ssu: int = 50, n_sdp: int = 50, output_dir: str = "results") -> dict:
    """
    condition: 'Plain' | 'SAGE_original' | 'SAGE_enhanced'
    returns: {task_type: [{'sample_id', 'prediction', 'ground_truth', 'correct'}, ...]}
    """
    client = make_client()
    subset = sample_subset(n_ssr=n_ssr, n_ssu=n_ssu, n_sdp=n_sdp)
    results = defaultdict(list)
    graph_fail = 0
    total = len(subset)
    start = time.time()

    logger.info(f"=== {condition} 시작 ({total}개) ===")

    for i, sample in enumerate(subset):
        task_type = TASK_MAP[sample.category]
        try:
            video_path = download_video(
                sample.video_path,
                subtitle_condition="origin",
                cache_dir=HF_CACHE_DIR,
                local_video_dir=LOCAL_VIDEO_DIR,
            )
            frames_b64 = sample_frames_uniform(video_path, num_frames=16)

            if condition == "Plain":
                out = _run_plain(frames_b64, sample.question, sample.options, client)
            elif condition == "SAGE_original":
                out = _run_sage_original(frames_b64, sample.question, sample.options, task_type, client)
            else:
                out = _run_sage_enhanced(frames_b64, sample.question, sample.options, task_type, client)

            if out["graph"] is None and condition != "Plain":
                graph_fail += 1

            correct = (out["predicted"] is not None) and (out["predicted"] == sample.correct_answer_index)
            results[task_type].append({
                "sample_id": sample.question_id,
                "prediction": out["predicted"],
                "ground_truth": sample.correct_answer_index,
                "correct": correct,
            })

            elapsed = time.time() - start
            done = sum(len(v) for v in results.values())
            acc = sum(r["correct"] for v in results.values() for r in v) / done
            logger.info(
                f"[{i+1}/{total}] {sample.question_id} | {task_type} | "
                f"pred={out['predicted']} gt={sample.correct_answer_index} | "
                f"{'O' if correct else 'X'} | acc={acc*100:.1f}% | {elapsed:.0f}s"
            )

        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "Entry Not Found" in err_str:
                logger.warning(f"[{i+1}] {sample.question_id} 비디오 없음(404), 건너뜀")
                total -= 1
                continue
            logger.error(f"[{i+1}] {sample.question_id} 실패: {e}")
            if condition != "Plain":
                graph_fail += 1
            results[task_type].append({
                "sample_id": sample.question_id,
                "prediction": None,
                "ground_truth": sample.correct_answer_index,
                "correct": False,
            })

        time.sleep(0.5)

    if condition != "Plain":
        logger.info(f"graph 추출 실패: {graph_fail}/{total}")

    # 저장
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{condition.lower()}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(results), f, ensure_ascii=False, indent=2)
    logger.info(f"저장: {out_path}")

    return dict(results)


def compute_accuracy(results: dict) -> dict:
    accs = {}
    all_correct, all_total = 0, 0
    for task, records in results.items():
        correct = sum(r["correct"] for r in records)
        total = len(records)
        accs[task] = correct / total if total else 0.0
        all_correct += correct
        all_total += total
    accs["avg"] = all_correct / all_total if all_total else 0.0
    return accs


def print_comparison_table(all_results: Dict[str, dict]):
    task_keys = [
        "SSR_intent", "SSR_relation", "SSR_emotion", "SSR_attitude",
        "SSU_action", "SSU_facial", "SSU_attribute", "SSU_environment",
        "SDP_factual", "SDP_counterfactual",
    ]
    short = {
        "SSR_intent": "int", "SSR_relation": "rel", "SSR_emotion": "emo", "SSR_attitude": "att",
        "SSU_action": "act", "SSU_facial": "fac", "SSU_attribute": "atr", "SSU_environment": "env",
        "SDP_factual": "fct", "SDP_counterfactual": "cft",
    }
    header = f"{'condition':<20}" + "".join(f" {short[t]:>6}" for t in task_keys) + f" {'avg':>6}"
    print("\n" + "=" * 90)
    print(header)
    print("-" * 90)
    for cond, results in all_results.items():
        accs = compute_accuracy(results)
        row = f"{cond:<20}"
        for t in task_keys:
            v = accs.get(t)
            row += f"  {v*100:>4.1f}%" if v is not None else f"  {'--':>5}"
        row += f"  {accs['avg']*100:>4.1f}%"
        print(row)
    print("=" * 90)


def save_comparison(all_results: Dict[str, dict], output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "results_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"비교 결과 저장: {out_path}")


def save_report(all_results: Dict[str, dict], output_dir: str):
    task_keys = ["SSR_intent", "SSR_relation", "SSR_emotion", "SSR_attitude"]
    accs_map = {cond: compute_accuracy(res) for cond, res in all_results.items()}

    lines = []
    lines.append("# SAGE Experiment Report: Task A\n")
    lines.append("## 1. 결과 테이블\n")
    lines.append(f"| condition | intent | relation | emotion | attitude | avg |")
    lines.append(f"|-----------|--------|----------|---------|----------|-----|")
    for cond, accs in accs_map.items():
        row = f"| {cond} "
        for t in task_keys:
            row += f"| {accs.get(t,0)*100:.1f}% "
        row += f"| {accs['avg']*100:.1f}% |"
        lines.append(row)

    lines.append("\n## 2. 보강 내용 요약\n")
    lines.append("### GRAPH_EXTRACTION_PROMPT 변경사항")
    lines.append("- **역할 강화**: expert identity + 명시적 규칙 섹션 ([RULES], [OUTPUT FORMAT], [EXAMPLE])")
    lines.append("- **appearance 필드 추가**: age_group (child|teen|adult|elder), gender (male|female|unknown)")
    lines.append("- **symbol 시스템**: `<A>`, `<B>` 형식으로 캐릭터 참조 일관화 (SocialGPT의 `<P1>/<P2>` 방식 차용)")
    lines.append("- **event.type 카테고리 제한**: argument|greeting|cooperation|confrontation|comfort|negotiation|other")
    lines.append("- **few-shot example**: 1개 JSON 예시 포함으로 파싱 안정성 향상\n")
    lines.append("### build_qa_prompt 변경사항")
    lines.append("- **[EXPECTATION] 섹션**: 모델 역할과 태스크 목표 명시")
    lines.append("- **[CONTEXT] 섹션**: 태스크별 정의 및 추론 기준 주입 (SocialGPT의 관계 정의 방식 차용)")
    lines.append("- **[GUIDANCE] 섹션**: 태스크별 reasoning example 1개 in-context 추가")
    lines.append("- **graph_to_text 재정렬**: task_type별 관련 정보 우선 배치")
    lines.append("  - SSR_intent: event → role → state_change → interaction")
    lines.append("  - SSR_relation: interaction → role → event → state_change")
    lines.append("  - SSR_emotion: state_change → role → interaction → event")
    lines.append("  - SSR_attitude: interaction → state_change → event → role\n")

    lines.append("## 3. 오류 분석\n")
    for cond, results in all_results.items():
        if cond == "Plain":
            continue
        total = sum(len(v) for v in results.values())
        none_pred = sum(1 for v in results.values() for r in v if r["prediction"] is None)
        lines.append(f"- **{cond}**: 예측 실패(None) {none_pred}/{total}건")

    lines.append("\n## 4. Next Step (Task B 준비)\n")
    lines.append("- SocialGPT `social-story-main-PIPA/main.py`를 SIV-Bench에 맞게 포팅")
    lines.append("  - BLIP-2 / SAM 없이 GPT-4o-mini가 직접 story 생성")
    lines.append("  - 동일 200개 seed-42 subset에서 SAGE_enhanced vs SocialGPT_story 비교")
    lines.append("- 전체 4,951개 SSR 샘플 풀 실험 검토")

    report_path = Path(output_dir) / "experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"보고서 저장: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAGE Enhanced 실험")
    parser.add_argument(
        "--condition",
        choices=["Plain", "SAGE_original", "SAGE_enhanced", "all"],
        default="all",
        help="실행할 조건 (all = 세 조건 모두)",
    )
    parser.add_argument("--n_ssr",  type=int, default=50,  help="SSR 태스크별 샘플 수")
    parser.add_argument("--n_ssu",  type=int, default=50,  help="SSU 태스크별 샘플 수")
    parser.add_argument("--n_sdp",  type=int, default=50,  help="SDP 태스크별 샘플 수")
    parser.add_argument("--output_dir", type=str, default="results", help="결과 저장 경로")
    args = parser.parse_args()

    conditions = ["Plain", "SAGE_original", "SAGE_enhanced"] if args.condition == "all" else [args.condition]
    all_results = {}

    # 이미 저장된 결과 재사용
    for cond in conditions:
        cached = Path(args.output_dir) / f"{cond.lower()}_results.json"
        if cached.exists():
            logger.info(f"{cond}: 캐시 파일 발견, 재사용 ({cached})")
            with open(cached) as f:
                all_results[cond] = json.load(f)
        else:
            all_results[cond] = run_experiment(
                cond, n_ssr=args.n_ssr, n_ssu=args.n_ssu, n_sdp=args.n_sdp,
                output_dir=args.output_dir,
            )

    print_comparison_table(all_results)

    if len(all_results) > 1:
        save_comparison(all_results, args.output_dir)
        save_report(all_results, args.output_dir)


if __name__ == "__main__":
    main()