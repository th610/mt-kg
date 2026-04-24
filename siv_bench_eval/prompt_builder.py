"""
prompt_builder.py

question + options + frames (+ optional subtitle) 를 묶어서
o4-mini API에 전달할 messages 리스트를 만든다.

반환 형식: OpenAI messages 리스트
  [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": [ {image}, {image}, ..., {text} ]}
  ]
"""

from typing import List, Optional


SYSTEM_PROMPT = (
    "You are an expert at understanding human social interactions in videos. "
    "You will be shown frames from a video clip and asked a multiple-choice question "
    "about the social interaction depicted. "
    "Respond with ONLY the letter of the correct answer (e.g. A, B, C, ...). "
    "Do not add any explanation or punctuation."
)


def build_plain_prompt(
    question: str,
    options: List[str],
    frames_b64: List[str],
    subtitle: Optional[str] = None,
) -> List[dict]:
    """plain baseline용 messages 리스트를 생성한다.

    Args:
        question: 질문 텍스트
        options: ['A. text', 'B. text', ...] 형식의 선택지 리스트
        frames_b64: base64 인코딩된 JPEG 프레임 리스트
        subtitle: 자막 텍스트 (w_sub 조건에서 사용, None이면 생략)

    Returns:
        OpenAI chat completions API에 전달할 messages 리스트
    """
    user_content = []

    # 프레임 이미지 추가
    for b64 in frames_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    # 질문 텍스트 구성
    question_text = f"Question: {question}\n\nOptions:\n"
    for opt in options:
        question_text += f"  {opt}\n"

    # 자막이 있으면 추가
    if subtitle:
        question_text += f"\nSubtitle / Dialogue:\n{subtitle}\n"

    question_text += "\nAnswer with only the option letter (e.g. A):"

    user_content.append({"type": "text", "text": question_text})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
