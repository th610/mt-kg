"""
answer_parser.py

모델 출력 문자열에서 선택지 알파벳(A~N)을 파싱한다.

전략:
1. 응답이 단일 알파벳이면 바로 반환
2. "Answer: B" 같은 패턴 추출
3. 첫 번째로 등장하는 대문자 알파벳(A~N) 추출
4. 전부 실패하면 None 반환
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# SIV-Bench 최대 선택지 N까지 (14개 관계 라벨 기준)
VALID_LETTERS = set("ABCDEFGHIJKLMN")


def parse_answer(model_output: str) -> Optional[str]:
    """모델 출력에서 선택지 알파벳을 추출한다.

    Args:
        model_output: 모델이 반환한 원본 문자열

    Returns:
        'A'~'N' 중 하나의 대문자 문자열, 파싱 실패 시 None
    """
    if not model_output:
        return None

    text = model_output.strip()

    # 1. 단일 알파벳 (e.g. "B" or "B.")
    if re.fullmatch(r"[A-Na-n]\.?", text):
        return text[0].upper()

    # 2. "Answer: B" 또는 "answer is B" 패턴
    m = re.search(r"(?:answer(?:\s+is)?|option)[:\s]+([A-Na-n])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. "(B)" 또는 "B)" 패턴
    m = re.search(r"\(?([A-Na-n])\)?[.)\s]", text)
    if m:
        return m.group(1).upper()

    # 4. 첫 번째로 등장하는 유효 알파벳
    m = re.search(r"\b([A-Na-n])\b", text)
    if m:
        return m.group(1).upper()

    logger.warning(f"파싱 실패: {repr(text[:100])}")
    return None
