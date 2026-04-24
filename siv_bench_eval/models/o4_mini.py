"""
models/o4_mini.py

OpenAI o4-mini API 호출 래퍼.
- 환경변수 OPENAI_API_KEY, OPENAI_BASE_URL 사용
- 비전 입력(base64 이미지) 지원
- 재시도 로직 포함
"""

import logging
import os
import time
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "o4-mini"
MAX_RETRIES = 5
RETRY_DELAY = 2.0  # seconds


class O4Mini:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Args:
            api_key: OpenAI API key. None이면 환경변수 OPENAI_API_KEY 사용
            base_url: API base URL. None이면 환경변수 OPENAI_BASE_URL 사용
                      (학교 프록시 서버가 있으면 여기에 입력)
            model: 사용할 모델명
        """
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )

    def chat(self, messages: List[dict]) -> str:
        """messages를 API에 전송하고 응답 텍스트를 반환한다.

        Args:
            messages: OpenAI chat completions 형식의 messages 리스트

        Returns:
            모델 응답 텍스트

        Raises:
            RuntimeError: MAX_RETRIES 초과 시
        """
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # o4-mini는 reasoning tokens를 내부적으로 사용하므로
                    # max_completion_tokens를 충분히 크게 설정해야 함
                    # reasoning ~64토큰 + 실제 답변 토큰을 합산한 여유값
                    max_completion_tokens=2048,
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                last_error = e
                logger.warning(f"API 호출 실패 (시도 {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        raise RuntimeError(f"API 호출 {MAX_RETRIES}회 실패: {last_error}")
