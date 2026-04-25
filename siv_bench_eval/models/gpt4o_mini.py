"""
models/gpt4o_mini.py

GPT-4o-mini API 호출 래퍼.
- temperature, max_tokens을 호출별로 조정 가능 (SAGE Step 1/2가 다른 설정을 씀)
- 재시도 로직 포함
"""

import logging
import os
import time
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
RETRY_DELAY = 2.0


class GPT4oMini:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str:
        """messages를 API에 전송하고 응답 텍스트를 반환한다.

        Args:
            messages: OpenAI chat completions 형식의 messages 리스트
            max_tokens: 최대 출력 토큰 수
            temperature: 샘플링 온도 (0.0 = greedy)

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
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                last_error = e
                logger.warning(f"API 호출 실패 (시도 {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        raise RuntimeError(f"API 호출 {MAX_RETRIES}회 실패: {last_error}")