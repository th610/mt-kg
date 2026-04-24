"""
frame_sampler.py

비디오 파일에서 균일하게 N개 프레임을 추출한다.
- OpenCV(cv2) 기반
- 반환값: base64 인코딩된 JPEG 문자열 리스트 (o4-mini API 입력 형식)
"""

import base64
import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_NUM_FRAMES = 16


def sample_frames_uniform(video_path: str, num_frames: int = DEFAULT_NUM_FRAMES) -> List[str]:
    """비디오에서 균일 간격으로 num_frames개 프레임을 추출한다.

    Args:
        video_path: 로컬 비디오 파일 경로 (.mp4 등)
        num_frames: 추출할 프레임 수 (기본 16)

    Returns:
        base64 인코딩된 JPEG 이미지 문자열 리스트 (길이 = num_frames)
        비디오가 num_frames보다 짧으면 마지막 프레임으로 패딩

    Raises:
        ValueError: 비디오를 열 수 없는 경우
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"프레임 수를 읽을 수 없습니다: {video_path}")

    # 균일 샘플링 인덱스 계산 (0 ~ total_frames-1 구간을 num_frames 등분)
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        # 양 끝을 포함한 균일 분포
        step = (total_frames - 1) / (num_frames - 1)
        indices = [round(step * i) for i in range(num_frames)]

    frames_b64: List[str] = []
    last_valid_frame: np.ndarray = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            last_valid_frame = frame
        else:
            # 읽기 실패 시 마지막 유효 프레임 재사용
            logger.warning(f"프레임 {idx} 읽기 실패, 이전 프레임 사용: {video_path}")
            if last_valid_frame is None:
                # 아무 것도 없으면 검은 프레임
                last_valid_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame = last_valid_frame

        frames_b64.append(_frame_to_base64(frame))

    cap.release()

    # total_frames < num_frames 였을 때 패딩
    while len(frames_b64) < num_frames:
        frames_b64.append(frames_b64[-1])

    logger.debug(f"추출 완료: {len(frames_b64)}프레임 from {video_path}")
    return frames_b64


def _frame_to_base64(frame: np.ndarray) -> str:
    """OpenCV BGR 프레임을 base64 JPEG 문자열로 변환."""
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("JPEG 인코딩 실패")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
