"""
configs/default.py

실험 설정값. 실행 전에 이 파일 또는 .env 파일을 수정하세요.
"""

import os
from pathlib import Path

# ── 프로젝트 루트 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── HuggingFace 캐시 위치 ──────────────────────────────────────
# 서버 홈 디렉토리 아래 datasets/ 폴더를 캐시로 사용
# (기본값 ~/.cache/huggingface를 사용하려면 None으로 설정)
HF_CACHE_DIR = str(Path.home() / "datasets" / "siv_bench_cache")

# ── 미리 다운로드한 비디오 루트 (없으면 None) ───────────────────
# 구조: LOCAL_VIDEO_DIR/{subtitle_condition}/{video_path}
LOCAL_VIDEO_DIR = None

# ── 평가 설정 ──────────────────────────────────────────────────
SUBTITLE_CONDITION = "w_sub"   # 'origin' | 'w_sub' | 'wo_sub'
NUM_FRAMES = 16
MAX_SAMPLES = 50               # 50개 테스트. 전체 실행 시 None으로 변경

# ── 결과 저장 경로 ─────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULT_SAVE_PATH = str(RESULTS_DIR / f"plain_baseline_{SUBTITLE_CONDITION}.json")

# ── API 설정 (우선순위: .env > 환경변수 > 여기) ────────────────
# 직접 입력하지 말고 .env 파일을 사용하세요
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")  # 학교 프록시 URL
