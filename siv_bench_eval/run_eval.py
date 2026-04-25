"""
run_eval.py

실험 실행 진입점.

사용법:
    cd /home/mindrium-admin3/siv_bench_eval
    python run_eval.py                          # plain baseline (기본 설정)
    python run_eval.py --mode mtkg              # mtKG baseline
    python run_eval.py --max_samples 100        # 100개
    python run_eval.py --max_samples 0          # 전체 (0 = None)
    python run_eval.py --condition wo_sub       # 자막 없는 조건
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# .env 파일 로드 (python-dotenv가 없으면 환경변수만 사용)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# siv_bench_eval/ 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from configs.default import (
    HF_CACHE_DIR,
    LOCAL_VIDEO_DIR,
    SUBTITLE_CONDITION,
    NUM_FRAMES,
    MAX_SAMPLES,
    RESULT_SAVE_PATH,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


CATEGORY_PRESETS = {
    "all": None,
    "ssr": [
        "Relation Inference",
        "Emotion Inference",
        "Intent Inference",
        "Attitude Inference",
    ],
    "core_social": [
        "Relation Inference",
        "Intent Inference",
        "Attitude Inference",
    ],
    "relation_only": [
        "Relation Inference",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="SIV-Bench evaluation (plain / mtkg baseline)")
    parser.add_argument(
        "--condition",
        choices=["origin", "w_sub", "wo_sub"],
        default=SUBTITLE_CONDITION,
        help="subtitle 조건",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=MAX_SAMPLES,
        help="평가할 최대 샘플 수 (0이면 전체)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=NUM_FRAMES,
        help="비디오당 샘플링 프레임 수",
    )
    parser.add_argument(
        "--mode",
        choices=["plain", "mtkg", "sage"],
        default="plain",
        help="실행 모드: plain (plain baseline) | mtkg (mtKG baseline) | sage (SAGE)",
    )
    parser.add_argument(
        "--category_preset",
        choices=list(CATEGORY_PRESETS.keys()),
        default="all",
        help="미리 정의된 category subset",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help="쉼표로 구분한 category 직접 지정. 예: 'Relation Inference,Intent Inference'",
    )
    return parser.parse_args()


def resolve_categories(category_preset: str, categories_arg: str) -> tuple[Optional[List[str]], str]:
    if categories_arg.strip():
        categories = [c.strip() for c in categories_arg.split(",") if c.strip()]
        if not categories:
            raise ValueError("--categories가 비어 있습니다.")
        return categories, "custom"

    preset_categories = CATEGORY_PRESETS[category_preset]
    if preset_categories is None:
        return None, "all"
    return list(preset_categories), category_preset


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("eval.log", encoding="utf-8"),
        ],
    )

    args = parse_args()
    max_samples = args.max_samples if args.max_samples > 0 else None
    categories, category_tag = resolve_categories(args.category_preset, args.categories)

    logging.getLogger(__name__).info(
        f"실행 설정: mode={args.mode}, condition={args.condition}, "
        f"max_samples={max_samples}, num_frames={args.num_frames}, "
        f"category_tag={category_tag}, categories={categories or 'ALL'}"
    )

    result_path = str(
        Path(RESULT_SAVE_PATH).parent
        / f"{args.mode}_baseline_{args.condition}_{category_tag}_f{args.num_frames}.json"
    )

    if args.mode == "plain":
        from pipeline.plain_baseline import run_plain_baseline
        run_plain_baseline(
            subtitle_condition=args.condition,
            num_frames=args.num_frames,
            max_samples=max_samples,
            categories=categories,
            cache_dir=HF_CACHE_DIR,
            local_video_dir=LOCAL_VIDEO_DIR,
            result_save_path=result_path,
            api_key=OPENAI_API_KEY or None,
            base_url=OPENAI_BASE_URL or None,
        )
    elif args.mode == "mtkg":
        from pipeline.mtkg_baseline import run_mtkg_baseline
        run_mtkg_baseline(
            subtitle_condition=args.condition,
            num_frames=args.num_frames,
            max_samples=max_samples,
            categories=categories,
            cache_dir=HF_CACHE_DIR,
            local_video_dir=LOCAL_VIDEO_DIR,
            result_save_path=result_path,
            api_key=OPENAI_API_KEY or None,
            base_url=OPENAI_BASE_URL or None,
        )
    else:  # sage
        from pipeline.sage import run_sage_pipeline
        run_sage_pipeline(
            subtitle_condition=args.condition,
            num_frames=args.num_frames,
            max_samples=max_samples,
            categories=categories,
            cache_dir=HF_CACHE_DIR,
            local_video_dir=LOCAL_VIDEO_DIR,
            result_save_path=result_path,
            api_key=OPENAI_API_KEY or None,
            base_url=OPENAI_BASE_URL or None,
        )


if __name__ == "__main__":
    main()
