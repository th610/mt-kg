"""
SIV-Bench dataset loader.

- Downloads SIV-Bench-QA.tsv from HuggingFace Hub (3.36 MB, one-time)
- Provides on-demand per-video download to avoid pulling the full 45 GB at once
- Subtitle conditions: 'origin', 'w_sub', 'wo_sub'

TSV columns (confirmed from HF repo):
    index, video_path, video, question_id, question, answer,
    options, correct_answer_index, category

Video paths in HF repo:
    {condition}/{video_path}   e.g.  origin/boss-employee/video_141.mp4
"""

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO_ID = "Fancylalala/SIV-Bench"
QA_TSV_FILENAME = "SIV-Bench-QA.tsv"
SUBTITLE_CONDITIONS = ("origin", "w_sub", "wo_sub")


@dataclass
class SIVBenchSample:
    """One QA item from SIV-Bench-QA.tsv."""
    index: str                    # e.g. '0001'
    video_path: str               # e.g. 'boss-employee/video_141.mp4'
    question_id: str              # e.g. '0001-1'
    question: str
    answer: str                   # full answer text
    options: List[str]            # ['A. text', 'B. text', ...]
    correct_answer_index: str     # 'A', 'B', ..., 'N'
    category: str                 # e.g. 'Relation Inference', 'Environment Perception'


def _parse_options(options_str: str) -> List[str]:
    """Split 'A. foo, B. bar, C. baz' into ['A. foo', 'B. bar', 'C. baz'].

    Handles commas that may appear inside option text by splitting only
    at patterns like ', A.' or ', B.' (capital letter preceded by comma+space).
    """
    parts = re.split(r",\s*(?=[A-N]\.)", options_str)
    return [p.strip() for p in parts if p.strip()]


def load_qa_tsv(cache_dir: Optional[str] = None) -> List[SIVBenchSample]:
    """Download SIV-Bench-QA.tsv and return all QA samples.

    The TSV is only ~3.4 MB so downloading it once is fine.
    Subsequent calls reuse the HF Hub cache.

    Args:
        cache_dir: HuggingFace Hub cache directory (None = HF default).

    Returns:
        List of SIVBenchSample (all conditions share the same QA pairs).
    """
    tsv_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=QA_TSV_FILENAME,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    logger.info(f"QA TSV loaded from: {tsv_path}")

    samples: List[SIVBenchSample] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append(
                SIVBenchSample(
                    index=row["index"],
                    video_path=row["video_path"],
                    question_id=row["question_id"],
                    question=row["question"],
                    answer=row["answer"],
                    options=_parse_options(row["options"]),
                    correct_answer_index=row["correct_answer_index"],
                    category=row["category"],
                )
            )
    logger.info(f"Loaded {len(samples)} QA samples from TSV")
    return samples


def get_video_local_path(
    video_path: str,
    subtitle_condition: str,
    local_video_dir: Optional[str],
) -> Optional[str]:
    """Check if video already exists locally. Returns path string or None."""
    if local_video_dir is None:
        return None
    candidate = Path(local_video_dir) / subtitle_condition / video_path
    return str(candidate) if candidate.exists() else None


def download_video(
    video_path: str,
    subtitle_condition: str = "origin",
    cache_dir: Optional[str] = None,
    local_video_dir: Optional[str] = None,
) -> str:
    """Return local path to a video file, downloading from HF Hub if needed.

    Args:
        video_path: relative path from TSV, e.g. 'boss-employee/video_141.mp4'
        subtitle_condition: one of 'origin', 'w_sub', 'wo_sub'
        cache_dir: HF Hub cache directory
        local_video_dir: root dir of pre-downloaded videos (skips HF download)

    Returns:
        Absolute local path to the video file.
    """
    if subtitle_condition not in SUBTITLE_CONDITIONS:
        raise ValueError(
            f"Invalid subtitle_condition '{subtitle_condition}'. "
            f"Choose from {SUBTITLE_CONDITIONS}."
        )

    # Check local cache first
    local_path = get_video_local_path(video_path, subtitle_condition, local_video_dir)
    if local_path is not None:
        logger.debug(f"Using local video: {local_path}")
        return local_path

    hf_filename = f"{subtitle_condition}/{video_path}"
    logger.debug(f"Downloading from HF: {hf_filename}")
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=hf_filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def iter_samples(
    subtitle_condition: str = "origin",
    cache_dir: Optional[str] = None,
    local_video_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
) -> Iterator[tuple]:
    """Iterate over (SIVBenchSample, video_local_path) pairs.

    Videos are downloaded on demand — one at a time — to avoid pulling
    the full 45 GB dataset upfront.

    Args:
        subtitle_condition: one of 'origin', 'w_sub', 'wo_sub'
        cache_dir: HF Hub cache directory
        local_video_dir: root dir of pre-downloaded videos
        max_samples: stop after this many QA items (None = all)
        categories: include only these category names (None = all)

    Yields:
        (SIVBenchSample, video_local_path: str)
    """
    samples = load_qa_tsv(cache_dir=cache_dir)
    category_filter = set(categories) if categories else None
    count = 0
    for sample in samples:
        if category_filter is not None and sample.category not in category_filter:
            continue
        if max_samples is not None and count >= max_samples:
            break
        try:
            video_path = download_video(
                sample.video_path,
                subtitle_condition=subtitle_condition,
                cache_dir=cache_dir,
                local_video_dir=local_video_dir,
            )
        except Exception as e:
            logger.warning(f"영상 다운로드 실패, 건너뜀: {sample.video_path} ({e})")
            continue
        yield sample, video_path
        count += 1
