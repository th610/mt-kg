#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/home/mindrium-admin3/datasets/moviegraphs"
SOURCE_REPO="/home/mindrium-admin3/MovieGraphs-Dataset-CVPR2018-main"
DOWNLOAD_SUBTITLES="no"
DOWNLOAD_FRAMES="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --source-repo)
      SOURCE_REPO="$2"
      shift 2
      ;;
    --download-subtitles)
      DOWNLOAD_SUBTITLES="$2"
      shift 2
      ;;
    --download-frames)
      DOWNLOAD_FRAMES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$TARGET_DIR"/{splits,labels,frames,faces,subtitles,cache,predictions,raw}

for f in split.json movies_list.txt dvds.txt README.md; do
  if [[ -f "$SOURCE_REPO/$f" ]]; then
    cp "$SOURCE_REPO/$f" "$TARGET_DIR/splits/$f"
  fi
done

if [[ -d "$SOURCE_REPO/py3loader_new" ]]; then
  rm -rf "$TARGET_DIR/labels/py3loader_new"
  cp -r "$SOURCE_REPO/py3loader_new" "$TARGET_DIR/labels/py3loader_new"
fi

if [[ "$DOWNLOAD_SUBTITLES" == "yes" ]]; then
  curl -L "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/makarand_tapaswi_iiit_ac_in/EY5UCcXRgAdJteMhG-vucwwB6vfsatQ-LobaUfEf1OK5Mw?e=b0iJf7" \
    -o "$TARGET_DIR/raw/moviegraphs_subtitles_download.bin"
fi

if [[ "$DOWNLOAD_FRAMES" == "yes" ]]; then
  curl -L "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/makarand_tapaswi_iiit_ac_in/EXLfOOllPXBArTmY_K9c4XABJ0CmOOBy88IA9W34L7aa4A?e=mRDZcC" \
    -o "$TARGET_DIR/raw/moviegraphs_1fps_download.bin"
fi

echo "[DONE] MovieGraphs assets prepared at: $TARGET_DIR"
echo "- split/meta files: $TARGET_DIR/splits"
echo "- graph annotations: $TARGET_DIR/labels/py3loader_new"
echo "- raw downloads: $TARGET_DIR/raw"

