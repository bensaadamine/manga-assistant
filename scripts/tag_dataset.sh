#!/usr/bin/env bash
# Replace the placeholder captions written by manga.preprocess with real tags.
# WD14 is the standard anime/manga tagger.
#
#   bash scripts/tag_dataset.sh data/datasets/conan aoymstyle
set -euo pipefail

DATA_DIR="${1:?usage: tag_dataset.sh <data_dir> <trigger_word>}"
TRIGGER="${2:?usage: tag_dataset.sh <data_dir> <trigger_word>}"
SCRIPTS="${SD_SCRIPTS:-./sd-scripts}"

python "$SCRIPTS/finetune/tag_images_by_wd14_tagger.py" \
  "$DATA_DIR" \
  --repo_id SmilingWolf/wd-v1-4-convnextv2-tagger-v2 \
  --batch_size 4 \
  --caption_extension .txt \
  --general_threshold 0.35 \
  --remove_underscore

# prepend the trigger word to every caption
for f in "$DATA_DIR"/*.txt; do
  printf '%s, %s' "$TRIGGER" "$(cat "$f")" > "$f.tmp" && mv "$f.tmp" "$f"
done

echo "tagged $(ls "$DATA_DIR"/*.txt | wc -l) captions with trigger '$TRIGGER'"
