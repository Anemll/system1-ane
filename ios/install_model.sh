#!/usr/bin/env bash
# Copy the Tetris build onto an iPhone: ios/install_model.sh <device-id> [models-dir]
# Needs the app installed once (Documents container exists) and scripts/download_model.py tetris done.
set -euo pipefail
DEV="${1:?device id or name (xcrun devicectl list devices)}"
SRC="${2:-${JEFF_MODELS_DIR:-$HOME/Models/jeff-ane}}/tetris"
BUNDLE="${BUNDLE_ID:?set BUNDLE_ID to your app's bundle identifier}"
for d in coreai model; do
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --source "$SRC/$d" --destination "Documents/tetris/$d" --remove-existing-content true
done
