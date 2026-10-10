#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Uses xcode-select, or the caller's DEVELOPER_DIR. Compiler failures propagate.
xcrun swiftc -O -swift-version 5 -emit-library -module-name CoreAIBridge \
  -Xlinker -install_name -Xlinker @rpath/libcoreai_bridge.dylib \
  CoreAIBridge.swift -o libcoreai_bridge.dylib
nm -gU libcoreai_bridge.dylib
