# System 1 ANE for iPhone

A SwiftUI app that plays Tetris with a small Qwen-based decision model (Jeff + the Tetris adapter) running **entirely on the iPhone's Neural Engine** through Core AI. Each move is one prefill-only forward pass of a 24-layer hybrid model (about 0.2 to 0.5 s on an A20 iPhone), then a 255-way readout picks the placement. No server is involved.

It is the on-device port of the Mac demo in this repository (`coreai/`, `scripts/system1_ane_serve.py`): the Tetris rules, the El-Tetris oracle, the Jeff prompt and the Qwen tokenizer are Swift ports checked byte for byte against the Python reference (`gen_golden.py`, `System1Tetris/Tools/golden`).

## What it does

- Autoplay loop with Real-time (animated, 120 Hz) and Fast modes. The next decision runs on the ANE while the current piece falls.
- Shows the model's top options with probabilities, the El-Tetris label, latency, prompt size and the ANE call times, so you can see the model, not a rule-based player, is deciding.
- 20-game sessions (a game ends at 120 pieces or on top-out), best score kept across launches.
- Screen recording button (ReplayKit).
- Every finished game is saved on the device as encrypted JSON (decisions only: board, piece, the model's probabilities and choice, the El-Tetris label; no identifiers) and can be exported from the app (Saved games). `tools/games_to_rows.py` turns an export into adapter-kit rows, and `projects/tetris_training/relabel.py` relabels games with a stronger search teacher (see `docs/TETRIS_FLYWHEEL.md`). An **Explore** switch (ellipsis menu) samples moves from the model's own probabilities so the logs contain varied moves. **The app has no networking for game data.**

## Requirements

- iPhone with the Neural Engine and **iOS 27**, Xcode 27, [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).
- The model (about 1.4 GB) downloads inside the app on first launch from Hugging Face (`anemll/jeff-ane-tetris-coreai`). The first run then specializes the packages for your device's ANE, which takes a few minutes.

## Build and run

```sh
cd ios/System1Tetris
xcodegen generate
xcodebuild -project System1ANE.xcodeproj -scheme System1ANE \
  -destination 'id=<your device id>' -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=<your team id> build
```

Set your own bundle identifier (`PRODUCT_BUNDLE_IDENTIFIER` in `project.yml`, currently the placeholder `com.example.system1ane`) and team. Then install the product on your iPhone (or run from Xcode). To preload the model instead of using the in-app download, run `scripts/download_model.py tetris` on the Mac and `BUNDLE_ID=<your bundle id> ios/install_model.sh <device>`.

If the ANE compile fails on a new chip, relaunch with `-aneMode 1` (or `0`) to set `MPSGRAPH_ANE_BONDED_COMPILE_MODE`; see `docs/ANE_COMPILE_MODE_POLICY.md`.

## Tests

`ios/gen_golden.py` writes reference prompts and token ids from the Python tokenizer; `System1Tetris/Tools/golden` checks the Swift tokenizer, prompt and rules against them on the Mac:

```sh
python ios/gen_golden.py <models>/tetris/model ios/System1Tetris/Tests/golden.json
xcrun swiftc -O -o golden Sources/Tetris.swift Sources/Prompt.swift Sources/Tokenizer.swift Tools/golden/main.swift && ./golden <models>/tetris/model Tests/golden.json
```

## Notes

- Uses Core AI on iOS 27. ReplayKit is deprecated in iOS 27 but still works; ScreenCaptureKit is the replacement.
- Weights and builds are not in Git; see `docs/MODELS.md`.
