# Adapters

Jeff v1.3 is adapter-first: a fixed base plus a LoRA, its own readout, and a fitted temperature. This repo **merges** adapters into weights and converts each merge to its own Core AI package. The server selects a loaded package; it does not apply LoRA at serve time.

## Sample LoRAs (this repo)

| Name | Trainer | Oracle | Typical prompt |
| --- | --- | --- | --- |
| `snake` | `jeff-train-lora --task snake` | Shortest safe path to food | 8×8 board in `latest` (`H` head, `#` body, `F` food) |
| `tetris` | `jeff-train-lora --task tetris` | El-Tetris placement | Board in `latest`; options are legal rotation+column sentences |

Code: `coreai/jeff_snake.py`, `coreai/jeff_tetris.py`, `coreai/jeff_lora.py`, `scripts/jeff_lora_train.py`.

Eval against a running server:

```sh
python scripts/jeff_snake_eval.py --url http://127.0.0.1:8796 --games 8 \
  --adapter base --adapter snake
python scripts/jeff_tetris_eval.py --url http://127.0.0.1:8796
```

## Published PEFT (mstrasser, v1.3, Apache-2.0)

| Name | Hub | What it does |
| --- | --- | --- |
| `triage` | `mstrasser/jeff-adapter-triage` | Route a support message |
| `tools` | `mstrasser/jeff-adapter-tools` | Answer vs tool |
| `guard` | `mstrasser/jeff-adapter-guard` | Injection / benign |
| `spam` | `mstrasser/jeff-adapter-spam` | Spam / phishing / ham |

```sh
python scripts/jeff_peft_merge.py \
  --base "$HOME/Models/jeff/jeff-base-v1.3" \
  --adapter "$HOME/Models/jeff/adapters/jeff-adapter-triage" \
  --output "$HOME/Models/jeff-published/triage"
python forge.py jeff-convert \
  --model "$HOME/Models/jeff-published/triage" \
  --output "$HOME/Models/jeff-coreai/adapters/triage"
python forge.py compile --build "$HOME/Models/jeff-coreai/adapters/triage/coreai"
```

`jeff-convert` stores the fitted temperature under `manifest["convert"]["temperature"]`. `JeffCoreAI` applies it so a served adapter uses its own temperature while embeddings stay on `--model`.

Score the merge before convert: `scripts/jeff_peft_torch.py`. Compare compiled head vs that merge: `scripts/jeff_lora_parity.py`.

## Import any jeffhub adapter

The 13 official adapters on [jeffhub.ai](https://jeffhub.ai/#adapters) are PEFT LoRAs for Jeff base v1.3, published as `mstrasser/jeff-adapter-<name>` (revision `v1.3`). They go through the same merge, convert, compile, serve path as the four named above. Checked end to end with `support-intents` on an M5 Max (merge 1 s, convert 32 s, compile 1 m 10 s; argmax matched PyTorch on 31 of 31 rows, mean KL 9.5e-5; its published `example.json` served through `/v1/systemone`). The other 12 were not converted. Only their configs were read: all are LoRA rank 16, alpha 32, base `mstrasser/jeff-base`, base revision `2fc06364`, prompt layout `live-last`, with the same codes and token ids as our base.

| Adapter | Licence on HF | Note |
| --- | --- | --- |
| `aml` | Apache-2.0 | Includes IBM AMLworld pattern rows (CDLA-Sharing-1.0); the card has not confirmed whether that affects the adapter's licence |
| `emotion` | Apache-2.0 | GoEmotions |
| `ground` | Apache-2.0 | Part of the data was generated with hosted models |
| `guard` | Apache-2.0 | Part of the data was generated with hosted models |
| `legal-clauses` | Apache-2.0 | LEDGAR |
| `nav` | Apache-2.0 | Part of the data was generated with hosted models |
| `sanctions` | **CC BY-NC 4.0** | Non-commercial (OpenSanctions data). The jeffhub page still says "to be decided" |
| `soc` | **CC BY-NC 4.0** | Non-commercial; some sources forbid commercial use or redistribution. The card marks release as unconfirmed |
| `spam` | Apache-2.0 | |
| `support-intents` | Apache-2.0 | |
| `tools` | Apache-2.0 | |
| `trading-desk` | Apache-2.0 | |
| `triage` | Apache-2.0 | |

A converted build is a derivative of its adapter and inherits that licence. Do not redistribute converted `sanctions` or `soc` builds, and keep them out of commercial use. Read each adapter's "Data and licence" section before publishing anything built from it; several adapters' training data was generated with hosted models, so check those terms too. The Jeff-Code adapters (`code`, `code-router`) were not checked and may use a different base. Do not assume they load here.

Steps (the base is the v1.3 checkpoint, here `$BASE_MODEL`):

```sh
hf download mstrasser/jeff-adapter-support-intents --revision v1.3 --local-dir adapters/support-intents
# Check by hand: adapter_config.json base_model_name_or_path is mstrasser/jeff-base, and
# decision_config.json "revision" is 2fc06364715b967f1860aea9cf38778875588b17 (the base's own decision_config.json says the same).
python scripts/jeff_peft_merge.py --base "$BASE_MODEL" \
  --adapter adapters/support-intents --output merged/support-intents
python forge.py jeff-convert --model merged/support-intents --output out/support-intents
python forge.py compile --build out/support-intents/coreai
python forge.py system1-ane-serve --model "$BASE_MODEL" --build "$BASE_BUILD" \
  --adapter support-intents=out/support-intents/coreai --host 127.0.0.1 --port 8796
curl -s http://127.0.0.1:8796/v1/systemone -H 'content-type: application/json' \
  -d @adapters/support-intents/example.json
```

`jeff_peft_merge.py` does not verify the base revision. Check it yourself as above; a v1.2 adapter or a different base has the same tensor shapes, so this script would merge it without an error although the result would be wrong (upstream jeff-serve refuses such an adapter). `$BASE_BUILD` is the base's compiled `coreai/` directory.

Check the result. `scripts/jeff_peft_torch.py` scores the merged checkpoint on the adapter's `example.json` plus rows from its `test.jsonl`. `scripts/jeff_lora_parity.py` (Core AI interpreter) compares those probabilities with the compiled build. `aml`, `sanctions`, `soc` and `trading-desk` ship no `test.jsonl`, and `jeff_peft_torch.py` reads that file, so it does not run on them as written.

```sh
python scripts/jeff_peft_torch.py --adapter support-intents=adapters/support-intents \
  --merged merged --output parity --rows 30     # merged/<name> and the adapter dir share the name
python scripts/jeff_lora_parity.py --model "$BASE_MODEL" \
  --build out/support-intents/coreai --rows parity/support-intents/parity_rows.json
```

Prompt and API compatibility:
- The server renders the v1.3 `live-last` layout, which is what these adapters were trained on. The adapter's fitted temperature goes into the build manifest and is applied at serve time.
- The request body is the same as upstream Jeff's, so an adapter's `example.json` posts unchanged. `"model": "<name>"` or `"adapter": "<name>"` selects a loaded build. The only change from upstream examples is the port (8796 here, 8765 in theirs).
- The set of builds is fixed when the server starts (`--adapter name=<build>/coreai`, repeatable). There is no equivalent of upstream's `/v1/adapters/reload`; restart the server to add or replace one.

Limits:
- Prompts over 2048 tokens do not fit the ANE build (`SYSTEM1_ANE_MAX_TOKENS`, 422). Expect that for long `tools`, `ground` and `legal-clauses` rows.
- Each converted adapter is about 955 MB (`coreai/`) and is its own package; adapters do not share weights on the ANE. Its `model/` directory (about 505 MB, mostly a copy of the base embeddings) is not needed to serve, because embeddings come from `--model`.
- Serve a handful at a time. Many resident builds can exhaust IOSurface memory (`NDArray` allocation failure; see "Serving several builds"). Two engines loaded without trouble; the full set of 13 resident at once was not tried and is unlikely to work.

Troubleshooting: if `compile` or `serve` fails with `OSError: dlopen(.../coreai/swift_bridge/libcoreai_bridge.dylib) ... no such file`, the native bridge has not been built in that checkout. It is not stored in Git. Build it once with `bash coreai/swift_bridge/build.sh` (runs `xcrun swiftc`, so Xcode or its command line tools must be installed), or point `COREAI_BRIDGE_LIB` (or `ANEMLL_CORE_AI_BRIDGE_LIB`) at an already built dylib.

## Serving several builds

The Tetris build below is the October 8 round-02 model (or `$JEFF_MODELS_DIR/tetris` from `scripts/download_model.py`). The older `jeff-coreai/adapters/tetris` export is superseded.

```sh
python forge.py system1-ane-serve \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --build "$HOME/Models/jeff-coreai/coreai" \
  --adapter snake="$HOME/Models/jeff-coreai/adapters/snake/coreai" \
  --adapter tetris="${JEFF_TETRIS_BUILD:-$HOME/Models/jeff-tetris-training/overnight-20261008/round-02/build}/coreai" \
  --adapter triage="$HOME/Models/jeff-coreai/adapters/triage/coreai" \
  --adapter tools="$HOME/Models/jeff-coreai/adapters/tools/coreai" \
  --adapter guard="$HOME/Models/jeff-coreai/adapters/guard/coreai" \
  --adapter spam="$HOME/Models/jeff-coreai/adapters/spam/coreai" \
  --host 127.0.0.1 --port 8796
```

Request field `"adapter": "snake"` or `"model": "snake"` selects a loaded build. Omitting both stays on `base`. `GET /health` lists `adapters`. The demo dropdown is filled from that list.

Loading six compiled builds at once can exhaust IOSurface during long Snake autoplay (`NDArray` allocation failure). A single decision on each adapter is fine. Prefer two or three resident engines for games.

## What is not in this tree

- **Runtime streamed LoRA** (ANE inputs, no recompile) lives on branch `cursor/jeff-lora-stream` (anemll-forge PR #10). This repo's serve path is merge-then-convert.
- **Prefix-cache policy** (live-last reuse) lives on `cursor/jeff-prefix-cache` (PR #7). The snapshot/resume **hook** is in `jeff_prefix.py` and the server; the cache object is not installed.
- The older checkpoint `Jeff-Qwen3.5-0.8B` (state-first prompt) is not a dropdown entry: this server always renders v1.3 live-last.
