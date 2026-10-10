---
license: apache-2.0
base_model: mstrasser/jeff-base
base_model_relation: finetune
library_name: coreai
pipeline_tag: zero-shot-classification
language:
- en
tags:
- coreai
- apple-neural-engine
- ane
- jeff
- decision-model
- anemll
---

# Jeff v1.3 base — Core AI build for the Apple Neural Engine

**Unofficial** Core AI conversion of [mstrasser/jeff-base](https://huggingface.co/mstrasser/jeff-base) (Jeff v1.3, a Qwen3.5-0.8B fine-tune) by [Anemll](https://github.com/Anemll/system1-ane). No retraining: the weights are Jeff's, converted to six 4-layer `.aimodel` chunks plus the 255-way readout head, all `fully_ane` on an M5 Max. You send a situation and named options; one forward pass returns a calibrated probability per option (about 65 ms for a short decision).

## What's here

| Path | What |
| --- | --- |
| `config.json` | Package descriptor (not a transformers config) |
| `coreai/` | Core AI packages and `manifest.json` (`p256_2k`, FP16, context 2048) |
| `model/` | Tokenizer, `decision_config.json`, readout, FP16 embeddings, transformers `config.json` and `model.safetensors` (the unmodified jeff-base v1.3 checkpoint that `system1-ane-serve --model` loads) |
| `LICENSE`, `NOTICE` | Apache-2.0 and attribution |

About 3.2 GB, exported not compiled. This is the server's base build; the Snake and Tetris builds load beside it with `--adapter` ([snake](https://huggingface.co/anemll/jeff-ane-snake-coreai), [tetris](https://huggingface.co/anemll/jeff-ane-tetris-coreai)).

## Use

```sh
git clone https://github.com/Anemll/system1-ane.git && cd system1-ane
python scripts/download_model.py base
export COREAI_PYTHON=/path/to/coreai-sdk/bin/python
python forge.py compile --build ~/Models/jeff-ane/base/coreai
python forge.py system1-ane-serve --model ~/Models/jeff-ane/base/model --build ~/Models/jeff-ane/base/coreai --port 8787
# open http://127.0.0.1:8787/ (routing panel)
```

Jeff v1.3 is meant to be used with task adapters; on its own it is weak on unfamiliar tasks with long option lists (see the jeff-base card).

## Results (Apple M5 Max, macOS 27.2)

Prefill parity vs PyTorch FP32 on real Jeff prompts: KL ≤ 8.3e-4, same argmax except one near-tie at 2,018 tokens. About 64 ms per short decision (one 256-row call). Details: [docs/RESULTS.md](https://github.com/Anemll/system1-ane/blob/main/docs/RESULTS.md).

## Credits

- **Jeff v1.3** by [mstrasser](https://huggingface.co/mstrasser): [mstrasser/jeff-base](https://huggingface.co/mstrasser/jeff-base) (Apache-2.0).
- **Qwen3.5-0.8B** by the Qwen Team, Alibaba Cloud: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).
- Conversion to Core AI and the ANE runtime/demo: [Anemll](https://github.com/Anemll/system1-ane). Unofficial: not affiliated with or endorsed by Alibaba Cloud or the Jeff author.

## License

Apache License 2.0 ([`LICENSE`](LICENSE), attribution in [`NOTICE`](NOTICE)), the same license as Qwen3.5-0.8B and jeff-base v1.3, which these weights derive from. The demo and conversion code in [Anemll/system1-ane](https://github.com/Anemll/system1-ane) is separate software under the MIT License.
