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
- snake
- lora
---

# Jeff Snake — Core AI build for the Apple Neural Engine

**Unofficial** Core AI conversion of a Snake-playing [Jeff](https://huggingface.co/mstrasser/jeff-base) decision model
(Jeff v1.3, a Qwen3.5-0.8B fine-tune) by [Anemll](https://github.com/Anemll/system1-ane). Each move is one decision, about 65 ms on an M5 Max. The packages run fully on the **Apple Neural Engine** (FP16, Core AI, `p256_2k`).

**This is a demo, not a strong player**: about 0.25 food per game in the 8-game self-play check (base: 0.12).

**Source / code:** [github.com/Anemll/system1-ane](https://github.com/Anemll/system1-ane) (MIT): runtime, demo, training and conversion code.

## What's here

`config.json` (package descriptor, not a transformers config), `coreai/` (six chunk packages + `head_readout`, `manifest.json`), `model/` (tokenizer, `decision_config.json`, readout, FP16 embeddings, transformers `config.json`), `LICENSE`, `NOTICE`. About 1.5 GB, exported not compiled.

## Use

```sh
git clone https://github.com/Anemll/system1-ane.git && cd system1-ane
python scripts/download_model.py snake base
export COREAI_PYTHON=/path/to/coreai-sdk/bin/python
python forge.py compile --build ~/Models/jeff-ane/base/coreai
python forge.py compile --build ~/Models/jeff-ane/snake/coreai
python forge.py system1-ane-serve --model ~/Models/jeff-ane/base/model --build ~/Models/jeff-ane/base/coreai \
  --adapter snake=$HOME/Models/jeff-ane/snake/coreai --port 8796
```

Serve it as an adapter next to [`anemll/jeff-ane-base-coreai`](https://huggingface.co/anemll/jeff-ane-base-coreai), which carries the full checkpoint `system1-ane-serve --model` needs.

## Training

Rank-16 LoRA plus the readout head, trained **locally on an Apple M5 Max with PyTorch MPS** (2 epochs, about 5 minutes) using this project's own LoRA code, on Snake states generated and labelled by the repo's search oracle. No third-party dataset; no Unsloth or Colab weights are shipped.

## Results (Apple M5 Max)

Held-out move accuracy (64 rows): base 0.281 → adapter 0.594 (MPS), 38/64 on the ANE build (matches PyTorch). Self-play (8 games): 0.25 food and 3.5 steps per game. All packages `fully_ane`. Details: [docs/RESULTS.md](https://github.com/Anemll/system1-ane/blob/main/docs/RESULTS.md).

## Credits

- **Jeff v1.3** by [mstrasser](https://huggingface.co/mstrasser): [mstrasser/jeff-base](https://huggingface.co/mstrasser/jeff-base) (Apache-2.0).
- **Qwen3.5-0.8B** by the Qwen Team, Alibaba Cloud: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).
- Conversion to Core AI and the ANE runtime/demo: [Anemll](https://github.com/Anemll/system1-ane). Unofficial: not affiliated with or endorsed by Alibaba Cloud or the Jeff author.

## License

Apache License 2.0 ([`LICENSE`](LICENSE), attribution in [`NOTICE`](NOTICE)), the same license as Qwen3.5-0.8B and jeff-base v1.3, which these weights derive from. The demo and conversion code in [Anemll/system1-ane](https://github.com/Anemll/system1-ane) is separate software under the MIT License.
