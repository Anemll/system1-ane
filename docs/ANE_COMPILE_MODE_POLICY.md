# Spec: ANE bonded compile mode policy by SoC generation

Jeff compile and `JeffCoreAI` apply this policy before loading packages. On the M5 Max (the chip Jeff was measured on) the effective mode is **1**. Origin: Anemll/anemll-forge; the 27B `serve` / `quick-test` names below are historical. `docs/SESSION_LESSONS.md`, `docs/SERVER.md`, `docs/VALIDATION.md`, `RESULTS_M5_MAX.md`, and `qwen38_server.*` were not copied into this Jeff tree.

**Status:** Implemented 2026-10-04. Selected defaults: an explicit mode override is honored and logged; pre-M5 fails on model-loading paths (`system1-ane-serve` / `compile`); `doctor` warns only. See §5 for the decisions.

## 1. Problem

`MPSGRAPH_ANE_BONDED_COMPILE_MODE` selects which MPSGraph ANew procedure variant is
specialized: `0` keeps bonded + non-bonded, `2` keeps bonded only (the Core AI default),
and `1` is the value that made the full target/drafter load succeed on an Apple M5.

Observed on **Apple M5 (Mac17,2, macOS 26B5091g, "H17")** on 2026-10-04:

- With the runtime default `2`, the 27B `forge.py serve` path failed to compile the
  target/drafter packages with `_ANECompiler : ANECCompile() FAILED … "Compiler internal
  error: Couldn't do topological sort"`.
- With `MPSGRAPH_ANE_BONDED_COMPILE_MODE=1` exported, that 27B serve proceeded to
  compile all 18 packages and start serving. Reproduced by the maintainer.
- An earlier M5 diagnostic (anemll-forge `docs/SESSION_LESSONS.md`, September 29, not
  copied here) showed mode `0` also fails (32 topological-sort failures) and consumes
  disk faster than mode `2`. Jeff on this tree uses the same helper via `system1-ane-serve` /
  `compile`.

Therefore the compile-mode default that is correct for M6 is not correct for M5, and
running on pre-M5 silicon is not a supported target.

## 2. Evidence summary

| SoC | ANE behavior | `soc-generation` | Source |
|---|---|---|---|
| **M5 base (H17)** | `2` fails quick-test/serve compile (topological sort); `1` succeeds; `0` fails and costs more disk | `"H17"` | maintainer run 2026-10-04; anemll-forge session notes (Sept 29) |
| **M6 family (H18)** | `2` is the measured default; parity with `0`, lower program memory, release measurements use `2` | `"H18"` | confirmed by maintainer (`ioreg`); anemll-forge M6 research notes (not copied) |
| **Newer than M6 (H19+)** | Unmeasured; defaults to the M6 policy (`2`) | `>= "H19"` | maintainer directive 2026-10-04 |
| **M5 Pro / M5 Max (H17)** | `1` required (maintainer); system1-ane-serve confirmed on M5 Max | `"H17"` | maintainer directive 2026-10-04; Jeff RESULTS.md (7 Oct 2026) |
| **pre-M5 (H16 and below)** | Not a supported target; no release measurements | `< "H17"` | README (M5/M6 target) |

Mode `1` is undocumented by Apple. Its semantics are not published; it is treated here as
an empirically required switch on H17, not as a general recommendation.

## 3. Detection

The policy is keyed on **SoC generation**, not marketing name. Detection order:

1. **Primary — Core AI architecture code** from the loaded package/device descriptor where
   available (`h17*`, `h18*`); `COREAI_ARCH` already overrides the build arch
   (`coreai/qwen38_coreai_build.py`).
2. **Fallback — platform generation**, in this order:
   - `ioreg` `soc-generation` property (`"H17"`, `"H18"`), or
   - `sysctl -n machdep.cpu.brand_string` → parse `Apple M<generation>` (observed:
     `Apple M5`), or
   - `sysctl -n hw.model` model class (`Mac17,*` ≈ M5 family, `Mac18,*` ≈ M6 family;
     observed M6 = `Mac18,5`, M5 Max = `Mac17,6`).

Generation order used for decisions: `H17 < H18`, and anything `< H17` is pre-M5.

Detection MUST distinguish the M5 sub-variant, because only the base M5 has been
observed:

- **M5 base:** `Apple M5` with no `Pro`/`Max` suffix; observed `Mac17,2`.
- **M5 Pro / M5 Max:** `Apple M5 Pro` / `Apple M5 Max`; M5 Max observed as `Mac17,6`
  with ANE target `h17c`.

Detection MUST be centralized in one helper returning a normalized class
(`pre_m5 | m5 | m5_pro_max | m6 | unknown`) plus the raw source string, so the error and
logs are consistent across `forge.py`, `coreai_compile.py`, and the wrapper.

## 4. Requirements

Normative language: **MUST**, **SHOULD**, **MAY**.

- **R1 (M5 family):** On any M5-family SoC (`H17`, including base M5, M5 Pro and M5 Max),
  the runtime and compile paths **MUST** set the effective
  `MPSGRAPH_ANE_BONDED_COMPILE_MODE` to `1` unless the user explicitly overrides it
  (see R5). Base M5 confirmed on 2026-10-04; M5 Pro/Max are a maintainer directive
  (see §9).
- **R2 (pre-M5):** On a SoC older than M5 (`< H17`), the runtime and compile paths **MUST
  fail fast before loading or compiling any package**, with a clear, actionable error that
  names the detected generation and the supported set. They **MUST NOT** attempt to compile
  or serve, even when an explicit mode override is set.
- **R3 (M6):** On M6-family (`H18`), the existing measured default `2` **MUST** be
  preserved.
- **R3b (newer than M6):** On a detected generation `H19` or later, the runtime **MUST**
  default to the M6 settings (bonded mode `2`). It **MUST** log that the generation is
  unvalidated and that M6 settings are being assumed, and an explicit override **MUST** be
  honored.
- **R4 (cache correctness):** Changing the effective mode **MUST** leave the existing
  cross-mode cache handling intact: cached specializations compiled in another mode are
  purged and recompiled (`scripts/qwen38_coreai_model.py`, `_align_mode`). A policy-driven
  mode change therefore starts a cold compile exactly once.
- **R5 (explicit override):** A user-provided `MPSGRAPH_ANE_BONDED_COMPILE_MODE` in the
  environment **MUST** be honored on supported generations (`M5`, `M6`, `H19+`) and on an
  undetectable chip, because every setter uses `setdefault`. It **MUST NOT** re-enable a
  pre-M5 SoC (R2). The startup log and `/health` **MUST** report both the detected
  generation and the effective mode so an override is never silent.
- **R6 (diagnostics):** Startup **MUST** print one line identifying the generation source
  and the effective mode, e.g. `ANE compile mode policy: M5 (H17) -> bonded mode 1`
  (and `override by env` when applicable). `GET /health` **SHOULD** expose
  `soc_generation` and `ane_bonded_compile_mode`.
- **R7 (undetectable generation):** When the generation cannot be parsed at all (not a
  recognized `H<number>`/brand/model), the runtime **MUST** fail closed by default unless
  the user explicitly overrides the mode; `COREAI_ALLOW_UNKNOWN_SOC=1` **MAY** be provided
  as an explicit escape hatch. A *detected but newer* generation (`H19+`) is **not**
  unknown and follows R3b.

## 5. Decisions and remaining questions

Resolved during implementation:

1. ~~Does “M5*” include M6?~~ **Resolved:** M6 (`H18`) defaults to `2`; `H19+` also uses M6 settings.
2. ~~M5 Pro / M5 Max default.~~ **Resolved:** all `H17` (base M5, M5 Pro, M5 Max) → `1`.
3. ~~Override allowed?~~ **Resolved:** an explicit `MPSGRAPH_ANE_BONDED_COMPILE_MODE` is honored and logged.
4. ~~Fail scope.~~ **Resolved:** fail on model-loading commands; warn-only on `doctor`.

Resolved after the Jeff copy:

5. **M5 Max reproduction.** Jeff `p256_2k` FP16 convert/compile/serve ran on an M5 Max
   (7 October 2026). Mode 1. See [RESULTS.md](RESULTS.md).
6. **Wrapper.** `qwen38_server.sh` is not in this tree. `forge.py system1-ane-serve` / `compile`
   apply the policy in Python.

## 6. Acceptance criteria

- Unit tests with a mocked detector cover `pre_m5`, `m5`, `m5_pro_max`, `m6`, `unknown`:
  - `m5` → effective mode `1`;
  - `m5_pro_max` → effective mode `1`, override honored;
  - `m6` → effective mode `2`;
  - `h19+` (newer) → effective mode `2` + unvalidated-generation warning, override honored;
  - `pre_m5` → fails before any package load, non-zero exit, message includes generation;
  - undetectable `unknown` → fails unless `COREAI_ALLOW_UNKNOWN_SOC=1`.
- A user env override is honored on `m5`/`m6` and reported in the startup line and `/health`.
- Switching generation/mode purges stale cache entries (`_align_mode`) and recompiles once.
- Docs: `README.md`, [SYSTEM1_ANE_SERVE.md](SYSTEM1_ANE_SERVE.md) (`/health` fields), this spec.

## 7. Implementation touch points

| File | Role |
|---|---|
| `scripts/ane_compile_mode.py` | SoC → bonded mode; fail pre-M5 |
| `scripts/qwen38_coreai_model.py` (`MODE_ENV`, `_align_mode`) | purge cache compiled in another mode |
| `forge.py` (`compile`, `system1-ane-serve`, `doctor`) | call helper; fail fast; report mode |
| `scripts/coreai_compile.py` | same policy before compiling |
| `coreai/jeff_coreai_runtime.py` | apply policy before loading packages |
| `tests/test_ane_compile_mode.py` | mocked detector |

## 8. Non-goals

- Re-deriving Apple's bonded/non-bonded semantics or proving physical ANE placement.
- Claiming a throughput or quality gain from mode `1`; the requirement is only that M5
  targets compile and load successfully.
- Changing the M6 release defaults or measured results.

## 9. Validation status

This spec is based on: one successful **base M5** mode-`1` full load (2026-10-04,
maintainer), the September 29 M5 diagnostic record, and existing M6 research.

- Mode `1` has **not** been quality- or throughput-validated on M5 base.
- **M5 Pro / M5 Max:** Jeff convert/compile/serve on an M5 Max is recorded in
  [RESULTS.md](RESULTS.md) (7 October 2026). The older 27B vector-LUT notes
  (`RESULTS_M5_MAX.md`) were not copied into this tree.
- M6 measurements remain on mode `2` and are unchanged.
- **Generations newer than M6 (`H19+`)** are unmeasured and inherit M6 settings by
  directive; this is not a validation of those chips.

Treat the policy as a raw requirement pending implementation and a reproduction pass.
