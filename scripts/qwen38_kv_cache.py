"""KV representation metadata and accepted-row writes (no model imports).

Origin: Anemll/anemll-forge. Jeff uses ``put_rows`` to commit each prefill call's
new key/value rows. The V8/KV8 helpers remain for the shared inspect/compile path.
"""

# cache inputs per attention layer: v8 = FP16 keys + INT8 values; kv8 = INT8 keys and values (scales per token/head)
KV_INPUTS = {"fp16": ("k", "v"), "v8": ("k", "v", "vs"), "kv8": ("k", "v", "ks", "vs")}
INT8_LAYOUTS = {"v8": ("float16", "int8"), "kv8": ("int8", "int8")}  # (keys, values)
# key cache layouts: token_dim (KV head, token, head dim), the default; dim_token (KV head, head dim, token), the
# operand QK reads, so the program transposes no key tile (builder KV_KEYS_T=1). Values are always token_dim.
KEY_LAYOUTS = ("token_dim", "dim_token")


def cache_formats(manifest):
    """Validate cache layouts without importing model or array dependencies."""
    metadata = manifest.get("kv_cache", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid kv_cache metadata")
    actual = metadata.get("format", "fp16")
    if actual == "selectable":
        layouts = metadata.get("formats")
        if not isinstance(layouts, dict) or set(layouts) != {"fp16", "v8"}:
            raise ValueError("Selectable KV exports must contain fp16 and v8 layouts")
        for name, layout in layouts.items():
            if not isinstance(layout, dict) or layout.get("format") != name:
                raise ValueError("Invalid selectable KV layout")
            cache_formats({"kv_cache": layout})
            if name == "fp16" and (layout.get("keys") != "float16" or layout.get("values") != "float16"):
                raise ValueError("FP16 layout requires FP16 keys and values")
        if metadata.get("default") not in layouts:
            raise ValueError("Invalid default KV format")
        for chunk in manifest.get("chunks", []):
            for name in layouts:
                cache_entries(manifest, chunk, name)
        return ("fp16", "v8")
    if actual not in KV_INPUTS:
        raise ValueError("KV cache format must be auto, fp16, v8 or kv8")
    if actual in INT8_LAYOUTS and ((metadata.get("keys"), metadata.get("values")) != INT8_LAYOUTS[actual]
                                   or metadata.get("scales") != "float16"
                                   or metadata.get("scale_granularity") != "token_head"):
        raise ValueError("V8 requires FP16 keys and INT8 values, KV8 INT8 keys and values, both with FP16 scales "
                         "per token/head")
    return (actual,)


def cache_entries(manifest, chunk, mode):
    """Map runtime's canonical names onto a selected format's physical functions."""
    if manifest.get("kv_cache", {}).get("format") != "selectable":
        return {name: name for name in chunk["entries"]}
    formats = chunk.get("entries_by_kv", {})
    aliases = formats.get(mode) if isinstance(formats, dict) else None
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError(f"Missing {mode} KV entry map for {chunk.get('file', 'chunk')}")
    required = {f"v8_{c // 1024}k" for c in manifest.get("ctxs", [])}
    if manifest.get("TP", 0):
        required.update(f"p64_{c // 1024}k" for c in manifest.get("pctxs", []))
    if not required.issubset(aliases):
        raise ValueError(f"Incomplete {mode} KV entry map for {chunk.get('file', 'chunk')}")
    if any(not isinstance(k, str) or not isinstance(v, str) or v not in chunk["entries"]
           for k, v in aliases.items()):
        raise ValueError(f"Invalid {mode} KV physical entry map")
    return dict(aliases)


def cache_format(manifest, requested="auto"):
    """Legacy manifests are FP16; selectable exports share weights for both modes."""
    formats = cache_formats(manifest)
    if requested != "auto" and requested not in KV_INPUTS:
        raise ValueError("KV cache format must be auto, fp16, v8 or kv8")
    actual = manifest.get("kv_cache", {}).get("default", formats[0]) if len(formats) > 1 else formats[0]
    if requested != "auto" and requested not in formats:
        raise ValueError(f"Requested {requested} KV cache, but this bundle contains {actual}. "
                         "Select a matching --build; cache compression requires different ANE entry points.")
    return actual if requested == "auto" else requested


def key_layout(manifest):
    """The key cache layout of a build (KEY_LAYOUTS; manifests without one are token_dim), checked against the chunks'
    KV_KEYS_T numerics so a runtime never writes keys in the other layout."""
    metadata = manifest.get("kv_cache", {})
    layouts = list(metadata.get("formats", {}).values()) if metadata.get("format") == "selectable" else [metadata]
    found = {layout.get("key_layout", "token_dim") for layout in layouts}
    if len(found) != 1 or not found <= set(KEY_LAYOUTS):
        raise ValueError(f"Invalid kv_cache key_layout {sorted(found)}")
    (layout,) = found
    want = "dim_token" if any((c.get("numerics") or {}).get("KV_KEYS_T") for c in manifest.get("chunks", [])) \
        else "token_dim"
    if layout != want:
        raise ValueError(f"kv_cache key_layout {layout} does not match the chunks (KV_KEYS_T: {want})")
    return layout


def put_rows(cache, rows, position, count, transposed=False):
    """rows (KV head, count, last) into cache positions [position, position + count): along axis 1, or along axis 2
    for a dim_token key cache (KV head, head dim, token)."""
    if transposed:
        cache[:, :, position:position + count] = rows.transpose(0, 2, 1)
    else:
        cache[:, position:position + count] = rows


def keep_rows(dst, src, keep, transposed=False):
    """Positions [0, keep) of src into dst (a context resize)."""
    if transposed:
        dst[:, :, :keep] = src[:, :, :keep]
    else:
        dst[:, :keep] = src[:, :keep]


def quantize_values(values):
    """Symmetric signed INT8; round using the scale that is actually stored."""
    import numpy as np
    values = np.asarray(values, dtype=np.float16).astype(np.float32)
    scales = np.maximum(np.max(np.abs(values), axis=-1) / 127, 1e-6).astype(np.float16)
    codes = np.clip(np.rint(values / scales.astype(np.float32)[..., None]), -127, 127).astype(np.int8)
    return codes, scales


def append_rows(kv, outputs, attention_indices, position, count, mode, keys_t=False):
    """Write accepted rows only. New rows arrive as FP16 (KV head, rows, head dim); v8 stores values and kv8 keys and
    values as INT8 codes with FP16 scales per token/head (quantize_values); keys_t: a dim_token key cache."""
    if not count:
        return
    end = position + count
    for j in attention_indices:
        keys = outputs[f"k{j}_new"][:, :count]
        values = outputs[f"v{j}_new"][:, :count]
        if mode == "kv8":
            codes, scales = quantize_values(keys)
            put_rows(kv[f"k{j}"][1], codes, position, count, keys_t)
            kv[f"ks{j}"][1][:, position:end] = scales
        else:
            put_rows(kv[f"k{j}"][1], keys, position, count, keys_t)
        if mode in INT8_LAYOUTS:
            codes, scales = quantize_values(values)
            kv[f"v{j}"][1][:, position:end] = codes
            kv[f"vs{j}"][1][:, position:end] = scales
        else:
            kv[f"v{j}"][1][:, position:end] = values
