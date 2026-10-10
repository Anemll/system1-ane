#!/usr/bin/env python3
"""Cast the Unsloth Clef JointSchemaHead (joint_head.safetensors, bf16 + 3 fp32 scalars) to fp16.
usage: convert_clef_head_fp16.py <clef_head dir>   -> writes joint_head_fp16.safetensors + joint_head_fp16_report.json (original untouched)
bf16 -> fp16 is value-exact for normal-range values (fp16 has 3 more mantissa bits); only |w|<6.1e-5 become subnormal/flush.
The 3 scalar logit scales stay fp32 (they are applied via exp/sigmoid, 12 bytes)."""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
d = Path(sys.argv[1]); sd = load_file(str(d / "joint_head.safetensors")); out = {}; rep = {"tensors": len(sd)}
mx = 0.0; lossy = 0; tiny = 0; total = 0; maxerr = 0.0
SCAL = {"prior_logit_scale", "joint_logit_scale", "residual_gate"}
for k, v in sd.items():
    if k in SCAL:
        out[k] = v.float().contiguous(); continue
    h = v.to(torch.float16)
    mx = max(mx, v.float().abs().max().item())
    back = h.float(); err = (back - v.float()).abs().max().item(); maxerr = max(maxerr, err)
    nz = v.float() != 0; tiny += int(((v.float().abs() < 6.1e-5) & nz).sum()); total += v.numel()
    assert torch.isfinite(h).all(), k
    out[k] = h.contiguous()
save_file(out, str(d / "joint_head_fp16.safetensors"))
rep.update(max_abs=mx, max_cast_error=maxerr, nonzero_below_fp16_normal=tiny, params=total,
           fp32_kept=sorted(SCAL), dtypes=sorted({str(v.dtype) for v in out.values()}))
(d / "joint_head_fp16_report.json").write_text(json.dumps(rep, indent=1)); print(rep)
