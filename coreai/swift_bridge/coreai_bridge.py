"""ctypes wrapper of libcoreai_bridge.dylib (CoreAIBridge.swift).

Origin: Anemll/anemll-forge. ``forge.py compile`` loads packages through this bridge
(``COREAI_BRIDGE_LIB`` / ``ANEMLL_CORE_AI_BRIDGE_LIB``). system1-ane-serve uses ``coreai.runtime``
for decisions; the dylib is still required to specialize packages.

ctypes wrapper of libcoreai_bridge.dylib (CoreAIBridge.swift): the native Core AI Swift runtime from Python, with
IOSurface buffers that numpy views zero-copy and that the runtime reads / writes in place (outputViews), so steady-state
calls allocate nothing (the Python binding allocates every output per call, which exhausts its IOSurface pool).

    import coreai_bridge as B
    m = B.Model("chunk.aimodel")                  # compute="ane" (neural engine preferred, as qwen38_coreai_model)
    f = m.function("v8_16k")
    ins = {n: f.buffer("input", n) for n in f.input_names}     # preferred (model) layout, zero-filled
    outs = {n: f.buffer("output", n) for n in f.output_names}
    ins["x"].np[...] = ...                        # numpy view of the IOSurface (strided like the model wants)
    b = f.bind(ins, outs)                         # validated once; reuse every call
    B.run([b])                                    # blocks; runs in a .userInitiated Swift task; raises BridgeError
    y = outs["y"].np                              # written in place
Several bindings run back to back in one call (B.run([b0, b1, ...]) or a prebuilt B.Plan) - e.g. all chunks of a model,
with chunk k's output buffer bound as chunk k+1's input.
Env: COREAI_BRIDGE_LIB (path of the dylib; default next to this file)."""
from __future__ import annotations

import ctypes
import json
import os
import threading
from pathlib import Path

import numpy as np

LIB_PATH = Path(os.environ.get("COREAI_BRIDGE_LIB") or os.environ.get("ANEMLL_CORE_AI_BRIDGE_LIB")
                or Path(__file__).resolve().with_name("libcoreai_bridge.dylib"))
DTYPES = {"float16": (0, np.float16), "float32": (1, np.float32), "int32": (2, np.int32), "int8": (3, np.int8),
          "uint8": (4, np.uint8), "int16": (6, np.int16), "uint16": (7, np.uint16), "int64": (8, np.int64),
          "float64": (9, np.float64), "bool": (10, np.bool_)}
COMPUTE = {"ane": 0, "neural_engine": 0, "gpu": 1, "cpu": 2, "default": 3}
_c = ctypes
_P, _S, _I, _D = _c.c_void_p, _c.c_char_p, _c.c_int32, _c.c_double
_lib = None
_tls = threading.local()


class BridgeError(RuntimeError):
    pass


def available() -> bool:
    return LIB_PATH.exists()


def lib():
    global _lib
    if _lib is None:
        L = _c.CDLL(str(LIB_PATH))  # CDLL releases the GIL during calls
        sig = {
            "cai_abi_version": ([], _I),
            "cai_release": ([_P], None),
            "cai_os_mpsgraph_version": ([_c.c_char_p, _I], _I),
            "cai_package_mpsgraph_version": ([_S, _c.c_char_p, _I], _I),
            "cai_model_load": ([_S, _I, _I, _c.c_char_p, _I], _P),
            "cai_model_function_names": ([_P, _c.c_char_p, _I], _I),
            "cai_function_load": ([_P, _S, _c.c_char_p, _I], _P),
            "cai_function_describe": ([_P, _c.c_char_p, _I], _I),
            "cai_buffer_create": ([_I, _I, _c.POINTER(_c.c_int64), _c.POINTER(_c.c_int64), _c.c_char_p, _I], _P),
            "cai_buffer_address": ([_P], _P),
            "cai_buffer_nbytes": ([_P], _c.c_int64),
            "cai_binding_create": ([_P, _I, _c.POINTER(_S), _c.POINTER(_P), _I, _c.POINTER(_S), _c.POINTER(_P),
                                    _I, _c.POINTER(_S), _c.POINTER(_P), _c.c_char_p, _I], _P),
            "cai_run": ([_I, _c.POINTER(_P), _c.POINTER(_D), _c.c_char_p, _I], _I),
        }
        for n, (a, r) in sig.items():
            fn = getattr(L, n)
            fn.argtypes, fn.restype = a, r
        if L.cai_abi_version() != 1:
            raise BridgeError(f"{LIB_PATH}: ABI {L.cai_abi_version()} != 1")
        _lib = L
    return _lib


def _err():
    b = getattr(_tls, "err", None)
    if b is None:
        b = _tls.err = _c.create_string_buffer(4096)
    return b


def _string(fn, *args) -> str:
    n = fn(*args, None, 0)
    buf = _c.create_string_buffer(max(n, 1))
    fn(*args, buf, n)
    return buf.value.decode()


def os_mpsgraph_version() -> str:
    return _string(lib().cai_os_mpsgraph_version)


def package_mpsgraph_version(path) -> str:
    return _string(lib().cai_package_mpsgraph_version, str(path).encode())


def _vtuple(v: str):
    return tuple(int(x) for x in v.split(".") if x.isdigit())


def loadable(path) -> bool:
    """False for a compiled package whose MPSGraph package is newer than this OS reads (loading it would crash)."""
    pv, ov = package_mpsgraph_version(path), os_mpsgraph_version()
    return not pv or not ov or _vtuple(pv) <= _vtuple(ov)


class _Handle:
    _h = None

    def __del__(self):
        h, self._h = self._h, None
        if h and _lib is not None:
            _lib.cai_release(h)


class Model(_Handle):
    def __init__(self, path, compute: str = "ane", check_version: bool = True):
        self.path = Path(path)
        e = _err()
        self._h = lib().cai_model_load(str(self.path).encode(), COMPUTE[compute], int(check_version), e, len(e))
        if not self._h:
            raise BridgeError(e.value.decode())

    @property
    def function_names(self) -> list[str]:
        s = _string(lib().cai_model_function_names, self._h)
        return s.split("\n") if s else []

    def function(self, name: str) -> "Function":
        return Function(self, name)


class Function(_Handle):
    def __init__(self, model: Model, name: str):
        self.model, self.name = model, name
        e = _err()
        self._h = lib().cai_function_load(model._h, name.encode(), e, len(e))
        if not self._h:
            raise BridgeError(e.value.decode())
        self.desc = json.loads(_string(lib().cai_function_describe, self._h))
        self.inputs = {v["name"]: v for v in self.desc["inputs"]}
        self.outputs = {v["name"]: v for v in self.desc["outputs"]}
        self.states = {v["name"]: v for v in self.desc["states"]}
        self.input_names = [v["name"] for v in self.desc["inputs"]]
        self.output_names = [v["name"] for v in self.desc["outputs"]]
        self.state_names = [v["name"] for v in self.desc["states"]]

    def spec(self, kind: str, name: str) -> dict:
        return {"input": self.inputs, "output": self.outputs, "state": self.states}[kind][name]

    def buffer(self, kind: str, name: str, packed: bool = False) -> "Buffer":
        """A zero-filled buffer matching an input / output / state (the model's preferred layout unless packed)."""
        s = self.spec(kind, name)
        return Buffer(s["shape"], s["dtype"], None if packed else s["strides"])

    def bind(self, inputs: dict, outputs: dict | None = None, states: dict | None = None) -> "Binding":
        return Binding(self, inputs, outputs or {}, states or {})

    def __call__(self, inputs: dict, outputs: dict | None = None, states: dict | None = None) -> dict:
        """One-off call (allocates a binding; missing outputs get new buffers). For loops, keep a Binding."""
        outputs = dict(outputs or {})
        for n in self.output_names:
            if n not in outputs:
                outputs[n] = self.buffer("output", n)
        run([self.bind(inputs, outputs, states)])
        return outputs


class Buffer(_Handle):
    """IOSurface-backed tensor. .np is a zero-copy numpy view (keeps the buffer alive). strides in elements."""

    def __init__(self, shape, dtype="float16", strides=None):
        if not isinstance(dtype, str):
            dtype = np.dtype(dtype).name
        code, self.dtype = DTYPES[dtype][0], np.dtype(DTYPES[dtype][1])
        self.shape = tuple(int(s) for s in shape)
        r = len(self.shape)
        sh = (_c.c_int64 * r)(*self.shape)
        st = (_c.c_int64 * r)(*[int(s) for s in strides]) if strides is not None else None
        e = _err()
        self._h = lib().cai_buffer_create(code, r, sh, st, e, len(e))
        if not self._h:
            raise BridgeError(e.value.decode())
        if strides is None:
            strides = [1] * r
            for j in range(r - 2, -1, -1):
                strides[j] = strides[j + 1] * self.shape[j + 1]
        self.strides = tuple(int(s) for s in strides)
        self.address = lib().cai_buffer_address(self._h)
        self.nbytes = lib().cai_buffer_nbytes(self._h)

    @property
    def np(self):
        # The ndarray retains this Buffer as its base. Caching the view on self
        # would form an uncollectable cycle through the untracked ndarray and
        # keep discarded IOSurfaces alive after context changes.
        return np.asarray(self)

    @property
    def __array_interface__(self):
        return {"shape": self.shape, "typestr": self.dtype.str, "data": (self.address, False),
                "strides": tuple(s * self.dtype.itemsize for s in self.strides), "version": 3}

    def zero(self):
        _c.memset(self.address, 0, self.nbytes)


def _names(ns):
    return (_S * len(ns))(*[n.encode() for n in ns]) if ns else None


def _handles(bs):
    return (_P * len(bs))(*[b._h for b in bs]) if bs else None


class Binding(_Handle):
    """Function + name -> Buffer maps, validated once (dtype, shape, all inputs / states bound, no input aliasing an
    output of the same call). Holds references to everything it binds."""

    def __init__(self, fn: Function, inputs: dict, outputs: dict, states: dict):
        self.fn, self.inputs, self.outputs, self.states = fn, dict(inputs), dict(outputs), dict(states)
        e = _err()
        ins, outs, sts = list(self.inputs), list(self.outputs), list(self.states)
        self._h = lib().cai_binding_create(
            fn._h, len(ins), _names(ins), _handles([self.inputs[n] for n in ins]),
            len(outs), _names(outs), _handles([self.outputs[n] for n in outs]),
            len(sts), _names(sts), _handles([self.states[n] for n in sts]), e, len(e))
        if not self._h:
            raise BridgeError(e.value.decode())


class Plan:
    """A fixed sequence of bindings (the ctypes argument array is built once)."""

    def __init__(self, bindings):
        self.bindings = list(bindings)
        self._arr = (_P * len(self.bindings))(*[b._h for b in self.bindings])
        self.times = np.zeros(len(self.bindings))

    def run(self, times: bool = False):
        e = _err()
        t = self.times.ctypes.data_as(_c.POINTER(_D)) if times else None
        rc = lib().cai_run(len(self.bindings), self._arr, t, e, len(e))
        if rc:
            raise BridgeError(e.value.decode())
        return self.times if times else None


def run(bindings, times: bool = False):
    return Plan(bindings).run(times)
