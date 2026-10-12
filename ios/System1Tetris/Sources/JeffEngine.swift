// Core AI runner for a Jeff decision build (port of coreai/jeff_coreai_runtime.py, prefill-only, single width).
// Chunks of 4 layers run in sequence; GDN conv / recurrent state ping-pongs between two buffer sets, k / v rows are
// copied into the KV caches, and the readout head scores the last prompt row.
import CoreAI
import Foundation
import IOSurface

struct EngineError: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}

enum Dtype {
    static func size(_ t: NDArray.ScalarType) -> Int {
        switch t {
        case .float16, .bfloat16, .int16, .uint16: return 2
        case .float32, .int32: return 4
        case .int8, .uint8, .bool: return 1
        default: return 8
        }
    }
}

struct IOSpec { let type: NDArray.ScalarType; let shape: [Int] }

func ioSpec(_ v: InferenceValue.Descriptor?) -> IOSpec? {
    guard let v else { return nil }
    if case .ndArray(let d) = v { return IOSpec(type: d.scalarType, shape: d.shape) }
    return nil
}

/// Packed IOSurface tensor (zero-filled). width = last dimension, as Core AI requires.
final class Buf {
    let surface: IOSurface
    let type: NDArray.ScalarType
    let shape: [Int]
    let strides: [Int]
    var count: Int { shape.reduce(1, *) }
    init(_ type: NDArray.ScalarType, _ shape: [Int]) throws {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else { throw EngineError("bad shape \(shape)") }
        var st = [Int](repeating: 1, count: shape.count)
        for j in stride(from: shape.count - 2, through: 0, by: -1) { st[j] = st[j + 1] * shape[j + 1] }
        let es = Dtype.size(type), width = shape.last!
        let rowBytes = width * es
        let height = max(1, shape.dropLast().reduce(1, *))
        let props: [IOSurfacePropertyKey: Any] = [.width: width, .height: height, .bytesPerElement: es,
                                                  .bytesPerRow: rowBytes, .allocSize: max(rowBytes * height, 16), .pixelFormat: 0]
        guard let s = IOSurface(properties: props) else { throw EngineError("IOSurface \(shape) failed") }
        s.lock(options: [], seed: nil); memset(s.baseAddress, 0, s.allocationSize); s.unlock(options: [], seed: nil)
        surface = s; self.type = type; self.shape = shape; strides = st
    }
    var f16: UnsafeMutablePointer<Float16> { surface.baseAddress.assumingMemoryBound(to: Float16.self) }
    func zero() { memset(surface.baseAddress, 0, count * Dtype.size(type)) }
    func readFloats() -> [Float] {
        switch type {
        case .float32: return Array(UnsafeBufferPointer(start: surface.baseAddress.assumingMemoryBound(to: Float.self), count: count))
        default: return (0..<count).map { Float(f16[$0]) }
        }
    }
}

final class Fn {
    let fn: InferenceFunction
    let inputs: [String: IOSpec], outputs: [String: IOSpec]
    init(_ fn: InferenceFunction) {
        self.fn = fn
        let d = fn.descriptor
        var i: [String: IOSpec] = [:], o: [String: IOSpec] = [:]
        for n in d.inputNames { if let s = ioSpec(d.inputDescriptor(of: n)) { i[n] = s } }
        for n in d.outputNames { if let s = ioSpec(d.outputDescriptor(of: n)) { o[n] = s } }
        inputs = i; outputs = o
    }

    func run(_ ins: [String: Buf], _ outs: [String: Buf]) async throws {
        var inputs = InferenceFunction.Inputs()
        for (n, b) in ins {
            let s = b.surface
            inputs.insert(_overrideLifetime(NDArray.RawView(ioSurface: s, scalarType: b.type, shape: b.shape,
                                                            strides: b.strides), borrowing: self), for: n)
        }
        var views = InferenceFunction.MutableViews()
        for (n, b) in outs {
            let s = b.surface
            views.insert(_overrideLifetime(NDArray.MutableRawView(ioSurface: s, scalarType: b.type, shape: b.shape,
                                                                  strides: b.strides), borrowing: self), for: n)
        }
        _ = try await fn.run(inputs: inputs, states: InferenceFunction.MutableViews(), outputViews: views)
    }
}

struct Decision {
    var probabilities: [Float]
    var best: Int
    var tokens: Int
    var calls: Int
    var callMs: [Double]
    var headMs: Double
    var totalMs: Double
}

final class JeffEngine {
    struct ChunkSpec: Decodable {
        let file: String; let entries: [String]; let gdn_j: [Int]; let att_j: [Int]
    }
    struct Head: Decodable { let file: String; let entry: String? }
    struct Convert: Decodable { let temperature: Double?; let model: String? }
    struct Manifest: Decodable {
        let TP: Int; let pend: Int; let pctxs: [Int]; let pkv_len: [String: Int]
        let chunks: [ChunkSpec]; let head: Head; let convert: Convert?
    }

    final class Chunk {
        let spec: ChunkSpec
        var fn: Fn!
        var model: AIModel!
        var cur: [String: Buf] = [:], alt: [String: Buf] = [:]   // state in / state out sets
        var kv: [String: Buf] = [:], kvNew: [String: Buf] = [:]
        var y: Buf!
        init(_ s: ChunkSpec) { spec = s }
    }

    let build: URL, modelDir: URL
    let man: Manifest
    let codes: [String]
    let temperature: Float
    let tokenizer: QwenTokenizer
    let hid: Int, nkv: Int, hd: Int, rot: Int, ctx: Int, TP: Int, P: Int
    let inv: [Double]
    private var emb: Data
    private let embOffset: Int
    private(set) var chunks: [Chunk] = []
    private var headFn: Fn!
    private var headModel: AIModel!
    private var shared: [String: Buf] = [:]
    private var x: Buf!
    private var hx: Buf!
    private var logits: Buf!
    private var pos = 0
    var onProgress: ((String) -> Void)?

    init(build: URL, modelDir: URL) throws {
        self.build = build; self.modelDir = modelDir
        man = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: build.appendingPathComponent("manifest.json")))
        let dec = try JSONSerialization.jsonObject(with: Data(contentsOf: modelDir.appendingPathComponent("decision_config.json"))) as! [String: Any]
        codes = dec["codes"] as! [String]
        temperature = Float(man.convert?.temperature ?? (dec["temperature"] as! Double))
        tokenizer = try QwenTokenizer(url: modelDir.appendingPathComponent("tokenizer.json"))
        let cfgRaw = try JSONSerialization.jsonObject(with: Data(contentsOf: modelDir.appendingPathComponent("config.json"))) as! [String: Any]
        let c = (cfgRaw["text_config"] as? [String: Any]) ?? cfgRaw
        hid = c["hidden_size"] as! Int; nkv = c["num_key_value_heads"] as! Int; hd = c["head_dim"] as! Int
        let rp = c["rope_parameters"] as! [String: Any]
        let rotary = Int(Double(c["head_dim"] as! Int) * (rp["partial_rotary_factor"] as! Double))
        let theta = (rp["rope_theta"] as! NSNumber).doubleValue
        rot = rotary
        inv = stride(from: 0, to: rotary, by: 2).map { 1.0 / pow(theta, Double($0) / Double(rotary)) }
        TP = man.TP; P = man.pend; ctx = man.pkv_len[String(man.pctxs[0])]!
        emb = try Data(contentsOf: modelDir.appendingPathComponent("embed_tokens_fp16.npy"), options: .alwaysMapped)
        let hlen = Int(emb[8]) | Int(emb[9]) << 8
        embOffset = 10 + hlen
    }

    static func freeGB() -> String {
        let v = try? URL(fileURLWithPath: NSHomeDirectory()).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return String(format: "%.1f", Double(v?.volumeAvailableCapacityForImportantUsage ?? 0) / 1e9)
    }

    var modelName: String { "\(man.convert?.model ?? "jeff") · \(man.chunks.count) chunks · fp16 · \(entry)" }

    var entry: String { "p\(TP)_\(man.pctxs[0] / 1024)k" }

    /// Loads (and on first launch specializes for the ANE) every package. Slow the first time.
    func load(computeUnit: SpecializationOptions = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)) async throws {
        let t0 = Date()
        var pins: [String: IOSpec] = [:]
        for spec in man.chunks {
            let t1 = Date()
            onProgress?("Loading \(spec.file)")
            let ch = Chunk(spec)
            do { ch.model = try await AIModel(contentsOf: build.appendingPathComponent(spec.file), options: computeUnit) }
            catch { throw EngineError("\(spec.file): \(String(reflecting: error)) (free space \(Self.freeGB()) GB)") }
            let loaded: AnyObject?
            do { loaded = try ch.model.loadFunction(named: entry) as AnyObject? }
            catch { throw EngineError("\(spec.file) loadFunction: \(String(reflecting: error)) \(error as NSError)") }
            guard loaded != nil, let f = try ch.model.loadFunction(named: entry) else { throw EngineError("\(spec.file): no function \(entry)") }
            ch.fn = Fn(f)
            let stateBases = ["conv", "rec", "pend"]
            for (name, s) in ch.fn.inputs {
                if ["cos", "sin", "conv_sel", "commit", "commit_last", "conv_sel_out", "valid", "mask", "x"].contains(name) {
                    pins[name] = s
                } else if name.hasPrefix("k") || name.hasPrefix("v") {
                    ch.kv[name] = try Buf(s.type, s.shape)
                } else if stateBases.contains(where: { name.hasPrefix($0) }) {
                    ch.cur[name] = try Buf(s.type, s.shape); ch.alt[name] = try Buf(s.type, s.shape)
                } else { throw EngineError("\(spec.file): unexpected input \(name)") }
            }
            for (name, s) in ch.fn.outputs {
                if name == "y" { ch.y = try Buf(s.type, s.shape) }
                else if name.hasSuffix("_new") { ch.kvNew[String(name.dropLast(4))] = try Buf(s.type, s.shape) }
                else if !name.hasSuffix("_out") { throw EngineError("\(spec.file): unexpected output \(name)") }
            }
            chunks.append(ch)
            onProgress?("Loaded \(spec.file) in \(String(format: "%.1f", Date().timeIntervalSince(t1)))s")
        }
        for (name, s) in pins {
            let b = try Buf(s.type, s.shape)
            if name == "x" { x = b } else { shared[name] = b }
        }
        onProgress?("Loading head")
        headModel = try await AIModel(contentsOf: build.appendingPathComponent(man.head.file), options: computeUnit)
        let he = man.head.entry ?? "h1"
        guard let hf = try headModel.loadFunction(named: he) else { throw EngineError("head: no function \(he)") }
        headFn = Fn(hf)
        guard let hs = headFn.inputs["x"], let ls = headFn.outputs["logits"] else { throw EngineError("head I/O mismatch") }
        hx = try Buf(hs.type, hs.shape); logits = try Buf(ls.type, ls.shape)
        onProgress?("Warming up the ANE")
        _ = try await decide(ids: [Int](repeating: 0, count: TP), nOptions: 1)   // first call after load is ~5x slower
        onProgress?("Ready in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
    }

    func reset() {
        for ch in chunks { for b in ch.cur.values { b.zero() }; for b in ch.alt.values { b.zero() } }
        pos = 0
    }

    func tokens(for text: String) -> [Int] { tokenizer.encode(text) }

    private func fillPins(n: Int, p0: Int) throws {
        let w = TP
        guard let cos = shared["cos"], let sin = shared["sin"], let convSel = shared["conv_sel"], let commit = shared["commit"],
              let commitLast = shared["commit_last"], let convSelOut = shared["conv_sel_out"], let valid = shared["valid"],
              let mask = shared["mask"] else { throw EngineError("missing shared inputs: \(shared.keys.sorted())") }
        // embeddings -> x (1, hid, 1, w): x[h * w + i]
        x.zero()
        let xp = x.f16
        embedRows { i, row in for h in 0..<hid { xp[h * w + i] = row[h] } }
        // rope tables (w, rot); positions past n repeat the last one
        let cp = cos.f16, sp = sin.f16
        for i in 0..<w {
            let pi = Double(min(p0 + i, p0 + n - 1))
            for k in 0..<(rot / 2) {
                let a = pi * inv[k], cv = Float16(Foundation.cos(a)), sv = Float16(Foundation.sin(a))
                cp[i * rot + k] = cv; cp[i * rot + k + rot / 2] = cv
                sp[i * rot + k] = sv; sp[i * rot + k + rot / 2] = sv
            }
        }
        convSel.zero(); commit.zero(); commitLast.zero(); convSelOut.zero(); valid.zero()
        let cs = convSel.f16, cso = convSelOut.f16, cols = convSel.shape[1], colsOut = convSelOut.shape[1]
        for r in 0..<3 { cs[r * cols + r] = 1; cso[r * colsOut + n + r] = 1 }
        for i in 0..<n { valid.f16[i] = 1 }
        let mp = mask.f16
        for j in 0..<mask.count { mp[j] = j < p0 ? 0 : Float16(-1e4) }
    }

    private var pendingIds: [Int] = []
    private func embedRows(_ body: (Int, UnsafePointer<Float16>) -> Void) {
        emb.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: embOffset).assumingMemoryBound(to: Float16.self)
            for (i, id) in pendingIds.enumerated() { body(i, base + id * hid) }
        }
    }

    private func block(_ ids: [Int]) async throws {
        let n = ids.count, p0 = pos
        guard n > 0, n <= TP, p0 + n <= ctx else { throw EngineError("\(p0 + n) positions exceed the \(ctx)-row KV cache") }
        pendingIds = ids
        try fillPins(n: n, p0: p0)
        var xin: Buf = x
        for ch in chunks {
            var ins = shared
            ins["x"] = xin
            for (k, b) in ch.cur { ins[k] = b }
            for (k, b) in ch.kv { ins[k] = b }
            var outs: [String: Buf] = ["y": ch.y]
            for (k, b) in ch.alt { outs[k + "_out"] = b }
            for (k, b) in ch.kvNew { outs[k + "_new"] = b }
            try await ch.fn.run(ins.filter { ch.fn.inputs[$0.key] != nil }, outs)
            swap(&ch.cur, &ch.alt)
            for (k, newRows) in ch.kvNew {   // (nkv, TP, hd) -> cache[:, p0:p0+n]
                let cache = ch.kv[k]!
                for h in 0..<nkv {
                    memcpy(cache.f16 + (h * ctx + p0) * hd, newRows.f16 + h * TP * hd, n * hd * 2)
                }
            }
            xin = ch.y
        }
        // last row of the final y (1, hid, 1, TP) -> hx
        let y = chunks.last!.y!.f16, hp = hx.f16
        for h in 0..<hid { hp[h] = y[h * TP + n - 1] }
        pos = p0 + n
    }

    /// One independent decision: prefill every token in TP-row calls, then the readout head.
    func decide(ids: [Int], nOptions: Int) async throws -> Decision {
        guard !ids.isEmpty else { throw EngineError("empty prompt") }
        reset()
        let t0 = Date()
        var callMs: [Double] = []
        var i = 0
        while i < ids.count {
            let t1 = Date()
            try await block(Array(ids[i..<min(i + TP, ids.count)]))
            callMs.append(Date().timeIntervalSince(t1) * 1e3)
            i += TP
        }
        let t2 = Date()
        try await headFn.run(["x": hx], ["logits": logits])
        let headMs = Date().timeIntervalSince(t2) * 1e3
        let raw = logits.readFloats()
        let z = raw.prefix(nOptions).map { Double($0) / Double(temperature) }
        let m = z.max() ?? 0
        let e = z.map { exp($0 - m) }, s = e.reduce(0, +)
        let p = e.map { Float($0 / s) }
        let best = p.indices.max { p[$0] < p[$1] } ?? 0
        return Decision(probabilities: p, best: best, tokens: ids.count, calls: callMs.count, callMs: callMs, headMs: headMs,
                        totalMs: Date().timeIntervalSince(t0) * 1e3)
    }
}
