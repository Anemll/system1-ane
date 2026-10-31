// Qwen byte-level BPE (tokenizer.json): NFC, split regex, GPT-2 byte map, ranked merges, added/special tokens.
import Foundation

final class QwenTokenizer {
    private var vocab: [String: Int] = [:]
    private var ranks: [String: Int] = [:]
    private var added: [(content: String, id: Int)] = []
    private let regex: NSRegularExpression
    private var byteChar: [String] = Array(repeating: "", count: 256)
    private var cache: [String: [Int]] = [:]

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let v = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else { throw NSError(domain: "tokenizer", code: 1) }
        vocab = v
        ranks.reserveCapacity(merges.count)
        for (i, m) in merges.enumerated() {
            if let p = m as? [String], p.count == 2 { ranks[p[0] + " " + p[1]] = i }
            else if let s = m as? String { ranks[s] = i }
        }
        for a in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            if let c = a["content"] as? String, let id = a["id"] as? Int { added.append((c, id)) }
        }
        // the Split regex of the Qwen pre-tokenizer
        var pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        if let pre = root["pre_tokenizer"] as? [String: Any], let list = pre["pretokenizers"] as? [[String: Any]],
           let pat = (list.first?["pattern"] as? [String: Any])?["Regex"] as? String { pattern = pat }
        regex = try NSRegularExpression(pattern: pattern)
        // GPT-2 bytes_to_unicode
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) { bs.append(b); cs.append(256 + n); n += 1 }
        for (b, c) in zip(bs, cs) { byteChar[b] = String(UnicodeScalar(UInt32(c))!) }
    }

    private func bpe(_ piece: String) -> [Int] {
        if let hit = cache[piece] { return hit }
        var syms = piece.utf8.map { byteChar[Int($0)] }
        while syms.count > 1 {
            var best = Int.max, at = -1
            for i in 0..<(syms.count - 1) {
                if let r = ranks[syms[i] + " " + syms[i + 1]], r < best { best = r; at = i }
            }
            if at < 0 { break }
            syms.replaceSubrange(at...(at + 1), with: [syms[at] + syms[at + 1]])
        }
        let ids = syms.compactMap { vocab[$0] }
        cache[piece] = ids
        return ids
    }

    private func encodePlain(_ text: String) -> [Int] {
        let s = text.precomposedStringWithCanonicalMapping
        let ns = s as NSString
        var out: [Int] = []
        var last = 0
        for m in regex.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            if m.range.location > last { out += bpe(ns.substring(with: NSRange(location: last, length: m.range.location - last))) }
            out += bpe(ns.substring(with: m.range))
            last = m.range.location + m.range.length
        }
        if last < ns.length { out += bpe(ns.substring(from: last)) }
        return out
    }

    /// Token ids with add_special_tokens=False; added tokens in the text are matched first.
    func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            var hit: (Range<Substring.Index>, Int)? = nil
            for a in added {
                if let r = rest.range(of: a.content), hit == nil || r.lowerBound < hit!.0.lowerBound { hit = (r, a.id) }
            }
            guard let (r, id) = hit else { out += encodePlain(String(rest)); break }
            if r.lowerBound > rest.startIndex { out += encodePlain(String(rest[rest.startIndex..<r.lowerBound])) }
            out.append(id)
            rest = rest[r.upperBound...]
        }
        return out
    }
}
