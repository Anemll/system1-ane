import Foundation
// usage: golden_check MODEL_DIR golden.json
let a = CommandLine.arguments
let tok = try QwenTokenizer(url: URL(fileURLWithPath: a[1] + "/tokenizer.json"))
let dec = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: a[1] + "/decision_config.json"))) as! [String: Any]
let codes = dec["codes"] as! [String]
let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: a[2]))) as! [[String: Any]]
var bad = 0
for (i, c) in cases.enumerated() {
    let rows = (c["board"] as! String).split(separator: "\n").map { $0.map { UInt8($0 == "#" ? 1 : 0) } }
    let piece = c["piece"] as! String
    let opts = Tetris.placements(rows, piece)
    let keys = opts.map(\.key)
    let text = Prompt.text(board: rows, piece: piece, options: opts, codes: codes)
    let ids = tok.encode(text)
    let okKeys = keys == (c["keys"] as! [String]), okText = text == (c["text"] as! String)
    let okIds = ids == (c["ids"] as! [Int]), okOracle = Tetris.oracleKey(opts) == (c["oracle"] as! String)
    if !(okKeys && okText && okIds && okOracle) { bad += 1 }
    print(i, "keys", okKeys, "text", okText, "ids", okIds, "oracle", okOracle, ids.count)
}
print(bad == 0 ? "ALL MATCH" : "MISMATCHES: \(bad)")

// spawn rule parity (optional third argument: Tests/spawn.json)
if a.count > 3 {
    let spawn = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: a[3]))) as! [[String: Any]]
    var badSpawn = 0
    for c in spawn {
        let rows = (c["board"] as! String).split(separator: "\n").map { $0.map { UInt8($0 == "#" ? 1 : 0) } }
        if Tetris.spawnBlocked(rows, c["piece"] as! String) != (c["blocked"] as! Bool) { badSpawn += 1 }
    }
    print(badSpawn == 0 ? "SPAWN RULE MATCHES (\(spawn.count) boards)" : "SPAWN MISMATCHES: \(badSpawn)")
}
