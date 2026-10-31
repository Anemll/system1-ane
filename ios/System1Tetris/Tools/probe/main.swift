import Foundation
// usage: probe BUILD_DIR(coreai) MODEL_DIR(model) golden.json [mode]
let a = CommandLine.arguments
if a.count > 4 { setenv("MPSGRAPH_ANE_BONDED_COMPILE_MODE", a[4], 1) }
let eng = try JeffEngine(build: URL(fileURLWithPath: a[1]), modelDir: URL(fileURLWithPath: a[2]))
eng.onProgress = { print($0); fflush(stdout) }
try await eng.load()
let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: a[3]))) as! [[String: Any]]
var agree = 0
for (i, c) in cases.enumerated() {
    let ids = c["ids"] as! [Int]
    let keys = c["keys"] as! [String]
    let d = try await eng.decide(ids: ids, nOptions: keys.count)
    let pick = keys[d.best], ok = pick == (c["oracle"] as! String)
    if ok { agree += 1 }
    print(i, "tok", ids.count, "pick", pick, "oracle", c["oracle"]!, ok ? "OK" : "DIFF", String(format: "conf %.3f total %.0fms calls %@ head %.1fms", d.probabilities[d.best], d.totalMs, d.callMs.map { String(format: "%.0f", $0) }.joined(separator: ","), d.headMs))
}
print("agree with oracle: \(agree)/\(cases.count)")
