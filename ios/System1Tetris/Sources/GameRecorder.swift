// Records every game as a compact JSON file (decisions only: no device identifiers, no video), encrypted at rest, kept on the
// device and exportable from the app. Options are not stored: they are a pure function of (board, piece), so a trainer
// rebuilds them with placements(); only the model's probabilities, its pick and the El-Tetris label are kept.
// This build has no networking for game data.
import CryptoKit
import Foundation
import Security

/// AES-256-GCM key kept in the Keychain (this device only, never synced or backed up). Games are stored sealed with it.
enum GameVault {
    private static let account = "system1ane.games.key"
    private static var key: SymmetricKey = {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrAccount as String: account,
                                kSecReturnData as String: true]
        var out: CFTypeRef?
        if SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data { return SymmetricKey(data: d) }
        let k = SymmetricKey(size: .bits256)
        let add: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrAccount as String: account,
                                  kSecValueData as String: k.withUnsafeBytes { Data($0) },
                                  kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]
        SecItemAdd(add as CFDictionary, nil)
        return k
    }()
    static func seal(_ d: Data) -> Data? { try? AES.GCM.seal(d, using: key).combined }
    static func open(_ d: Data) -> Data? { try? AES.GCM.open(AES.GCM.SealedBox(combined: d), using: key) }
}

struct DecisionRecord: Codable {
    let board: String        // 20 rows of "#"/"." joined by "\\n", before the move
    let piece: String
    let pick: String         // option key the model chose, e.g. "r1c4"
    let oracle: String?      // El-Tetris label for the same position
    let probs: [Float]       // model probabilities in placements() order, rounded
    let ms: Int
    let tokens: Int
}

struct GameRecord: Codable {
    let schema = "system1-ane-tetris-v1"
    let id: String
    let app: String
    let model: String
    let mode: String
    let started: Date
    var lines = 0
    var pieces = 0
    var toppedOut = false
    var capped = false
    var decisions: [DecisionRecord] = []
}

@MainActor
final class GameRecorder: ObservableObject {
    @Published var pending = 0

    private(set) var current: GameRecord?
    private let dir: URL

    init() {
        dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("games", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var res = URLResourceValues(); res.isExcludedFromBackup = true
        var d = dir; try? d.setResourceValues(res)   // not in iCloud / iTunes backups
        refresh()
    }

    func begin(model: String, mode: String) {
        current = GameRecord(id: UUID().uuidString.lowercased(), app: appVersion, model: model, mode: mode, started: Date())
    }

    func add(_ d: DecisionRecord) { current?.decisions.append(d) }

    /// Close the current game; games with fewer than 3 decisions are dropped.
    func finish(lines: Int, pieces: Int, toppedOut: Bool, capped: Bool = false) {
        guard var g = current else { return }
        current = nil
        guard g.decisions.count >= 3 else { return }
        g.lines = lines; g.pieces = pieces; g.toppedOut = toppedOut; g.capped = capped
        let enc = JSONEncoder(); enc.dateEncodingStrategy = .iso8601; enc.outputFormatting = [.sortedKeys]
        if let data = try? enc.encode(g), let sealed = GameVault.seal(data) {
            try? sealed.write(to: dir.appendingPathComponent("\(g.id).game"), options: [.atomic, .completeFileProtection])
        }
        trimLocal()
        refresh()
    }

    private var appVersion: String {
        "\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?")(\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"))"
    }

    private func files() -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "game" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func refresh() { pending = files().count }

    /// Keep at most 300 saved games on the phone.
    private func trimLocal() { for f in files().dropLast(300) { try? FileManager.default.removeItem(at: f) } }

    /// All saved games as one JSON-lines file for the share sheet.
    func exportFile() -> URL? {
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("system1-ane-games.jsonl")
        var data = Data()
        for f in files() { if let sealed = try? Data(contentsOf: f), let d = GameVault.open(sealed) { data.append(d); data.append(10) } }
        guard !data.isEmpty else { return nil }
        try? data.write(to: out)
        return out
    }

    func deleteLocal() { for f in files() { try? FileManager.default.removeItem(at: f) }; refresh() }
}
