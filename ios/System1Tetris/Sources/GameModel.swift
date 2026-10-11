// Autoplay loop: random piece -> legal placements -> Jeff prompt -> ANE decision -> animate and apply -> repeat.
import Foundation
import SwiftUI
import UIKit

struct OptionScore: Identifiable {
    let id: String   // placement key
    let text: String
    let probability: Float
    let isOracle: Bool
    let isPick: Bool
}

enum RunState: Equatable { case needsModel, loading, ready, playing, paused, sessionDone, failed(String) }

func appLog(_ s: String) {
    print("[System1ANE] \(s)"); fflush(stdout)
    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("system1ane.log")
    let line = "\(Date().formatted(date: .omitted, time: .standard)) \(s)\n"
    if let h = try? FileHandle(forWritingTo: url) { h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close() }
    else { try? line.write(to: url, atomically: true, encoding: .utf8) }
}

@MainActor
final class GameModel: ObservableObject {
    static let colors: [Color] = [.gray, .cyan, .yellow, .purple, .green, .red, .blue, .orange]   // index = piece id (1...7)
    static let pieceId: [String: UInt8] = ["I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7]

    @Published var run: RunState = .loading
    @Published var log: [String] = []
    @Published var board: [[UInt8]] = Tetris.emptyBoard()
    @Published var piece = "T"
    @Published var nextPiece = "I"
    struct Fall {
        let cells: [Tetris.Cell]; let id: UInt8
        let start: Date; let duration: Double
        let fromCol: Int, toCol: Int, toRow: Int
    }
    @Published var fall: Fall? = nil   // the piece in flight; BoardView interpolates it per display frame (120 Hz on ProMotion)
    @Published var lines = 0 { didSet { bestLines = max(bestLines, lines) } }
    @Published var pieces = 0
    @Published var games = 1
    @Published var bestLines = UserDefaults.standard.integer(forKey: "bestLines") { didSet { UserDefaults.standard.set(bestLines, forKey: "bestLines") } }
    @Published var totalLines = 0
    @Published var lastMs = 0.0
    @Published var avgMs = 0.0
    @Published var tokens = 0
    @Published var calls = 0
    @Published var confidence: Float = 0
    @Published var agree = 0
    @Published var decided = 0
    @Published var top: [OptionScore] = []
    @Published var status = "Starting"
    enum Mode: String, CaseIterable { case realtime = "Real-time", fast = "Fast" }
    @Published var mode: Mode = .realtime
    @Published var overrides = 0
    @Published var explore = false   // sample moves from the model's own probabilities instead of always taking the top one
    @Published var callMs: [Double] = []
    @Published var modelName = ""

    let downloader = ModelDownloader()
    let recorder = GameRecorder()
    static let sessionLimit = 20   // games per session; then the player taps to start another batch
    @Published var sessionGames = 0
    @Published var sessionLines = 0
    @Published var toast: String?
    static let maxPieces = 120   // a game ends here even if the model has not topped out: bounded files, steady uploads
    private var engine: JeffEngine?
    private var task: Task<Void, Never>?
    private var rng = SystemRandomNumberGenerator()
    private var msSum = 0.0

    var oracleAgreement: Double { decided == 0 ? 0 : Double(agree) / Double(decided) }

    static var root: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("tetris", isDirectory: true)
    }

    private func note(_ s: String) { appLog(s); log.append(s); if log.count > 60 { log.removeFirst() }; status = s }

    @Published var loadStep = 0
    let loadTotal = 7
    private var starting = false

    func start() {
        guard !starting, engine == nil else { return }
        starting = true
        if let mode = UserDefaults.standard.string(forKey: "aneMode"), !mode.isEmpty {
            setenv("MPSGRAPH_ANE_BONDED_COMPILE_MODE", mode, 1)
            note("ANE compile mode \(mode)")
        }
        let build = Self.root.appendingPathComponent("coreai"), model = Self.root.appendingPathComponent("model")
        guard ModelDownloader.isInstalled(Self.root) else { starting = false; run = .needsModel; return }
        run = .loading; loadStep = 0
        task = Task {
            do {
                let e = try JeffEngine(build: build, modelDir: model)
                e.onProgress = { [weak self] s in Task { @MainActor in self?.note(s); if s.hasPrefix("Loaded") { self?.loadStep += 1 } } }
                try await e.load()
                engine = e
                modelName = e.modelName
                run = .playing
                note("Playing")
                await loop(e)
            } catch {
                starting = false; run = .failed("\(error)"); note("Failed: \(error)")
            }
        }
    }

    func downloadModel() {
        Task {
            if await downloader.download(to: Self.root) { start() }
        }
    }

    /// Delete the installed build and fetch it again (also the way to recover from a bad copy).
    func redownloadModel() {
        task?.cancel(); task = nil; engine = nil; starting = false
        try? FileManager.default.removeItem(at: Self.root)
        run = .needsModel
        downloadModel()
    }

    func togglePause() {
        switch run {
        case .playing: run = .paused
        case .paused: run = .playing
        default: break
        }
    }

    private var epoch = 0   // bumped by every reset; work started in an older epoch is discarded

    func newGame() { recorder.finish(lines: lines, pieces: pieces, toppedOut: false); epoch += 1; resetBoard(); games += 1 }

    /// Called when a game ends (top-out or piece cap): toast the result, and stop after sessionLimit games.
    private func gameEnded(lines l: Int) {
        sessionGames += 1; sessionLines += l
        toast = "Game \(sessionGames) of \(Self.sessionLimit) complete: \(l) lines"
        Task {
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            toast = "Game saved on this iPhone"
            try? await Task.sleep(nanoseconds: 2_200_000_000); toast = nil
        }
        if sessionGames >= Self.sessionLimit { run = .sessionDone }
    }

    /// The player tapped "Play 20 more games".
    func startNextSession() {
        sessionGames = 0; sessionLines = 0; epoch += 1; resetBoard(); run = .playing
    }

    private func resetBoard() {
        recorder.begin(model: modelName, mode: mode.rawValue, explore: explore)
        board = Tetris.emptyBoard(); lines = 0; pieces = 0; top = []; fall = nil
        piece = randomPiece(); nextPiece = randomPiece()
    }

    private func randomPiece() -> String { Tetris.pieceNames[Int.random(in: 0..<7, using: &rng)] }

    private func binary(_ b: [[UInt8]]) -> Tetris.Board { b.map { $0.map { $0 != 0 ? 1 : 0 } } }

    private struct Result {
        let board: Tetris.Board, piece: String
        let options: [Tetris.Placement]
        let oracle: String?
        let decision: Decision
    }

    /// Prompt + tokenize + ANE decision for one board, off the main thread.
    private func decide(_ e: JeffEngine, board: Tetris.Board, piece: String) -> Task<Result?, Error> {
        Task.detached(priority: .userInitiated) {
            let options = Tetris.placements(board, piece)
            guard !options.isEmpty, !Tetris.spawnBlocked(board, piece) else { return nil }   // game over
            let ids = e.tokens(for: Prompt.text(board: board, piece: piece, options: options, codes: e.codes))
            let d = try await e.decide(ids: ids, nOptions: options.count)
            return Result(board: board, piece: piece, options: options, oracle: Tetris.oracleKey(options), decision: d)
        }
    }

    /// Index drawn from a probability vector (the model's calibrated distribution, temperature already applied).
    private static func sample(_ p: [Float]) -> Int {
        var u = Float.random(in: 0..<1)
        for (i, x) in p.enumerated() { u -= x; if u < 0 { return i } }
        return p.count - 1
    }

    private func loop(_ e: JeffEngine) async {
        resetBoard()
        var pending = decide(e, board: binary(board), piece: piece)
        while !Task.isCancelled {
            if run == .paused || run == .sessionDone { try? await Task.sleep(nanoseconds: 100_000_000); continue }
            do {
                guard let r = try await pending.value else {   // top-out: record, pause briefly, start the next game
                    bestLines = max(bestLines, lines); totalLines += lines
                    note("Game over: \(lines) lines, \(pieces) pieces")
                    recorder.finish(lines: lines, pieces: pieces, toppedOut: true)
                    gameEnded(lines: lines)
                    try? await Task.sleep(nanoseconds: 2_500_000_000)
                    epoch += 1; games += 1; resetBoard()
                    pending = decide(e, board: binary(board), piece: piece); continue
                }
                if r.board != binary(board) || r.piece != piece {   // New game (or reset) while this was in flight
                    pending = decide(e, board: binary(board), piece: piece); continue
                }
                let myEpoch = epoch
                let d = r.decision
                let chosen = explore ? Self.sample(d.probabilities) : d.best
                let pick = r.options[chosen]
                recorder.add(DecisionRecord(board: Tetris.boardText(r.board), piece: r.piece, pick: r.options[chosen].key, oracle: r.oracle,
                                            probs: r.decision.probabilities.map { (($0 * 1000).rounded() / 1000) }, ms: Int(r.decision.totalMs), tokens: r.decision.tokens))
                appLog("piece \(r.piece) pick \(pick.key) oracle \(r.oracle ?? "-") conf \(d.probabilities[d.best]) tok \(d.tokens) ms \(Int(d.totalMs)) calls \(d.callMs.map { Int($0) })")
                lastMs = d.totalMs; msSum += d.totalMs; decided += 1; avgMs = msSum / Double(decided)
                tokens = d.tokens; calls = d.calls; callMs = d.callMs; confidence = d.probabilities[d.best]
                if pick.key == r.oracle { agree += 1 } else { overrides += 1 }
                top = d.probabilities.indices.sorted { d.probabilities[$0] > d.probabilities[$1] }.prefix(4).map {
                    OptionScore(id: r.options[$0].key, text: "\(r.options[$0].key)  \(Tetris.describe(r.piece, r.options[$0]).components(separatedBy: ", ").dropFirst().prefix(2).joined(separator: ", "))",
                                probability: d.probabilities[$0], isOracle: r.options[$0].key == r.oracle, isPick: $0 == chosen)
                }
                // The board after this move is known now, so the next decision runs on the ANE while the piece animates.
                let id = Self.pieceId[piece]!
                var nb = board
                for (dr, dc) in pick.shape { nb[pick.row + dr][pick.column + dc] = id }
                let kept = nb.filter { row in !row.allSatisfy { $0 != 0 } }
                let cleared = nb.count - kept.count
                nb = Array(repeating: Array(repeating: UInt8(0), count: Tetris.width), count: cleared) + kept
                let upcoming = nextPiece
                pending = decide(e, board: binary(nb), piece: upcoming)
                await animate(pick, id: id)
                if myEpoch != epoch {   // New game pressed during the animation: drop this move and its look-ahead
                    fall = nil; pending.cancel()
                    pending = decide(e, board: binary(board), piece: piece); continue
                }
                board = nb; fall = nil
                lines += cleared; pieces += 1
                piece = upcoming; nextPiece = randomPiece()
                if pieces >= Self.maxPieces {   // capped: close the record and start the next game
                    note("Game \(games) complete: \(lines) lines, \(pieces) pieces")
                    recorder.finish(lines: lines, pieces: pieces, toppedOut: false, capped: true)
                    gameEnded(lines: lines)
                    totalLines += lines; pending.cancel()
                    try? await Task.sleep(nanoseconds: 1_200_000_000)
                    epoch += 1; games += 1; resetBoard()
                    pending = decide(e, board: binary(board), piece: piece)
                }
            } catch {
                run = .failed("\(error)"); note("Failed: \(error)"); return
            }
        }
    }

    /// Real-time: the piece spawns at the top, slides to its column and falls to its slot in ~0.1-0.28 s. The motion itself
    /// is interpolated by BoardView every display frame, so it is smooth at the panel's full refresh rate. Fast: no animation.
    private func animate(_ p: Tetris.Placement, id: UInt8) async {
        guard mode == .realtime else { return }
        let duration = min(0.28, max(0.10, 0.06 + 0.012 * Double(p.row)))
        fall = Fall(cells: p.shape, id: id, start: Date(), duration: duration, fromCol: 3, toCol: p.column, toRow: p.row)
        try? await Task.sleep(nanoseconds: UInt64((duration + 0.03) * 1e9))
    }
}
