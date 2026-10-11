import SwiftUI

struct ContentView: View {
    @StateObject private var game = GameModel()
    @StateObject private var recorder = ScreenRecorder()
    @State private var showLog = false
    @State private var showData = false
    @State private var exportItem: ShareItem?
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(.systemBackground), Color(.secondarySystemBackground)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            playing
            overlay
            if let t = game.toast {
                Text(t).font(.footnote.weight(.semibold)).padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.thinMaterial, in: Capsule()).frame(maxHeight: .infinity, alignment: .top).padding(.top, 54)
                    .transition(.opacity)
            }
        }
        .task { UIApplication.shared.isIdleTimerDisabled = true; game.start() }
        .sheet(item: Binding(get: { recorder.lastFile.map { ShareItem(url: $0) } }, set: { if $0 == nil { recorder.lastFile = nil } })) { item in
            ShareSheet(url: item.url)
        }
        .sheet(isPresented: $showData) { dataSheet }
        .sheet(item: $exportItem) { ShareSheet(url: $0.url) }
    }

    private var dataSheet: some View {
        DataSheet(recorder: game.recorder) { item in showData = false; DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { exportItem = item } }
    }

    private func header(_ sub: String) -> some View {
        VStack(spacing: 2) {
            Text("System 1 ANE").font(.title2.weight(.bold))
            Text(sub).font(.footnote).foregroundStyle(.secondary)
        }
    }

    // MARK: status overlay (the Tetris UI is always visible behind it)

    @ViewBuilder private var overlay: some View {
        switch game.run {
        case .needsModel: card { downloadCard }
        case .loading: card { loadingCard }
        case .sessionDone: card { sessionCard }
        case .failed(let msg): card { failedCard(msg) }
        default: EmptyView()
        }
    }

    private func card<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()
            VStack(spacing: 14) { content() }
                .padding(20).frame(maxWidth: 340)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
        }.transition(.opacity)
    }

    private var downloadCard: some View {
        Group {
            Image(systemName: "cpu").font(.system(size: 34)).foregroundStyle(.tint)
            Text("Get the Tetris model").font(.headline)
            Text("Jeff and the Tetris adapter, 1.4 GB from Hugging Face. It stays on this iPhone and runs on the Neural Engine.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            if game.downloader.busy {
                ProgressView(value: game.downloader.fraction)
                Text(String(format: "%.0f%%  %@", game.downloader.fraction * 100, game.downloader.detail))
                    .font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(2).multilineTextAlignment(.center)
            } else {
                Button { game.downloadModel() } label: { Label("Download", systemImage: "arrow.down.circle.fill").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
            }
            if let e = game.downloader.error { Text(e).font(.caption).foregroundStyle(.red) }
        }
    }

    private var sessionCard: some View {
        Group {
            Image(systemName: "checkmark.seal.fill").font(.system(size: 34)).foregroundStyle(.green)
            Text("Session complete").font(.headline)
            Text("\(GameModel.sessionLimit) games played · \(game.sessionLines) lines")
                .font(.subheadline.monospacedDigit())
            Text("Your games are saved, encrypted, on this iPhone. Export them from the ••• menu.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button { game.startNextSession() } label: { Label("Play \(GameModel.sessionLimit) more games", systemImage: "play.fill").frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent).controlSize(.large)
        }
    }

    private var loadingCard: some View {
        Group {
            ProgressView(value: Double(game.loadStep), total: Double(game.loadTotal))
            Text("Loading Jeff on the Neural Engine").font(.headline)
            Text("\(game.loadStep) of \(game.loadTotal) packages · \(game.status)").font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(2).multilineTextAlignment(.center)
            Text("The first launch specializes the model for this iPhone and can take a few minutes.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
    }

    private func failedCard(_ msg: String) -> some View {
        Group {
            Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 30)).foregroundStyle(.orange)
            Text("Could not start").font(.headline)
            ScrollView { Text(msg).font(.caption.monospaced()).textSelection(.enabled) }.frame(maxHeight: 180)
            Button("Try again") { game.start() }.buttonStyle(.bordered)
        }
    }

    // MARK: playing

    private var playing: some View {
        VStack(spacing: 10) {
            header("Jeff · Tetris adapter · on-device ANE")
            Picker("Mode", selection: $game.mode) { ForEach(GameModel.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }
                .pickerStyle(.segmented).frame(maxWidth: 260)
            HStack(alignment: .top, spacing: 12) {
                BoardView(game: game).aspectRatio(0.5, contentMode: .fit)
                sidebar.frame(width: 150, alignment: .topLeading)
            }
            decisions
            Spacer(minLength: 0)
            controls
        }.padding(.horizontal, 16).padding(.vertical, 8).frame(maxHeight: .infinity, alignment: .top)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("NEXT").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                MiniPiece(name: game.piece, label: "now")
                MiniPiece(name: game.nextPiece, label: "next")
            }
            stat("Lines", "\(game.lines)")
            stat("Pieces", "\(game.pieces)")
            stat("Game", "\(game.sessionGames + 1) of \(GameModel.sessionLimit)  best \(game.bestLines)")
            Divider()
            stat("Decision", String(format: "%.0f ms", game.lastMs))
            stat("Average", String(format: "%.0f ms", game.avgMs))
            stat("Prompt", "\(game.tokens) tok · \(game.calls) ANE calls")
            stat("Confidence", String(format: "%.0f%%", game.confidence * 100))
            stat("Same as El-Tetris", String(format: "%.0f%%  (%d / %d)", game.oracleAgreement * 100, game.agree, game.decided))
            stat("Model's own choice", "\(game.overrides) moves")
        }
    }

    private func stat(_ k: String, _ v: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(k).font(.caption2).foregroundStyle(.secondary)
            Text(v).font(.subheadline.weight(.semibold).monospacedDigit()).minimumScaleFactor(0.6).lineLimit(1)
        }.frame(height: 32, alignment: .topLeading)
    }

    private var decisions: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ANE · \(game.modelName)  ·  calls \(game.callMs.map { String(format: "%.0f", $0) }.joined(separator: " "))ms")
                .font(.system(size: 9).monospaced()).foregroundStyle(.secondary).lineLimit(2, reservesSpace: true)
            ForEach(game.top) { o in
                HStack(spacing: 6) {
                    Text(o.isPick ? "▶" : " ").font(.caption2)
                    Text(o.text).font(.caption.monospaced()).lineLimit(1)
                    if o.isOracle { Text("El-Tetris").font(.system(size: 9).weight(.bold)).padding(.horizontal, 4)
                        .background(Capsule().fill(Color.green.opacity(0.25))) }
                    Spacer()
                    GeometryReader { g in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.secondary.opacity(0.2))
                            Capsule().fill(o.isPick ? Color.accentColor : Color.secondary).frame(width: g.size.width * CGFloat(o.probability))
                        }
                    }.frame(width: 70, height: 6)
                    Text(String(format: "%.0f%%", o.probability * 100)).font(.caption2.monospacedDigit()).frame(width: 34, alignment: .trailing)
                }
            }
        }.frame(maxWidth: .infinity, minHeight: 124, maxHeight: 124, alignment: .topLeading)
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button(game.run == .paused ? "Resume" : "Pause") { game.togglePause() }.buttonStyle(.bordered)
            Button("New game") { game.newGame() }.buttonStyle(.bordered)
            Menu {
                Button("Saved games…", systemImage: "square.and.arrow.up") { showData = true }
                Button("Re-download model", systemImage: "arrow.clockwise") { game.redownloadModel() }
                            } label: { Image(systemName: "ellipsis.circle").font(.title3) }
            Button { recorder.toggle() } label: {
                Label(recorder.isRecording ? "Stop" : "Record", systemImage: recorder.isRecording ? "stop.circle.fill" : "record.circle")
            }.buttonStyle(.borderedProminent).tint(recorder.isRecording ? .red : .accentColor).disabled(!recorder.available)
        }
        .overlay(alignment: .bottom) {
            if let e = recorder.error { Text(e).font(.caption2).foregroundStyle(.red).offset(y: 22) }
        }
    }
}

struct BoardView: View {
    @ObservedObject var game: GameModel
    var body: some View {
        TimelineView(.animation) { tl in
            Canvas { ctx, size in
                let w = size.width / CGFloat(Tetris.width), h = size.height / CGFloat(Tetris.height)
                ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(white: 0.08)))
                func cell(_ r: CGFloat, _ c: CGFloat, _ id: UInt8) {
                    let rect = CGRect(x: c * w + 0.5, y: r * h + 0.5, width: w - 1, height: h - 1)
                    ctx.fill(Path(roundedRect: rect, cornerRadius: 2), with: .color(GameModel.colors[Int(id)]))
                }
                for r in 0..<Tetris.height {
                    for c in 0..<Tetris.width {
                        if game.board[r][c] == 0 {
                            let rect = CGRect(x: CGFloat(c) * w + w / 2 - 0.5, y: CGFloat(r) * h + h / 2 - 0.5, width: 1, height: 1)
                            ctx.fill(Path(rect), with: .color(Color(white: 0.25)))
                        } else { cell(CGFloat(r), CGFloat(c), game.board[r][c]) }
                    }
                }
                if let f = game.fall {
                    let raw: Double = tl.date.timeIntervalSince(f.start) / f.duration
                    let t: Double = min(1.0, max(0.0, raw))
                    let smooth: Double = t * t * (3.0 - 2.0 * t)
                    let drop = CGFloat(smooth * 0.35 + t * 0.65)                       // gentle ease, ends exactly at 1
                    let slide = CGFloat(min(1.0, t * 2.2))                             // lands in its column early
                    let row: CGFloat = CGFloat(f.toRow) * drop
                    let col: CGFloat = CGFloat(f.fromCol) + CGFloat(f.toCol - f.fromCol) * slide
                    for (dr, dc) in f.cells { cell(row + CGFloat(dr), col + CGFloat(dc), f.id) }
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.4), lineWidth: 1))
    }
}

struct MiniPiece: View {
    let name: String, label: String
    var body: some View {
        HStack {
            Canvas { ctx, size in
                let shape = Tetris.pieces[name]![0]
                let s: CGFloat = 12
                for (r, c) in shape {
                    ctx.fill(Path(roundedRect: CGRect(x: CGFloat(c) * s, y: CGFloat(r) * s, width: s - 1, height: s - 1), cornerRadius: 2),
                             with: .color(GameModel.colors[Int(GameModel.pieceId[name]!)]))
                }
            }.frame(width: 52, height: 26)
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct ShareItem: Identifiable { let url: URL; var id: URL { url } }
struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController { UIActivityViewController(activityItems: [url], applicationActivities: nil) }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

struct DataSheet: View {
    @ObservedObject var recorder: GameRecorder
    var export: (ShareItem) -> Void
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Saved games", value: "\(recorder.pending)")
                    Button("Export games (.jsonl)") { if let u = recorder.exportFile() { export(ShareItem(url: u)) } }.disabled(recorder.pending == 0)
                    Button("Delete saved games", role: .destructive) { recorder.deleteLocal() }.disabled(recorder.pending == 0)
                } footer: {
                    Text("Each finished game is saved on this iPhone: for every move, the board, the piece, the model's probabilities and choice, and the El-Tetris label. Games are encrypted with a key held in the Keychain and excluded from backups. Nothing is sent anywhere.")
                }
            }
            .navigationTitle("Saved games").navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
        }
        .presentationDetents([.medium, .large])
    }
}
