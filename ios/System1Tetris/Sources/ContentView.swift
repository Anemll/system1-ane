import SwiftUI

struct ContentView: View {
    @StateObject private var game = GameModel()
    @StateObject private var recorder = ScreenRecorder()
    @State private var showLog = false
    @State private var showData = false
    @State private var downloadTapped = false
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
        .onChange(of: game.downloader.busy) { _, busy in if !busy { downloadTapped = false } }
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
        case .cooling: card { coolingCard }
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

    /// True from the instant of the tap (local state, no network involved) until the download ends.
    private var downloadBusy: Bool { game.downloader.busy || downloadTapped }

    private var downloadCard: some View {
        Group {
            Image(systemName: "cpu").font(.system(size: 34)).foregroundStyle(.tint)
            Text(downloadBusy ? "Step 1 of 2: downloading the model" : "Get the Tetris model").font(.headline)
            if downloadBusy {
                if game.downloader.bytesTotal == 0 {
                    ProgressView()
                    TimelineView(.periodic(from: .now, by: 1)) { ctx in
                        Text("\(game.downloader.phase)...  \(Int(ctx.date.timeIntervalSince(game.downloader.startedAt))) s")
                            .font(.footnote.monospacedDigit()).foregroundStyle(.secondary)
                    }
                } else {
                    ProgressView(value: game.downloader.fraction)
                    Text("\(megabytes(game.downloader.bytesDone)) of \(megabytes(game.downloader.bytesTotal))  ·  \(String(format: "%.0f%%", game.downloader.fraction * 100))"
                         + (game.downloader.speedMBps > 0 ? String(format: "  ·  %.1f MB/s", game.downloader.speedMBps) : ""))
                        .font(.caption.monospacedDigit())
                    Text(game.downloader.detail).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1)
                }
                Text("Keep the app open. Next, step 2 prepares the model for your Neural Engine.")
                    .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                Button("Cancel", role: .cancel) { game.downloader.cancel() }.buttonStyle(.bordered)
            } else {
                Text("Jeff and the Tetris adapter: 1.4 GB, one time, from Hugging Face. Wi-Fi is recommended. Then a few minutes to prepare it for this iPhone's Neural Engine, only on first launch. It all stays on this iPhone.")
                    .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                Button { downloadTapped = true; game.downloadModel() } label: { Label("Download", systemImage: "arrow.down.circle.fill").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                Toggle("Allow cellular data", isOn: Binding(get: { game.downloader.allowCellular }, set: { game.downloader.allowCellular = $0 }))
                    .font(.footnote)
            }
            if let e = game.downloader.error {
                Text(e).font(.caption).foregroundStyle(.red).multilineTextAlignment(.center)
                if game.downloader.needsWifi && !downloadBusy {
                    Button("Use cellular data for this download") { game.downloader.allowCellular = true; downloadTapped = true; game.downloadModel() }
                        .buttonStyle(.bordered).font(.footnote)
                }
            }
        }
    }

    private func megabytes(_ b: Int64) -> String {
        b >= 1_000_000_000 ? String(format: "%.2f GB", Double(b) / 1e9) : String(format: "%.0f MB", Double(b) / 1e6)
    }

    private var loadingCard: some View {
        Group {
            ProgressView(value: Double(game.loadStep), total: Double(game.loadTotal))
            Text("Step 2 of 2: preparing for the Neural Engine").font(.headline).multilineTextAlignment(.center)
            Text("\(game.loadStep) of \(game.loadTotal) parts ready").font(.caption.monospacedDigit())
            TimelineView(.periodic(from: .now, by: 1)) { ctx in
                let el = Int(ctx.date.timeIntervalSince(game.loadStarted))
                let left = game.loadRemaining(at: ctx.date).map { "  ·  about \(max(1, Int(($0 / 60).rounded(.up)))) min left" } ?? "  ·  estimating after the first part"
                Text("Elapsed \(el / 60):\(String(format: "%02d", el % 60))" + left).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text("First launch only: the model is compiled for this iPhone, which takes a few minutes. Later launches take seconds. Keep the app open; the game starts by itself.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button("Cancel", role: .cancel) { game.cancelLoading() }.buttonStyle(.bordered)
        }
    }

    private func failedCard(_ msg: String) -> some View {
        Group {
            Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 30)).foregroundStyle(.orange)
            Text("Could not start").font(.headline)
            Text("The model could not be prepared for the Neural Engine. Try again first. If it fails again, restart the iPhone, then reopen the app. If that does not help, tap Recompile (no download). Re-download is in the ... menu as a last resort.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            DisclosureGroup("How to restart the iPhone") {
                Text("Press and release Volume Up, press and release Volume Down, then hold the Side button until the Apple logo appears. Or open Settings > General > Shut Down, then turn it on again.")
                    .font(.footnote).foregroundStyle(.secondary)
            }.font(.footnote)
            ScrollView { Text(msg).font(.caption.monospaced()).textSelection(.enabled) }.frame(maxHeight: 80)
            HStack {
                Button("Try again") { game.start() }.buttonStyle(.bordered)
                Button("Recompile") { game.recompileModel() }.buttonStyle(.bordered)
            }
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

    private var coolingCard: some View {
        Group {
            Image(systemName: "thermometer.high").font(.system(size: 34)).foregroundStyle(.orange)
            Text("Cooling down").font(.headline)
            Text("Your iPhone is hot, so play is paused to protect it. It resumes by itself when the phone has cooled.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            ProgressView()
        }
    }

    // MARK: playing

    private var playing: some View {
        VStack(spacing: 10) {
            header("Jeff · Tetris adapter · on-device ANE")
            Picker("Mode", selection: $game.mode) { ForEach(GameModel.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }
                .pickerStyle(.segmented).frame(maxWidth: 260)
            thermalLine   // always visible: the iPhone does not expose degrees, only this four-level state
            HStack(alignment: .top, spacing: 12) {
                BoardView(game: game).aspectRatio(0.5, contentMode: .fit)
                sidebar.frame(width: 150, alignment: .topLeading)
            }
            decisions
            Spacer(minLength: 0)
            controls
        }.padding(.horizontal, 16).padding(.vertical, 8).frame(maxHeight: .infinity, alignment: .top)
    }

    private var thermalState: (name: String, color: Color) {
        switch game.thermal {
        case .nominal: return ("Nominal", .green)
        case .fair: return ("Fair: slowing down", .yellow)
        case .serious: return ("Hot: playing slowly", .orange)
        case .critical: return ("Too hot: paused", .red)
        @unknown default: return ("Unknown", .gray)
        }
    }

    private var thermalLine: some View {
        let t = thermalState
        let ratio = game.aneColdMs > 0 ? game.aneNowMs / game.aneColdMs : 1
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        return HStack(spacing: 6) {
            Image(systemName: "thermometer.medium").foregroundStyle(t.color)
            Text("Thermal \(t.name)").foregroundStyle(t.color)
            if game.aneNowMs > 0 {
                Text(String(format: "· ANE %.0f ms (cold %.0f, x%.1f)", game.aneNowMs, game.aneColdMs, ratio))
                    .foregroundStyle(ratio > 1.3 ? Color.orange : Color.secondary)
            }
            if lowPower { Text("· Low Power").foregroundStyle(.yellow) }
        }
        .font(.caption2.monospacedDigit()).lineLimit(1).minimumScaleFactor(0.7).frame(height: 16)
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
                Button(game.explore ? "Explore: on (tap to turn off)" : "Explore: sample moves", systemImage: "dice") { game.explore.toggle() }
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
