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
        VStack(alignment: .trailing, spacing: 2) {
            Text("System 1 ANE").font(.title2.weight(.bold)).accessibilityAddTraits(.isHeader)
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
            ViewThatFits(in: .vertical) {
                VStack(spacing: 14) { content() }
                    .padding(20).frame(maxWidth: 340)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
                ScrollView {
                    VStack(spacing: 14) { content() }
                        .padding(20).frame(maxWidth: 340)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
                        .frame(maxWidth: .infinity)
                }
            }
            .padding(12)
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
        GameplayScreen(game: game, recorder: recorder, showData: $showData)
    }
}

/// Hinge capability belongs to the view's device, not its current compact/regular size.
@available(iOS 27.1, *)
private struct HingeCapabilityObserver: UIViewRepresentable {
    @Binding var hasHinge: Bool
    @Binding var isHingeClosed: Bool

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let detected = $hasHinge
        let closed = $isHingeClosed
        view.addInteraction(UIHingeInteraction { _, update in
            // A nil update can occur while moving between display hierarchies.
            // Retain positive detection while this screen lives, including across folding.
            guard let hinge = update.hinge else { return }
            let isClosed: Bool?
            switch hinge.status {
            case .closed: isClosed = true
            case .partiallyOpen, .fullyOpen: isClosed = false
            case .unknown: isClosed = nil
            @unknown default: isClosed = nil
            }
            Task { @MainActor in
                if !detected.wrappedValue { detected.wrappedValue = true }
                if let isClosed, closed.wrappedValue != isClosed {
                    closed.wrappedValue = isClosed
                }
            }
        })
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

private struct GameplayScreen: View {
    @ObservedObject var game: GameModel
    @ObservedObject var recorder: ScreenRecorder
    @Binding var showData: Bool
    @State private var hasHinge = false
    @State private var isHingeClosed = false
    @AppStorage("gameplayLayout") private var layoutChoice = "Automatic"

    private var duo: Bool {
        layoutChoice == "Duo" || (layoutChoice == "Automatic" && hasHinge)
    }

    var body: some View {
        Group {
            if duo {
                GameplayLayout(isOuterDisplay: isHingeClosed) { landscape in
                    GameplayHeader(game: game, showMode: !landscape, showThermal: !landscape, thermalStatus: landscape ? thermalState.name : nil, thermalColor: thermalState.color) { thermalLine }
                } info: { landscape, compactLandscape in
                    VStack(alignment: .trailing, spacing: 12) {
                        if compactLandscape {
                            Picker("Mode", selection: $game.mode) {
                                ForEach(GameModel.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                            }
                            .pickerStyle(.segmented)
                            HStack(spacing: 16) {
                                MiniPiece(name: game.piece, label: "Now", labelAbove: true)
                                MiniPiece(name: game.nextPiece, label: "Next", labelAbove: true)
                            }
                        }
                        if landscape {
                            if game.aneNowMs > 0 {
                                Text(String(format: "ANE %.0f ms (cold %.0f, x%.1f)", game.aneNowMs, game.aneColdMs, game.aneColdMs > 0 ? game.aneNowMs / game.aneColdMs : 1))
                                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                            }
                            if ProcessInfo.processInfo.isLowPowerModeEnabled {
                                Text("Low Power").font(.caption2)
                            }
                        }
                        sidebar
                        if landscape {
                            VStack(alignment: .trailing, spacing: 4) {
                                Text(game.modelName).font(.caption2.monospaced())
                                Text("ANE calls: \(game.callMs.map { String(format: "%.0f", $0) }.joined(separator: ", ")) ms")
                                    .font(.caption2.monospaced())
                            }
                            .foregroundStyle(.secondary)
                            if let error = recorder.error { Text(error).font(.caption) }
                        }
                    }
                    .multilineTextAlignment(.trailing)
                } board: {
                    BoardView(game: game)
                } decisions: { showSummary, compactLandscape in
                    VStack(alignment: .leading, spacing: 4) {
                        if showSummary { decisionSummary }
                        if showSummary { decisionBars } else { compactDecisionBars(narrow: compactLandscape) }
                        if showSummary, let error = recorder.error { Text(error).font(.caption) }
                    }
                } controls: { landscape, compactLandscape in
                    VStack(spacing: 16) {
                        if !compactLandscape {
                            if landscape { VerticalModeSelector(mode: $game.mode) }
                            MiniPiece(name: game.piece, label: "Now", labelAbove: true)
                            MiniPiece(name: game.nextPiece, label: "Next", labelAbove: true)
                        }
                        GameplayControls(game: game, recorder: recorder, showData: $showData, vertical: true, layoutChoice: $layoutChoice)
                    }
                }
            } else {
                GeometryReader { geometry in
                    ScrollView {
                        VStack(spacing: 10) {
                            GameplayHeader(game: game) { thermalLine }
                            let boardWidth = min(max(0, (geometry.size.width - 44) * 0.62), geometry.size.height * 0.34)
                            HStack(alignment: .top, spacing: 12) {
                                BoardView(game: game)
                                    .frame(width: boardWidth, height: boardWidth * 2)
                                ScrollView {
                                    VStack(alignment: .leading, spacing: 10) {
                                        Text("NEXT").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                                        MiniPiece(name: game.piece, label: "now")
                                        MiniPiece(name: game.nextPiece, label: "next")
                                        sidebar
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .frame(height: boardWidth * 2)
                            }
                            decisions
                            Spacer(minLength: 12)
                            ViewThatFits(in: .horizontal) {
                                controls
                                ScrollView(.horizontal) { controls }
                            }
                            if let error = recorder.error { Text(error).font(.caption) }
                        }
                        .padding(.horizontal, 16).padding(.vertical, 8)
                        .frame(minHeight: geometry.size.height, alignment: .top)
                    }
                }
            }
        }
        .background {
            if #available(iOS 27.1, *) {
                HingeCapabilityObserver(hasHinge: $hasHinge, isHingeClosed: $isHingeClosed)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
    }

    private var layoutPicker: some View {
        Picker("Layout", selection: $layoutChoice) {
            Text("Automatic").tag("Automatic")
            Text("iPhone").tag("iPhone")
            Text("Duo").tag("Duo")
        }
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
        let thermal = thermalState
        let ratio = game.aneColdMs > 0 ? game.aneNowMs / game.aneColdMs : 1
        return ViewThatFits(in: .horizontal) {
            HStack(spacing: 6) {
                Label {
                    Text("Thermal \(thermal.name)")
                } icon: {
                    Image(systemName: "thermometer.medium").foregroundStyle(thermal.color)
                }
                if game.aneNowMs > 0 {
                    Text(String(format: "· ANE %.0f ms (cold %.0f, x%.1f)", game.aneNowMs, game.aneColdMs, ratio))
                        .foregroundStyle(.secondary)
                }
                if ProcessInfo.processInfo.isLowPowerModeEnabled { Text("· Low Power") }
            }
            VStack(spacing: 4) {
                Label {
                    Text("Thermal \(thermal.name)")
                } icon: {
                    Image(systemName: "thermometer.medium").foregroundStyle(thermal.color)
                }
                if game.aneNowMs > 0 {
                    Text(String(format: "ANE %.0f ms (cold %.0f, x%.1f)", game.aneNowMs, game.aneColdMs, ratio))
                }
                if ProcessInfo.processInfo.isLowPowerModeEnabled { Text("Low Power") }
            }
        }
        .font(.caption2.monospacedDigit())
        .multilineTextAlignment(.center)
    }

    private var sidebar: some View {
        VStack(alignment: duo ? .trailing : .leading, spacing: 10) {
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
        VStack(alignment: duo ? .trailing : .leading, spacing: 0) {
            Text(k).font(.caption2).foregroundStyle(.secondary)
            Text(v).font(.subheadline.weight(.semibold).monospacedDigit()).fixedSize(horizontal: false, vertical: true)
        }.accessibilityElement(children: .combine)
    }

    private var decisionSummary: some View {
            Text("ANE · \(game.modelName) · calls \(game.callMs.map { String(format: "%.0f", $0) }.joined(separator: " ")) ms")
                .font(.caption2.monospaced()).foregroundStyle(.secondary)
    }

    private var decisions: some View {
        VStack(alignment: .leading, spacing: 4) {
            decisionSummary
            decisionBars
        }
    }

    private var decisionBars: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(game.top) { option in
                HStack(spacing: 6) {
                    Text(option.isPick ? "▶" : " ").accessibilityHidden(true)
                    Text(option.text).frame(maxWidth: .infinity, alignment: .leading)
                    if option.isOracle { Text("El-Tetris").font(.caption2.bold()) }
                    ProgressView(value: option.probability)
                        .frame(width: 70).accessibilityHidden(true)
                    Text(String(format: "%.0f%%", option.probability * 100))
                        .monospacedDigit()
                }
                .font(.caption)
                .accessibilityElement(children: .combine)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }
    private func compactDecisionBars(narrow: Bool) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(game.top) { option in
                HStack(spacing: narrow ? 3 : 6) {
                    if !narrow {
                        Text(option.isPick ? "▶" : " ")
                            .frame(width: 12).accessibilityHidden(true)
                    }
                    ViewThatFits(in: .horizontal) {
                        Text(option.text).fixedSize()
                        Text(String(option.text.split(separator: " ").first ?? ""))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    Image(systemName: "checkmark")
                        .opacity(option.isOracle ? 1 : 0)
                        .accessibilityHidden(true)
                    ProgressView(value: option.probability)
                        .frame(width: narrow ? 24 : 56).accessibilityHidden(true)
                    Text(String(format: "%.0f%%", option.probability * 100))
                        .monospacedDigit().frame(minWidth: narrow ? 28 : 32, alignment: .trailing)
                }
                .font(narrow ? .caption2.monospaced() : .caption.monospaced())
                .fontWeight(option.isPick ? .bold : .regular)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(option.isPick ? "Selected. " : "")\(option.text). \(option.isOracle ? "El-Tetris choice. " : "")\(Int((option.probability * 100).rounded())) percent")
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var controls: some View {
        GameplayControls(
            game: game,
            recorder: recorder,
            showData: $showData,
            vertical: false,
            layoutChoice: $layoutChoice
        )
    }
}

private struct GameplayHeader<Thermal: View>: View {
    @ObservedObject var game: GameModel
    var showMode = true
    var showThermal = true
    var thermalStatus: String? = nil
    var thermalColor: Color = .secondary
    @ViewBuilder var thermal: Thermal

    var body: some View {
        VStack(spacing: 10) {
            VStack(spacing: 2) {
                Text("System 1 ANE").font(.title2.bold()).accessibilityAddTraits(.isHeader)
                if let thermalStatus {
                    Text("Jeff · Tetris adapter · on-device ANE · \(Text(Image(systemName: "thermometer.medium")).foregroundColor(thermalColor)) Thermal \(thermalStatus)")
                        .font(.footnote).foregroundStyle(.secondary)
                        .accessibilityLabel("Jeff · Tetris adapter · on-device ANE · Thermal \(thermalStatus)")
                } else {
                    Text("Jeff · Tetris adapter · on-device ANE")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .multilineTextAlignment(.center)
            if showMode {
                Picker("Mode", selection: $game.mode) {
                ForEach(GameModel.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
                .pickerStyle(.segmented).frame(maxWidth: 260)
            }
            if showThermal { thermal }
        }
        .frame(maxWidth: .infinity)
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// Uses the full window width with right-side controls below system corner UI.
private struct GameplayLayout<Header: View, Info: View, Board: View, Decisions: View, Controls: View>: View {
    let isOuterDisplay: Bool
    @ViewBuilder var header: (Bool) -> Header
    @ViewBuilder var info: (Bool, Bool) -> Info
    @ViewBuilder var board: Board
    @ViewBuilder var decisions: (Bool, Bool) -> Decisions
    @ViewBuilder var controls: (Bool, Bool) -> Controls
    @ScaledMetric(relativeTo: .body) private var controlSymbolSize = 20
    @ScaledMetric(relativeTo: .caption) private var decisionPanelHeight = 84
    private var controlSize: CGFloat { max(44, controlSymbolSize + 12) }

    var body: some View {
        GeometryReader { safeGeometry in
            GeometryReader { geometry in
                let width = max(0, geometry.size.width - 24)
                let height = max(0, geometry.size.height - 24)
                let leadingInset = safeGeometry.safeAreaInsets.leading
                let controlsTop = geometry.containerCornerInsets.topTrailing.height
                let controlsBottom = geometry.containerCornerInsets.bottomTrailing.height
                let landscape = geometry.size.width > geometry.size.height
                // Only the folded outer display moves previews and mode to the left.
                // Window height and text size must not change the full-display arrangement.
                let compactLandscape = landscape && isOuterDisplay
                let railWidth = landscape && !compactLandscape ? max(controlSize + 12, 88) : controlSize + 12
                let controlSpace = railWidth + 16
                // Reserve a side rail on every display size; no bottom control shelf is needed.
                let contentWidth = max(0, width - leadingInset - controlSpace)
                let content = VStack(spacing: 10) {
                    header(false)
                    GeometryReader { playArea in
                        let boardWidth = min(contentWidth * 0.56, playArea.size.height / 2)
                        HStack(alignment: .top, spacing: 12) {
                            ScrollView {
                                info(false, false).frame(maxWidth: .infinity, alignment: .trailing)
                            }
                            .frame(width: max(0, contentWidth - boardWidth - 12))
                            board
                                .frame(width: boardWidth, height: boardWidth * 2)
                                .frame(height: playArea.size.height, alignment: .center)
                        }
                    }
                    .frame(minHeight: 180)
                    ScrollView { decisions(true, false) }
                        .frame(minHeight: 80, maxHeight: max(80, min(160, height * 0.25)))
                }
                Group {
                    if landscape {
                        VStack(spacing: 12) {
                            header(true)
                            HStack(alignment: .top, spacing: 16) {
                                ScrollView {
                                    info(true, compactLandscape).frame(maxWidth: .infinity, alignment: .trailing)
                                }
                                .frame(width: max(0, contentWidth * 0.38 - 8))
                                GeometryReader { column in
                                    let sectionWidth = min(column.size.width, 400)
                                    let footerHeight = min(decisionPanelHeight, column.size.height * 0.36)
                                    let boardWidth = max(0, min(sectionWidth, (column.size.height - footerHeight - 10) / 2))
                                    VStack(spacing: 10) {
                                        board
                                            .frame(width: boardWidth, height: boardWidth * 2)
                                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                                        ScrollView { decisions(false, compactLandscape) }
                                            .frame(height: footerHeight)
                                    }
                                    .frame(width: compactLandscape ? boardWidth : sectionWidth, height: column.size.height)
                                    .frame(maxWidth: .infinity)
                                }
                            }
                        }
                    } else {
                        ViewThatFits(in: .vertical) {
                            content
                            ScrollView {
                                content.fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
                .frame(width: contentWidth, height: height, alignment: .topTrailing)
                .padding(.leading, leadingInset)
                .frame(width: width, alignment: .leading)
                .overlay(alignment: .topTrailing) {
                    ScrollView(.vertical) {
                        controls(landscape, compactLandscape)
                    }
                    .scrollBounceBehavior(.basedOnSize)
                    .frame(width: railWidth, height: max(0, height - controlsTop - controlsBottom))
                    .padding(.top, controlsTop)
                }
                .padding(12)
            }
            .ignoresSafeArea(.container, edges: .horizontal)
        }
    }
}

private struct VerticalModeSelector: View {
    @Binding var mode: GameModel.Mode

    var body: some View {
        VStack(spacing: 4) {
            ForEach(GameModel.Mode.allCases, id: \.self) { option in
                Button { mode = option } label: {
                    VStack(spacing: 2) {
                        Text(option.rawValue)
                            .multilineTextAlignment(.center)
                        if mode == option {
                            Image(systemName: "checkmark").accessibilityHidden(true)
                        }
                    }
                    .font(.caption.weight(.semibold))
                    .frame(maxWidth: .infinity, minHeight: 44)
                    .background(mode == option ? Color.accentColor.opacity(0.2) : .clear,
                                in: RoundedRectangle(cornerRadius: 12))
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(mode == option ? [.isSelected] : [])
            }
        }
        .padding(4)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Playback mode")
    }
}

private struct GameplayControls: View {
    @ObservedObject var game: GameModel
    @ObservedObject var recorder: ScreenRecorder
    @Binding var showData: Bool
    let vertical: Bool
    @Binding var layoutChoice: String
    @ScaledMetric(relativeTo: .body) private var controlSymbolSize = 20
    private var controlSize: CGFloat { max(44, controlSymbolSize + 12) }
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast

    var body: some View {
        let layout = vertical ? AnyLayout(VStackLayout(spacing: 4)) : AnyLayout(HStackLayout(spacing: 4))
        layout {
            Button { game.togglePause() } label: {
                Label(game.run == .paused ? "Resume" : "Pause", systemImage: game.run == .paused ? "play.fill" : "pause.fill")
                    .frame(minWidth: controlSize, minHeight: controlSize)
            }
            Button { game.newGame() } label: {
                Label("New game", systemImage: "arrow.clockwise").frame(minWidth: controlSize, minHeight: controlSize)
            }
            Button { recorder.toggle() } label: {
                Label(recorder.isRecording ? "Stop recording" : "Record screen", systemImage: recorder.isRecording ? "stop.circle.fill" : "record.circle")
                    .frame(minWidth: controlSize, minHeight: controlSize)
            }
            .disabled(!recorder.available)
            Menu {
                Picker("Layout", selection: $layoutChoice) {
                    Text("Automatic").tag("Automatic")
                    Text("iPhone").tag("iPhone")
                    Text("Duo").tag("Duo")
                }
                Button("Saved games…", systemImage: "square.and.arrow.up") { showData = true }
                Button(game.explore ? "Explore: on (tap to turn off)" : "Explore: sample moves", systemImage: "dice") { game.explore.toggle() }
                Button("Re-download model", systemImage: "arrow.clockwise") { game.redownloadModel() }
            } label: {
                Label("More options", systemImage: "ellipsis").frame(minWidth: controlSize, minHeight: controlSize)
            }
        }
        .labelStyle(.iconOnly)
        .font(.system(size: controlSymbolSize, weight: .semibold))
        .buttonStyle(.plain)
        .foregroundStyle(.primary)
        .padding(6)
        .modifier(GameplayControlSurface(opaque: reduceTransparency || contrast == .increased))
    }
}

private struct GameplayControlSurface: ViewModifier {
    let opaque: Bool

    func body(content: Content) -> some View {
        if opaque {
            content.background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 24))
                .overlay(RoundedRectangle(cornerRadius: 24).stroke(.primary, lineWidth: 1))
        } else {
            content.glassEffect(.regular, in: RoundedRectangle(cornerRadius: 24))
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
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Tetris board")
        .accessibilityValue("\(game.lines) lines, \(game.pieces) pieces placed. Current piece \(game.piece), next piece \(game.nextPiece).")
    }
}

struct MiniPiece: View {
    let name: String, label: String
    var labelAbove = false

    var body: some View {
        let layout = labelAbove ? AnyLayout(VStackLayout(spacing: 6)) : AnyLayout(HStackLayout())
        layout {
            if labelAbove {
                Text(label).font(.caption2.weight(.semibold))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Canvas { ctx, size in
                let shape = Tetris.pieces[name]![0]
                let s: CGFloat = 12
                for (r, c) in shape {
                    ctx.fill(Path(roundedRect: CGRect(x: CGFloat(c) * s, y: CGFloat(r) * s, width: s - 1, height: s - 1), cornerRadius: 2),
                             with: .color(GameModel.colors[Int(GameModel.pieceId[name]!)]))
                }
            }.frame(width: 52, height: 26)
            if !labelAbove {
                Text(label).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label) piece, \(name)")
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
