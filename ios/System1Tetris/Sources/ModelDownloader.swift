// Downloads the Tetris build (coreai/ + model/) from Hugging Face into Documents/tetris, resumable per file.
import Foundation

@MainActor
final class ModelDownloader: ObservableObject {
    static let repo = "anemll/jeff-ane-tetris-coreai"
    @Published var fraction = 0.0
    @Published var detail = ""
    @Published var busy = false
    @Published var bytesDone: Int64 = 0
    @Published var bytesTotal: Int64 = 0      // 0 until the file list has been fetched ("Starting...")
    @Published var speedMBps = 0.0
    private(set) var startedAt = Date()
    @Published var allowCellular = false   // 1.4 GB: Wi-Fi only unless the user allows cellular data
    @Published var phase = ""              // what the downloader is doing right now, for the Connecting screen
    @Published var needsWifi = false       // the last attempt was refused because only cellular was available

    private func makeSession() -> URLSession {
        let c = URLSessionConfiguration.default
        c.allowsExpensiveNetworkAccess = allowCellular      // cellular, personal hotspot
        c.allowsConstrainedNetworkAccess = allowCellular    // Low Data Mode
        c.timeoutIntervalForRequest = 60
        return URLSession(configuration: c)
    }

    /// Plain-language message for a failed download.
    static func friendly(_ e: Error) -> String {
        guard let u = e as? URLError else { return e.localizedDescription }
        switch u.code {
        case .notConnectedToInternet, .dataNotAllowed:
            return u.networkUnavailableReason == .expensive || u.networkUnavailableReason == .constrained
                ? "The model is 1.4 GB, so it downloads over Wi-Fi only. Connect to Wi-Fi, or allow cellular data."
                : "No internet connection. Check your network and tap Download again."
        case .timedOut: return "The connection timed out. Check your network and tap Download to try again."
        case .networkConnectionLost: return "The connection was lost. Tap Download to continue; files already finished are kept."
        case .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed: return "Could not reach Hugging Face. Check your network and try again."
        default: return u.localizedDescription
        }
    }

    /// Called synchronously from the button, so the card changes the instant it is tapped.
    func begin() { cancelled = false; busy = true; error = nil; needsWifi = false; fraction = 0; bytesDone = 0; bytesTotal = 0; speedMBps = 0; detail = "Connecting to Hugging Face..."; phase = "Connecting"; startedAt = Date(); appLog("download: begin") }
    @Published var error: String?
    private var session: URLSession?
    private var cancelled = false

    /// Stops the download now; files already finished are kept, so the next Download continues.
    func cancel() { cancelled = true; session?.invalidateAndCancel() }

    private struct Entry: Decodable { let path: String; let size: Int?; let type: String }

    static func isInstalled(_ root: URL) -> Bool {
        ["coreai/manifest.json", "model/tokenizer.json", "model/embed_tokens_fp16.npy", "model/readout.safetensors"]
            .allSatisfy { FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path) }
            && ["chunk_L00-03", "chunk_L20-23", "head_readout"].allSatisfy {
                FileManager.default.fileExists(atPath: root.appendingPathComponent("coreai/\($0).aimodel/main.mlirb").path)
            }
    }

    private var running = false

    func download(to root: URL) async -> Bool {
        guard !running else { return false }
        running = true; defer { running = false }
        if !busy { begin() }
        defer { busy = false }
        do {
            let listURL = URL(string: "https://huggingface.co/api/models/\(Self.repo)/tree/main?recursive=1")!
            let session = makeSession()
            self.session = session
            var data = Data()
            for attempt in 1...3 {      // a stalled first request is retried quickly instead of waiting out the long timeout
                phase = attempt == 1 ? "Contacting Hugging Face" : "Contacting Hugging Face (try \(attempt) of 3)"
                detail = phase
                let t0 = Date()
                var req = URLRequest(url: listURL); req.timeoutInterval = 10
                do { data = try await session.data(for: req).0; appLog("download: file list in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s (try \(attempt))"); break }
                catch { appLog("download: file list try \(attempt) failed after \(String(format: "%.1f", Date().timeIntervalSince(t0)))s: \(error.localizedDescription)"); if attempt == 3 { throw error } }
            }
            phase = "Preparing the download"
            let files = try JSONDecoder().decode([Entry].self, from: data)
                .filter { $0.type == "file" && ($0.path.hasPrefix("coreai/") || $0.path.hasPrefix("model/")) }
            let total = Double(files.reduce(0) { $0 + ($1.size ?? 0) })
            bytesTotal = Int64(total)
            var done = 0.0
            for f in files {
                let dest = root.appendingPathComponent(f.path)
                let size = f.size ?? 0
                if let a = try? FileManager.default.attributesOfItem(atPath: dest.path)[.size] as? Int, a == size {
                    done += Double(size); fraction = done / total; bytesDone = Int64(done); bytesDone = Int64(done); continue
                }
                detail = "\(f.path)  (\(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)))"
                appLog("download: start \(f.path)")
                try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
                let url = URL(string: "https://huggingface.co/\(Self.repo)/resolve/main/\(f.path)")!
                let base = done
                let tmp = try await fetch(url, session: session) { [weak self] got in
                    Task { @MainActor in
                        guard let self else { return }
                        self.fraction = (base + Double(got)) / total
                        self.bytesDone = Int64(base) + got
                        let t = Date().timeIntervalSince(self.startedAt)
                        if t > 1 { self.speedMBps = Double(self.bytesDone) / t / 1_000_000 }
                    }
                }
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.moveItem(at: tmp, to: dest)
                done += Double(size); fraction = done / total
            }
            detail = "Done"
            return Self.isInstalled(root)
        } catch {
            if cancelled { detail = "Cancelled"; return false }
            self.error = Self.friendly(error)
            if let u = error as? URLError, u.networkUnavailableReason == .expensive || u.networkUnavailableReason == .constrained { needsWifi = true }
            return false
        }
    }

    private func fetch(_ url: URL, session: URLSession, progress: @escaping @Sendable (Int64) -> Void) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            let task = session.downloadTask(with: url) { tmp, resp, err in
                if let err { cont.resume(throwing: err); return }
                guard let tmp, let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                    cont.resume(throwing: URLError(.badServerResponse)); return
                }
                let keep = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                do { try FileManager.default.moveItem(at: tmp, to: keep); cont.resume(returning: keep) }
                catch { cont.resume(throwing: error) }
            }
            let timer = Timer(timeInterval: 0.3, repeats: true) { _ in progress(task.countOfBytesReceived) }
            RunLoop.main.add(timer, forMode: .common)
            DispatchQueue.global().async {
                while task.state == .running || task.state == .suspended { Thread.sleep(forTimeInterval: 0.3) }
                timer.invalidate()
            }
            task.resume()
        }
    }
}
