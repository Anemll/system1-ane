// Downloads the Tetris build (coreai/ + model/) from Hugging Face into Documents/tetris, resumable per file.
import Foundation

@MainActor
final class ModelDownloader: ObservableObject {
    static let repo = "anemll/jeff-ane-tetris-coreai"
    @Published var fraction = 0.0
    @Published var detail = ""
    @Published var busy = false
    @Published var error: String?

    private struct Entry: Decodable { let path: String; let size: Int?; let type: String }

    static func isInstalled(_ root: URL) -> Bool {
        ["coreai/manifest.json", "model/tokenizer.json", "model/embed_tokens_fp16.npy", "model/readout.safetensors"]
            .allSatisfy { FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path) }
            && ["chunk_L00-03", "chunk_L20-23", "head_readout"].allSatisfy {
                FileManager.default.fileExists(atPath: root.appendingPathComponent("coreai/\($0).aimodel/main.mlirb").path)
            }
    }

    func download(to root: URL) async -> Bool {
        busy = true; error = nil; fraction = 0
        defer { busy = false }
        do {
            let listURL = URL(string: "https://huggingface.co/api/models/\(Self.repo)/tree/main?recursive=1")!
            let (data, _) = try await URLSession.shared.data(from: listURL)
            let files = try JSONDecoder().decode([Entry].self, from: data)
                .filter { $0.type == "file" && ($0.path.hasPrefix("coreai/") || $0.path.hasPrefix("model/")) }
            let total = Double(files.reduce(0) { $0 + ($1.size ?? 0) })
            var done = 0.0
            for f in files {
                let dest = root.appendingPathComponent(f.path)
                let size = f.size ?? 0
                if let a = try? FileManager.default.attributesOfItem(atPath: dest.path)[.size] as? Int, a == size {
                    done += Double(size); fraction = done / total; continue
                }
                detail = "\(f.path)  (\(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)))"
                try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
                let url = URL(string: "https://huggingface.co/\(Self.repo)/resolve/main/\(f.path)")!
                let base = done
                let tmp = try await fetch(url) { [weak self] got in
                    Task { @MainActor in self?.fraction = (base + Double(got)) / total }
                }
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.moveItem(at: tmp, to: dest)
                done += Double(size); fraction = done / total
            }
            detail = "Done"
            return Self.isInstalled(root)
        } catch {
            self.error = "\(error.localizedDescription)"
            return false
        }
    }

    private func fetch(_ url: URL, progress: @escaping @Sendable (Int64) -> Void) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            let task = URLSession.shared.downloadTask(with: url) { tmp, resp, err in
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
