// In-app screen recording with ReplayKit; the finished movie is offered through the share sheet / Photos.
import Foundation
import ReplayKit

@MainActor
final class ScreenRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var lastFile: URL?
    @Published var error: String?

    var available: Bool { RPScreenRecorder.shared().isAvailable }

    func toggle() {
        isRecording ? stop() : start()
    }

    private func start() {
        let r = RPScreenRecorder.shared()
        r.isMicrophoneEnabled = false
        r.startRecording { [weak self] err in
            Task { @MainActor in
                if let err { self?.error = err.localizedDescription } else { self?.isRecording = true; self?.error = nil }
            }
        }
    }

    private func stop() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("System1ANE-Tetris-\(Int(Date().timeIntervalSince1970)).mp4")
        RPScreenRecorder.shared().stopRecording(withOutput: url) { [weak self] err in
            Task { @MainActor in
                self?.isRecording = false
                if let err { self?.error = err.localizedDescription } else { self?.lastFile = url }
            }
        }
    }
}
