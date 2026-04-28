import AVFoundation
import Foundation

final class AudioPlaybackQueue {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let lock = NSLock()

    private var queue: [AVAudioPCMBuffer] = []
    private var isScheduling = false
    private(set) var isPlaying = false

    var onSegmentFinished: (() -> Void)?

    init() {
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: nil)
        try? engine.start()
    }

    func ensureRunning() throws {
        if !engine.isRunning {
            try engine.start()
        }
    }

    func clear() {
        lock.lock()
        queue.removeAll()
        isScheduling = false
        lock.unlock()
        player.stop()
        isPlaying = false
    }

    func enqueue(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        queue.append(buffer)
        let shouldSchedule = !isScheduling
        if shouldSchedule { isScheduling = true }
        lock.unlock()

        if shouldSchedule {
            scheduleNextIfNeeded()
        }
    }

    func play() {
        player.play()
        isPlaying = true
    }

    func pause() {
        player.pause()
        isPlaying = false
    }

    private func scheduleNextIfNeeded() {
        lock.lock()
        guard !queue.isEmpty else {
            isScheduling = false
            lock.unlock()
            return
        }
        let next = queue.removeFirst()
        lock.unlock()

        player.scheduleBuffer(next, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            DispatchQueue.main.async {
                self.onSegmentFinished?()
                self.scheduleNextIfNeeded()
            }
        }

        if !isPlaying {
            player.play()
            isPlaying = true
        }
    }
}
