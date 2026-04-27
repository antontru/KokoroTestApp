import AVFoundation
import Foundation
import KokoroSwift
import MLX
import MLXUtilsLibrary
import NaturalLanguage
import SwiftUI

struct VoiceOption: Identifiable, Hashable {
    enum LocaleGroup: String {
        case us = "American English"
        case gb = "British English"
    }

    let id: String
    let name: String
    let localeGroup: LocaleGroup
    let descriptor: String
}

struct DebugLogEntry: Identifiable {
    enum Level {
        case info
        case warning
        case error
    }

    let id = UUID()
    let timestamp: Date
    let level: Level
    let message: String
}

final class TestAppModel: ObservableObject {
    enum PlaybackPhase {
        case idle
        case generating
        case playing
        case paused
        case done
    }

    @Published var voiceOptions: [VoiceOption] = []
    @Published var selectedVoiceID: String = ""
    @Published var speed: Double = 1.0

    @Published var sentences: [String] = []
    @Published var currentSentenceIndex: Int?
    @Published var playbackPhase: PlaybackPhase = .idle

    @Published var debugLogs: [DebugLogEntry] = []

    var usVoices: [VoiceOption] {
        voiceOptions.filter { $0.localeGroup == .us }
    }

    var gbVoices: [VoiceOption] {
        voiceOptions.filter { $0.localeGroup == .gb }
    }

    private let kokoroTTSEngine: KokoroTTS
    private let voices: [String: MLXArray]

    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let timePitchNode = AVAudioUnitTimePitch()

    private let synthesisQueue = DispatchQueue(label: "kokoro.tts.synthesis")
    private var playbackToken = UUID()
    private var prefetchToken = UUID()

    private var preparedNextBuffer: AVAudioPCMBuffer?
    private var preparedNextIndex: Int?
    private var preparedVoiceID: String?

    private var loadedText: String = ""
    private let defaults = UserDefaults.standard

    private let selectedVoiceDefaultsKey = "kokoro.selectedVoiceID"
    private let speedDefaultsKey = "kokoro.speed"

    init() {
        let modelPath = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")!
        kokoroTTSEngine = KokoroTTS(modelPath: modelPath)

        let voiceFilePath = Bundle.main.url(forResource: "voices", withExtension: "npz")!
        voices = NpyzReader.read(fileFromPath: voiceFilePath) ?? [:]

        configureVoices()
        configureAudioEngine()
        configureAudioSession()
        configureSettings()
        configureInterruptions()

        log("Model initialized with \(voiceOptions.count) voices")
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    var hasLoadedText: Bool {
        !sentences.isEmpty
    }

    func togglePlay(using text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if loadedText != text || sentences.isEmpty {
            loadText(text)
        }

        guard !sentences.isEmpty else { return }

        switch playbackPhase {
        case .playing:
            pausePlayback()
        case .paused:
            resumePlayback()
        case .idle:
            startPlayback(at: currentSentenceIndex ?? 0)
        case .done:
            startPlayback(at: 0)
        case .generating:
            break
        }
    }

    func stopPlayback() {
        invalidateSynthesisTokens()
        playerNode.stop()
        currentSentenceIndex = nil
        playbackPhase = .idle
        log("Playback stopped", level: .warning)
    }

    func clearText() {
        stopPlayback()
        sentences = []
        loadedText = ""
        log("Text cleared", level: .warning)
    }

    func importText(_ text: String) {
        stopPlayback()
        loadText(text)
        playbackPhase = .idle
        currentSentenceIndex = nil
        log("Imported text with \(sentences.count) sentences")
    }

    func skipNext() {
        guard !sentences.isEmpty else { return }

        let target = (currentSentenceIndex ?? -1) + 1
        guard target < sentences.count else {
            finishPlayback()
            return
        }

        startPlayback(at: target)
    }

    func skipPrevious() {
        guard !sentences.isEmpty else { return }

        let target = max((currentSentenceIndex ?? 0) - 1, 0)
        startPlayback(at: target)
    }

    func jumpToSentence(_ index: Int) {
        guard sentences.indices.contains(index) else { return }
        startPlayback(at: index)
    }

    func selectVoice(_ voiceID: String) {
        guard selectedVoiceID != voiceID else { return }
        selectedVoiceID = voiceID
        defaults.set(voiceID, forKey: selectedVoiceDefaultsKey)
        preparedNextBuffer = nil
        preparedNextIndex = nil
        preparedVoiceID = nil

        if case .playing = playbackPhase, let currentSentenceIndex {
            prefetchSentence(after: currentSentenceIndex)
        }

        log("Voice changed to \(displayName(for: voiceID))")
    }

    func updateSpeed(_ newSpeed: Double) {
        let clamped = min(2.0, max(0.5, newSpeed))
        speed = clamped
        timePitchNode.rate = Float(clamped * 100.0)
        defaults.set(clamped, forKey: speedDefaultsKey)
        log("Speed set to \(String(format: "%.1f", clamped))x")
    }

    func clearLogs() {
        debugLogs.removeAll()
    }

    private func configureVoices() {
        let options = voices.keys
            .compactMap { key -> VoiceOption? in
                let id = key.replacingOccurrences(of: ".npy", with: "")
                let parts = id.split(separator: "_")
                guard parts.count == 2 else { return nil }

                let prefix = String(parts[0])
                let rawName = String(parts[1])
                let locale: VoiceOption.LocaleGroup
                let descriptor: String

                switch prefix {
                case "af":
                    locale = .us
                    descriptor = "American English · Female"
                case "am":
                    locale = .us
                    descriptor = "American English · Male"
                case "bf":
                    locale = .gb
                    descriptor = "British English · Female"
                case "bm":
                    locale = .gb
                    descriptor = "British English · Male"
                default:
                    return nil
                }

                return VoiceOption(
                    id: id,
                    name: rawName.capitalized,
                    localeGroup: locale,
                    descriptor: descriptor
                )
            }
            .sorted { lhs, rhs in
                if lhs.localeGroup == rhs.localeGroup {
                    return lhs.name < rhs.name
                }
                return lhs.localeGroup == .us
            }

        voiceOptions = options
    }

    private func configureSettings() {
        let defaultVoice = usVoices.first?.id ?? voiceOptions.first?.id ?? ""
        let savedVoice = defaults.string(forKey: selectedVoiceDefaultsKey)
        selectedVoiceID = voiceOptions.contains(where: { $0.id == savedVoice }) ? (savedVoice ?? defaultVoice) : defaultVoice

        let savedSpeed = defaults.object(forKey: speedDefaultsKey) as? Double ?? 1.0
        speed = min(2.0, max(0.5, savedSpeed))
        timePitchNode.rate = Float(speed * 100.0)
    }

    private func configureAudioEngine() {
        audioEngine.attach(playerNode)
        audioEngine.attach(timePitchNode)
        audioEngine.connect(playerNode, to: timePitchNode, format: nil)
        audioEngine.connect(timePitchNode, to: audioEngine.mainMixerNode, format: nil)

        do {
            try audioEngine.start()
        } catch {
            log("Audio engine failed to start: \(error.localizedDescription)", level: .error)
        }
    }

    private func configureAudioSession() {
        #if os(iOS)
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
        } catch {
            log("Failed to configure AVAudioSession: \(error.localizedDescription)", level: .error)
        }
        #endif
    }

    private func configureInterruptions() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioInterruption(_:)),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
    }

    private func loadText(_ text: String) {
        loadedText = text
        sentences = tokenizeSentences(from: text)
        currentSentenceIndex = nil
        preparedNextBuffer = nil
        preparedNextIndex = nil
        preparedVoiceID = nil

        if sentences.isEmpty {
            playbackPhase = .idle
            log("No sentences found after tokenization", level: .warning)
        }
    }

    private func tokenizeSentences(from text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                result.append(sentence)
            }
            return true
        }
        return result
    }

    private func startPlayback(at index: Int) {
        guard sentences.indices.contains(index) else { return }

        invalidateSynthesisTokens()
        playerNode.stop()
        currentSentenceIndex = index

        if let preparedNextBuffer,
           preparedNextIndex == index,
           preparedVoiceID == selectedVoiceID {
            self.preparedNextBuffer = nil
            self.preparedNextIndex = nil
            self.preparedVoiceID = nil
            scheduleBufferAndPlay(preparedNextBuffer, at: index)
            return
        }

        playbackPhase = .generating
        let token = playbackToken
        let sentence = sentences[index]
        let voiceID = selectedVoiceID

        synthesisQueue.async { [weak self] in
            guard let self else { return }
            let result = self.generateBuffer(for: sentence, voiceID: voiceID)

            DispatchQueue.main.async {
                guard self.playbackToken == token else { return }

                switch result {
                case .success(let buffer):
                    self.scheduleBufferAndPlay(buffer, at: index)
                case .failure(let error):
                    self.playbackPhase = .idle
                    self.currentSentenceIndex = nil
                    self.log("Synthesis failed: \(error.localizedDescription)", level: .error)
                }
            }
        }
    }

    private func scheduleBufferAndPlay(_ buffer: AVAudioPCMBuffer, at index: Int) {
        do {
            if !audioEngine.isRunning {
                try audioEngine.start()
            }
        } catch {
            playbackPhase = .idle
            currentSentenceIndex = nil
            log("Audio engine restart failed: \(error.localizedDescription)", level: .error)
            return
        }

        currentSentenceIndex = index
        playbackPhase = .playing

        playerNode.stop()
        playerNode.scheduleBuffer(buffer, at: nil, options: []) { [weak self] in
            DispatchQueue.main.async {
                self?.handleSentenceCompletion(finishedIndex: index)
            }
        }
        playerNode.play()

        log("Playing sentence \(index + 1)/\(sentences.count)")
        prefetchSentence(after: index)
    }

    private func handleSentenceCompletion(finishedIndex: Int) {
        guard currentSentenceIndex == finishedIndex else { return }

        let nextIndex = finishedIndex + 1
        guard nextIndex < sentences.count else {
            finishPlayback()
            return
        }

        if let preparedNextBuffer,
           preparedNextIndex == nextIndex,
           preparedVoiceID == selectedVoiceID {
            self.preparedNextBuffer = nil
            self.preparedNextIndex = nil
            self.preparedVoiceID = nil
            scheduleBufferAndPlay(preparedNextBuffer, at: nextIndex)
            return
        }

        startPlayback(at: nextIndex)
    }

    private func prefetchSentence(after index: Int) {
        let nextIndex = index + 1
        guard sentences.indices.contains(nextIndex) else {
            preparedNextBuffer = nil
            preparedNextIndex = nil
            preparedVoiceID = nil
            return
        }

        prefetchToken = UUID()
        let token = prefetchToken
        let sentence = sentences[nextIndex]
        let voiceID = selectedVoiceID

        synthesisQueue.async { [weak self] in
            guard let self else { return }
            let result = self.generateBuffer(for: sentence, voiceID: voiceID)

            DispatchQueue.main.async {
                guard self.prefetchToken == token else { return }

                switch result {
                case .success(let buffer):
                    self.preparedNextBuffer = buffer
                    self.preparedNextIndex = nextIndex
                    self.preparedVoiceID = voiceID
                case .failure(let error):
                    self.preparedNextBuffer = nil
                    self.preparedNextIndex = nil
                    self.preparedVoiceID = nil
                    self.log("Prefetch failed: \(error.localizedDescription)", level: .warning)
                }
            }
        }
    }

    private func generateBuffer(for sentence: String, voiceID: String) -> Result<AVAudioPCMBuffer, Error> {
        guard let voice = voices[voiceID + ".npy"] else {
            return .failure(NSError(domain: "KokoroTestApp", code: 1, userInfo: [NSLocalizedDescriptionKey: "Voice data missing"]))
        }

        do {
            let language: KokoroTTS.Language = voiceID.hasPrefix("a") ? .enUS : .enGB
            let (audio, _) = try kokoroTTSEngine.generateAudio(voice: voice, language: language, text: sentence)

            let sampleRate = Double(KokoroTTS.Constants.samplingRate)
            let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!

            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(audio.count)
            ) else {
                return .failure(NSError(domain: "KokoroTestApp", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not create audio buffer"]))
            }

            buffer.frameLength = buffer.frameCapacity
            guard let channel = buffer.floatChannelData?[0] else {
                return .failure(NSError(domain: "KokoroTestApp", code: 3, userInfo: [NSLocalizedDescriptionKey: "Buffer channel unavailable"]))
            }

            audio.withUnsafeBufferPointer { pointer in
                guard let baseAddress = pointer.baseAddress else { return }
                let byteCount = pointer.count * MemoryLayout<Float>.stride
                UnsafeMutableRawPointer(channel).copyMemory(from: baseAddress, byteCount: byteCount)
            }

            return .success(buffer)
        } catch {
            return .failure(error)
        }
    }

    private func pausePlayback() {
        guard playbackPhase == .playing else { return }
        playerNode.pause()
        playbackPhase = .paused
        log("Playback paused")
    }

    private func resumePlayback() {
        guard playbackPhase == .paused else { return }
        playerNode.play()
        playbackPhase = .playing
        log("Playback resumed")
    }

    private func finishPlayback() {
        invalidateSynthesisTokens()
        playerNode.stop()
        currentSentenceIndex = nil
        playbackPhase = .done
        log("Playback completed")
    }

    private func invalidateSynthesisTokens() {
        playbackToken = UUID()
        prefetchToken = UUID()
        preparedNextBuffer = nil
        preparedNextIndex = nil
        preparedVoiceID = nil
    }

    private func displayName(for voiceID: String) -> String {
        voiceOptions.first(where: { $0.id == voiceID })?.name ?? voiceID
    }

    @objc
    private func handleAudioInterruption(_ notification: Notification) {
        guard
            let userInfo = notification.userInfo,
            let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
            let type = AVAudioSession.InterruptionType(rawValue: typeValue)
        else {
            return
        }

        if type == .began, playbackPhase == .playing {
            pausePlayback()
            log("Playback paused due to audio interruption", level: .warning)
        }
    }

    private func log(_ message: String, level: DebugLogEntry.Level = .info) {
        let entry = DebugLogEntry(timestamp: Date(), level: level, message: message)

        if Thread.isMainThread {
            debugLogs.append(entry)
        } else {
            DispatchQueue.main.async {
                self.debugLogs.append(entry)
            }
        }
    }
}
