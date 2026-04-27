import AVFoundation
import Combine
import Foundation
import KokoroSwift
import MediaPlayer
import MLX
import MLXUtilsLibrary
import NaturalLanguage
import SwiftUI
#if os(iOS)
import UIKit
#endif

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
    @Published var generationProgress: Double = 0

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
    private let playbackSampleRate = Double(KokoroTTS.Constants.samplingRate)
    private lazy var playbackFormat = AVAudioFormat(
        standardFormatWithSampleRate: playbackSampleRate,
        channels: 1
    )!

    private let synthesisQueue = DispatchQueue(label: "kokoro.tts.synthesis")
    private var playbackToken = UUID()
    private var isAppActive = true
    private var pendingResumeIndex: Int?
    private var bufferedSentenceAudio: [Int: AVAudioPCMBuffer] = [:]
    private var bufferedVoiceID: String?

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
        configureRemoteCommands()

        log("Model initialized with \(voiceOptions.count) voices")
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    var hasLoadedText: Bool {
        !sentences.isEmpty
    }

    var isPlaybackPrepared: Bool {
        guard !sentences.isEmpty else { return false }
        guard bufferedVoiceID == selectedVoiceID else { return false }
        return bufferedSentenceAudio.count == sentences.count
    }

    func togglePlay(using text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if loadedText != text || sentences.isEmpty {
            loadText(text)
        }

        guard !sentences.isEmpty else { return }
        guard isPlaybackPrepared else {
            preparePlayback(using: text)
            return
        }

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

    func preparePlayback(using text: String, autoPlay: Bool = false) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if loadedText != text || sentences.isEmpty {
            loadText(text)
        }

        guard !sentences.isEmpty else { return }
        if isPlaybackPrepared {
            if autoPlay {
                startPlayback(at: currentSentenceIndex ?? 0)
            }
            return
        }

        prepareAllSentences(autoPlay: autoPlay)
    }

    func stopPlayback() {
        invalidateSynthesisTokens()
        playerNode.stop()
        currentSentenceIndex = nil
        playbackPhase = .idle
        generationProgress = 0
        clearNowPlayingInfo()
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
        bufferedSentenceAudio = [:]
        bufferedVoiceID = nil
        generationProgress = 0

        if case .playing = playbackPhase, let currentSentenceIndex {
            startPlayback(at: currentSentenceIndex)
        }

        log("Voice changed to \(displayName(for: voiceID))")
    }

    func updateSpeed(_ newSpeed: Double) {
        let clamped = min(2.0, max(0.5, newSpeed))
        speed = clamped
        timePitchNode.rate = Float(clamped)
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
        timePitchNode.rate = Float(speed)
    }

    private func configureAudioEngine() {
        audioEngine.attach(playerNode)
        audioEngine.attach(timePitchNode)
        audioEngine.connect(playerNode, to: timePitchNode, format: playbackFormat)
        audioEngine.connect(timePitchNode, to: audioEngine.mainMixerNode, format: playbackFormat)

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
        #if os(iOS)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioInterruption(_:)),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleWillResignActive),
            name: UIApplication.willResignActiveNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
        #endif
    }

    private func configureRemoteCommands() {
        #if os(iOS)
        let commandCenter = MPRemoteCommandCenter.shared()

        commandCenter.playCommand.isEnabled = true
        commandCenter.pauseCommand.isEnabled = true
        commandCenter.nextTrackCommand.isEnabled = true
        commandCenter.previousTrackCommand.isEnabled = true

        commandCenter.playCommand.addTarget { [weak self] _ in
            guard let self else { return .commandFailed }
            if self.playbackPhase == .paused {
                self.resumePlayback()
                return .success
            }
            if self.playbackPhase == .idle || self.playbackPhase == .done {
                if self.isPlaybackPrepared {
                    self.startPlayback(at: self.currentSentenceIndex ?? 0)
                } else if self.isAppActive {
                    self.prepareAllSentences(autoPlay: true)
                } else {
                    return .commandFailed
                }
                return .success
            }
            return .commandFailed
        }

        commandCenter.pauseCommand.addTarget { [weak self] _ in
            guard let self else { return .commandFailed }
            if self.playbackPhase == .playing {
                self.pausePlayback()
                return .success
            }
            return .commandFailed
        }

        commandCenter.nextTrackCommand.addTarget { [weak self] _ in
            self?.skipNext()
            return .success
        }

        commandCenter.previousTrackCommand.addTarget { [weak self] _ in
            self?.skipPrevious()
            return .success
        }
        #endif
    }

    private func loadText(_ text: String) {
        loadedText = text
        sentences = tokenizeSentences(from: text)
        currentSentenceIndex = nil
        bufferedSentenceAudio = [:]
        bufferedVoiceID = nil
        generationProgress = 0

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
        guard isPlaybackPrepared else {
            log("Playback requested before generation completed", level: .warning)
            return
        }
        pendingResumeIndex = nil

        guard let cachedBuffer = bufferedSentenceAudio[index] else {
            log("Missing generated audio for sentence \(index + 1)", level: .error)
            playbackPhase = .idle
            return
        }

        scheduleBufferAndPlay(cachedBuffer, at: index)
    }

    private func prepareAllSentences(autoPlay: Bool) {
        guard !sentences.isEmpty else { return }
        guard isAppActive else {
            log("Cannot generate while app is inactive", level: .warning)
            return
        }

        playbackToken = UUID()
        let token = playbackToken
        let voiceID = selectedVoiceID
        let sourceSentences = sentences
        let startIndex = currentSentenceIndex ?? 0

        playbackPhase = .generating
        generationProgress = 0
        bufferedSentenceAudio = [:]
        bufferedVoiceID = voiceID
        pendingResumeIndex = nil

        synthesisQueue.async { [weak self] in
            guard let self else { return }
            var generated: [Int: AVAudioPCMBuffer] = [:]
            let totalCount = sourceSentences.count

            for sentenceIndex in sourceSentences.indices {
                guard self.playbackToken == token else { return }
                guard self.isAppActive else { return }
                guard self.selectedVoiceID == voiceID else { return }

                let result = self.generateBuffer(for: sourceSentences[sentenceIndex], voiceID: voiceID)
                switch result {
                case .success(let buffer):
                    generated[sentenceIndex] = buffer
                    let progress = Double(sentenceIndex + 1) / Double(totalCount)
                    DispatchQueue.main.async {
                        guard self.playbackToken == token else { return }
                        self.generationProgress = progress
                    }
                case .failure(let error):
                    DispatchQueue.main.async {
                        guard self.playbackToken == token else { return }
                        self.playbackPhase = .idle
                        self.currentSentenceIndex = nil
                        self.generationProgress = 0
                        self.log("Synthesis failed: \(error.localizedDescription)", level: .error)
                    }
                    return
                }
            }

            DispatchQueue.main.async {
                guard self.playbackToken == token else { return }
                guard self.selectedVoiceID == voiceID else { return }

                self.bufferedSentenceAudio = generated
                self.bufferedVoiceID = voiceID
                self.generationProgress = 1
                self.playbackPhase = .idle
                self.currentSentenceIndex = startIndex
                self.log("Audio generation complete for \(generated.count) sentences")

                if autoPlay {
                    self.startPlayback(at: startIndex)
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

        guard let playableBuffer = makePlayableBuffer(from: buffer) else {
            playbackPhase = .idle
            currentSentenceIndex = nil
            log("Audio buffer conversion failed", level: .error)
            return
        }

        playerNode.stop()
        playerNode.scheduleBuffer(
            playableBuffer,
            at: nil,
            options: [],
            completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            DispatchQueue.main.async {
                self?.handleSentenceCompletion(finishedIndex: index)
            }
        }
        playerNode.play()

        updateNowPlayingInfo(sentenceIndex: index, sentenceText: sentences[index], rate: speed)
        log("Playing sentence \(index + 1)/\(sentences.count)")
    }

    private func handleSentenceCompletion(finishedIndex: Int) {
        guard currentSentenceIndex == finishedIndex else { return }

        let nextIndex = finishedIndex + 1
        guard nextIndex < sentences.count else {
            finishPlayback()
            return
        }

        guard let cachedBuffer = bufferedSentenceAudio[nextIndex] else {
            playbackPhase = .idle
            currentSentenceIndex = nil
            log("Missing generated audio for sentence \(nextIndex + 1)", level: .error)
            return
        }

        scheduleBufferAndPlay(cachedBuffer, at: nextIndex)
    }

    private func generateBuffer(for sentence: String, voiceID: String) -> Result<AVAudioPCMBuffer, Error> {
        guard let voice = voices[voiceID + ".npy"] else {
            return .failure(NSError(domain: "KokoroTestApp", code: 1, userInfo: [NSLocalizedDescriptionKey: "Voice data missing"]))
        }

        do {
            let device: Device = .gpu
            let (audio, _) = try Device.withDefaultDevice(device) {
                try kokoroTTSEngine.generateAudio(
                    voice: voice,
                    language: voiceID.hasPrefix("a") ? .enUS : .enGB,
                    text: sentence
                )
            }

            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: playbackFormat,
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

    private func makePlayableBuffer(from buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let targetFormat = playerNode.outputFormat(forBus: 0)
        let sourceFormat = buffer.format

        let formatsAlreadyMatch =
            sourceFormat.sampleRate == targetFormat.sampleRate &&
            sourceFormat.channelCount == targetFormat.channelCount &&
            sourceFormat.commonFormat == targetFormat.commonFormat &&
            sourceFormat.isInterleaved == targetFormat.isInterleaved

        if formatsAlreadyMatch {
            return buffer
        }

        guard
            let converter = AVAudioConverter(from: sourceFormat, to: targetFormat),
            let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: AVAudioFrameCount(
                    Double(buffer.frameLength) * targetFormat.sampleRate / sourceFormat.sampleRate
                ) + 1
            )
        else {
            return nil
        }

        var didProvideSourceBuffer = false
        var conversionError: NSError?
        let status = converter.convert(to: convertedBuffer, error: &conversionError) { _, outStatus in
            if didProvideSourceBuffer {
                outStatus.pointee = .endOfStream
                return nil
            } else {
                didProvideSourceBuffer = true
                outStatus.pointee = .haveData
                return buffer
            }
        }

        guard conversionError == nil else {
            return nil
        }

        switch status {
        case .haveData, .inputRanDry, .endOfStream:
            return convertedBuffer
        default:
            return nil
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
        generationProgress = 0
        clearNowPlayingInfo()
        log("Playback completed")
    }

    private func invalidateSynthesisTokens() {
        playbackToken = UUID()
        bufferedSentenceAudio = [:]
        bufferedVoiceID = nil
        pendingResumeIndex = nil
        generationProgress = 0
    }

    private func displayName(for voiceID: String) -> String {
        voiceOptions.first(where: { $0.id == voiceID })?.name ?? voiceID
    }

    private func updateNowPlayingInfo(sentenceIndex: Int, sentenceText: String, rate: Double) {
        #if os(iOS)
        var info = MPNowPlayingInfoCenter.default().nowPlayingInfo ?? [:]
        info[MPMediaItemPropertyTitle] = "Kokoro Reader"
        info[MPMediaItemPropertyArtist] = displayName(for: selectedVoiceID)
        info[MPMediaItemPropertyAlbumTitle] = "Sentence \(sentenceIndex + 1) of \(sentences.count)"
        info[MPNowPlayingInfoPropertyPlaybackRate] = rate
        info[MPNowPlayingInfoPropertyElapsedPlaybackTime] = 0
        info[MPMediaItemPropertyPlaybackDuration] = 1
        info[MPNowPlayingInfoPropertyMediaType] = MPNowPlayingInfoMediaType.audio.rawValue
        info[MPMediaItemPropertyComposer] = sentenceText
        MPNowPlayingInfoCenter.default().nowPlayingInfo = info
        #endif
    }

    private func clearNowPlayingInfo() {
        #if os(iOS)
        MPNowPlayingInfoCenter.default().nowPlayingInfo = nil
        #endif
    }

    #if os(iOS)
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
            if let reasonValue = userInfo[AVAudioSessionInterruptionReasonKey] as? UInt,
               reasonValue == 1
            {
                log("Ignoring suspension interruption while background audio is active")
                return
            }
            pausePlayback()
            log("Playback paused due to audio interruption", level: .warning)
            return
        }

        if type == .ended,
           let optionValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt
        {
            let options = AVAudioSession.InterruptionOptions(rawValue: optionValue)
            if options.contains(.shouldResume), playbackPhase == .paused {
                resumePlayback()
                log("Playback resumed after audio interruption")
            }
        }
    }

    @objc
    private func handleWillResignActive() {
        isAppActive = false
        if playbackPhase == .generating {
            pendingResumeIndex = currentSentenceIndex ?? 0
            playbackToken = UUID()
            playbackPhase = .paused
        }
        log("App moved to inactive/background state; synthesis paused")
    }

    @objc
    private func handleDidBecomeActive() {
        isAppActive = true
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setActive(true)
            if !audioEngine.isRunning {
                try audioEngine.start()
            }
        } catch {
            log("Failed to reactivate audio on foreground: \(error.localizedDescription)", level: .warning)
        }
        if let pendingResumeIndex, playbackPhase == .paused {
            self.pendingResumeIndex = nil
            if isPlaybackPrepared {
                startPlayback(at: pendingResumeIndex)
            } else {
                currentSentenceIndex = pendingResumeIndex
                prepareAllSentences(autoPlay: true)
            }
        }
        log("App is active; synthesis resumed")
    }
    #endif

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
