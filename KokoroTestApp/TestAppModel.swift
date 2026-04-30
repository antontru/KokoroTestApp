import AVFoundation
import Combine
import Foundation
import MediaPlayer
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
    @Published var modelVariant: KokoroONNXSynthesiser.ModelVariant = .quantized
    @Published var lowMemoryModeEnabled: Bool = false
    @Published var coreMLWarningMessage: String?

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

    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let synthesisQueue = DispatchQueue(label: "kokoro.onnx.synthesis")
    private let synthesiserSetupQueue = DispatchQueue(label: "kokoro.onnx.setup", qos: .userInitiated)
    private let synthesisAudioFormat = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!

    private var synthesiser: KokoroONNXSynthesiser?

    private var playbackToken = UUID()
    private var isAppActive = true
    private var shouldConserveMemory = false
    private var pendingResumeIndex: Int?

    private var bufferedSentenceAudio: [Int: AVAudioPCMBuffer] = [:]
    private var bufferedVoiceID: String?
    private var generatingIndices: Set<Int> = []
    private var autoplayPendingIndices: Set<Int> = []
    private let defaultLookAheadSentenceCount = 3
    private let lowMemoryLookAheadSentenceCount = 1
    private let backgroundLookAheadSentenceCount = 0
    private let defaultCacheSentenceLimit = 6
    private let lowMemoryCacheSentenceLimit = 2
    private let lookAheadSentenceCount = 3
    private let retainedSentenceWindow = 1
    private let maxSentenceCharacterCount = 180
    private let maxSentenceWordCount = 36

    private var loadedText: String = ""
    private let defaults = UserDefaults.standard

    private let selectedVoiceDefaultsKey = "kokoro.selectedVoiceID"
    private let speedDefaultsKey = "kokoro.speed"
    private let modelVariantDefaultsKey = "kokoro_model_variant"
    private let lowMemoryModeDefaultsKey = "kokoro_low_memory_mode"

    init() {
        configureVoices()
        configureAudioEngine()
        configureAudioSession()
        configureSettings()
        createSynthesiser()
        configureInterruptions()
        configureRemoteCommands()
        logStartupConfiguration()
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
        return !bufferedSentenceAudio.isEmpty
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
            startStreamingPlayback(at: currentSentenceIndex ?? 0)
        case .done:
            startStreamingPlayback(at: 0)
        case .generating:
            if let currentSentenceIndex {
                startStreamingPlayback(at: currentSentenceIndex)
            } else {
                startStreamingPlayback(at: 0)
            }
        }
    }

    func preparePlayback(using text: String, autoPlay: Bool = false) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if loadedText != text || sentences.isEmpty {
            loadText(text)
        }

        guard !sentences.isEmpty else { return }
        let index = currentSentenceIndex ?? 0
        requestBufferIfNeeded(at: index, token: playbackToken, autoplayWhenReady: autoPlay)
        prefetchLookAhead(from: index, token: playbackToken)
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
        invalidateSynthesisTokens()
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
        defaults.set(clamped, forKey: speedDefaultsKey)
        log("Generation speed set to \(String(format: "%.1f", clamped))x")
    }

    func updateModelVariant(_ variant: KokoroONNXSynthesiser.ModelVariant) {
        guard modelVariant != variant else { return }
        modelVariant = variant
        defaults.set(variant.rawValue, forKey: modelVariantDefaultsKey)
        log("Switching model variant to \(variant.displayName)")
        invalidateSynthesisTokens()

        synthesiserSetupQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.synthesiser?.reload(variant: variant)
                DispatchQueue.main.async {
                    self.updateCoreMLWarningFromSynthesiser()
                    self.invalidateSynthesisTokens()
                    self.playbackPhase = .idle
                    self.currentSentenceIndex = nil
                    self.log("Model variant switched to \(variant.displayName)")
                }
            } catch {
                DispatchQueue.main.async {
                    self.log("Could not switch model variant: \(error.localizedDescription)", level: .error)
                }
            }
        }
    }

    func clearLogs() {
        debugLogs.removeAll()
    }

    private func configureVoices() {
        let entries: [(String, String)] = [
            ("af_heart", "American English · Female"),
            ("af_alloy", "American English · Female"),
            ("af_aoede", "American English · Female"),
            ("af_bella", "American English · Female"),
            ("af_jessica", "American English · Female"),
            ("af_kore", "American English · Female"),
            ("af_nicole", "American English · Female"),
            ("af_nova", "American English · Female"),
            ("af_river", "American English · Female"),
            ("af_sarah", "American English · Female"),
            ("af_sky", "American English · Female"),
            ("am_adam", "American English · Male"),
            ("am_echo", "American English · Male"),
            ("am_eric", "American English · Male"),
            ("am_fenrir", "American English · Male"),
            ("am_liam", "American English · Male"),
            ("am_michael", "American English · Male"),
            ("am_onyx", "American English · Male"),
            ("am_puck", "American English · Male"),
            ("am_santa", "American English · Male"),
            ("bf_alice", "British English · Female"),
            ("bf_emma", "British English · Female"),
            ("bf_isabella", "British English · Female"),
            ("bf_lily", "British English · Female"),
            ("bm_daniel", "British English · Male"),
            ("bm_fable", "British English · Male"),
            ("bm_george", "British English · Male"),
            ("bm_lewis", "British English · Male"),
        ]

        voiceOptions = entries.map { id, descriptor in
            let parts = id.split(separator: "_")
            let prefix = String(parts.first ?? "")
            let locale: VoiceOption.LocaleGroup = prefix.hasPrefix("b") ? .gb : .us
            return VoiceOption(
                id: id,
                name: String(parts.dropFirst().joined(separator: " ")).capitalized,
                localeGroup: locale,
                descriptor: descriptor
            )
        }
    }

    private func configureSettings() {
        let defaultVoice = usVoices.first?.id ?? voiceOptions.first?.id ?? ""
        let savedVoice = defaults.string(forKey: selectedVoiceDefaultsKey)
        selectedVoiceID = voiceOptions.contains(where: { $0.id == savedVoice }) ? (savedVoice ?? defaultVoice) : defaultVoice

        let savedSpeed = defaults.object(forKey: speedDefaultsKey) as? Double ?? 1.0
        speed = min(2.0, max(0.5, savedSpeed))

        let savedVariant = defaults.string(forKey: modelVariantDefaultsKey)

        lowMemoryModeEnabled = defaults.bool(forKey: lowMemoryModeDefaultsKey)
        if let savedVariant, let parsed = KokoroONNXSynthesiser.ModelVariant(rawValue: savedVariant) {
            modelVariant = parsed
        } else {
            if let savedVariant {
                log("Unknown persisted model variant '\(savedVariant)'; falling back to Quantized (8-bit)", level: .warning)
            }
            modelVariant = .quantized
            defaults.set(modelVariant.rawValue, forKey: modelVariantDefaultsKey)
        }
    }

    private func createSynthesiser() {
        logBundledResourceStatus()
        log("Initializing ONNX synthesiser…")
        let voiceIDs = voiceOptions.map(\.id)
        let variant = modelVariant

        synthesiserSetupQueue.async { [weak self] in
            guard let self else { return }
            do {
                let built = try KokoroONNXSynthesiser(voiceIDs: voiceIDs, variant: variant)
                DispatchQueue.main.async {
                    self.synthesiser = built
                    self.updateCoreMLWarningFromSynthesiser()
                    self.log("ONNX synthesiser initialized successfully")
                }
            } catch {
                DispatchQueue.main.async {
                    self.synthesiser = nil
                    self.coreMLWarningMessage = "CoreML execution provider unavailable. Running with CPU fallback where possible."
                    self.log("Failed to initialize ONNX synthesiser: \(error.localizedDescription)", level: .error)
                }
            }
        }
    }

    private func logBundledResourceStatus() {
        let fileNames = [
            "onnx/model.onnx",
            "onnx/model_quantized.onnx",
            "onnx/model_q4.onnx",
            "voices/af_heart.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ]
        for fileName in fileNames {
            let nsPath = fileName as NSString
            let resource = nsPath.deletingPathExtension
            let ext = nsPath.pathExtension.isEmpty ? nil : nsPath.pathExtension
            let subdir = (resource as NSString).deletingLastPathComponent
            let name = (resource as NSString).lastPathComponent
            let url: URL?
            if subdir.isEmpty || subdir == "." {
                url = Bundle.main.url(forResource: name, withExtension: ext)
            } else {
                url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdir)
                    ?? Bundle.main.url(forResource: name, withExtension: ext)
            }
            guard let url else {
                log("Resource missing from bundle: \(fileName)", level: .error)
                continue
            }
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? -1
            log("Resource \(fileName) size=\(size) bytes")
        }
    }

    private func updateCoreMLWarningFromSynthesiser() {
        guard let diagnostics = synthesiser?.diagnosticsInfo() else {
            coreMLWarningMessage = "ONNX synthesiser not available"
            return
        }

        if diagnostics.isCoreMLAvailable && diagnostics.isUsingCoreML {
            coreMLWarningMessage = nil
        } else {
            coreMLWarningMessage = "CoreML EP unavailable; inference may use CPU fallback."
        }
    }

    private func configureAudioEngine() {
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: synthesisAudioFormat)

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
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDidReceiveMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
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
                self.startStreamingPlayback(at: self.currentSentenceIndex ?? 0)
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
        invalidateSynthesisTokens()
        synthesiser?.clearCache()
        loadedText = text
        sentences = tokenizeSentences(from: text)
        currentSentenceIndex = nil
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
                let split = splitOversizedSentence(sentence)
                result.append(contentsOf: split)
            }
            return true
        }
        return result
    }

    private func splitOversizedSentence(_ sentence: String) -> [String] {
        guard needsSentenceSplit(sentence) else { return [sentence] }

        let separators = [", ", "; ", ": ", " — ", " - "]
        var pieces = [sentence]

        for separator in separators {
            if pieces.allSatisfy({ !needsSentenceSplit($0) }) {
                break
            }

            var refined: [String] = []
            for piece in pieces {
                guard needsSentenceSplit(piece) else {
                    refined.append(piece)
                    continue
                }
                let splitParts = piece.split(separator: Character(separator.trimmingCharacters(in: .whitespaces))).map {
                    $0.trimmingCharacters(in: .whitespacesAndNewlines)
                }
                if splitParts.count <= 1 {
                    refined.append(piece)
                    continue
                }
                refined.append(contentsOf: splitParts.filter { !$0.isEmpty })
            }
            pieces = refined
        }

        if pieces.allSatisfy({ !needsSentenceSplit($0) }) {
            return pieces
        }

        var chunked: [String] = []
        for piece in pieces {
            if !needsSentenceSplit(piece) {
                chunked.append(piece)
                continue
            }

            var currentWords: [Substring] = []
            var currentLength = 0
            for word in piece.split(whereSeparator: { $0.isWhitespace || $0.isNewline }) {
                let proposedLength = currentLength + (currentWords.isEmpty ? 0 : 1) + word.count
                if proposedLength > maxSentenceCharacterCount || currentWords.count >= maxSentenceWordCount {
                    if !currentWords.isEmpty {
                        chunked.append(currentWords.joined(separator: " "))
                        currentWords.removeAll(keepingCapacity: true)
                        currentLength = 0
                    }
                }
                currentWords.append(word)
                currentLength += (currentLength == 0 ? 0 : 1) + word.count
            }
            if !currentWords.isEmpty {
                chunked.append(currentWords.joined(separator: " "))
            }
        }

        return chunked.filter { !$0.isEmpty }
    }

    private func needsSentenceSplit(_ sentence: String) -> Bool {
        let wordCount = sentence.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        return sentence.count > maxSentenceCharacterCount || wordCount > maxSentenceWordCount
    }

    private func startPlayback(at index: Int) {
        startStreamingPlayback(at: index)
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

        guard let playableBuffer = ensurePlayerCompatibleBuffer(buffer) else {
            playbackPhase = .idle
            currentSentenceIndex = nil
            log("Buffer format conversion failed: input channels=\(buffer.format.channelCount), output channels=\(playerNode.outputFormat(forBus: 0).channelCount)", level: .error)
            return
        }

        playerNode.stop()
        playerNode.scheduleBuffer(playableBuffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            DispatchQueue.main.async {
                self?.handleSentenceCompletion(finishedIndex: index)
            }
        }
        playerNode.play()

        updateNowPlayingInfo(sentenceIndex: index, sentenceText: sentences[index], rate: speed)
        log("Playing sentence \(index + 1)/\(sentences.count)")
    }

    private func ensurePlayerCompatibleBuffer(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let outputFormat = playerNode.outputFormat(forBus: 0)
        if buffer.format.channelCount == outputFormat.channelCount,
           abs(buffer.format.sampleRate - outputFormat.sampleRate) < 0.5
        {
            return buffer
        }

        guard let converted = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: buffer.frameCapacity) else {
            return nil
        }
        converted.frameLength = buffer.frameLength

        guard let srcChannels = buffer.floatChannelData,
              let dstChannels = converted.floatChannelData
        else {
            return nil
        }

        let srcChannelCount = Int(buffer.format.channelCount)
        let dstChannelCount = Int(outputFormat.channelCount)
        let frames = Int(buffer.frameLength)

        for dst in 0..<dstChannelCount {
            let src = min(dst, srcChannelCount - 1)
            dstChannels[dst].assign(from: srcChannels[src], count: frames)
        }

        return converted
    }

    private func handleSentenceCompletion(finishedIndex: Int) {
        guard currentSentenceIndex == finishedIndex else {
            log("Ignoring completion for sentence \(finishedIndex + 1): current index changed to \(String(describing: currentSentenceIndex))", level: .warning)
            return
        }

        guard playbackPhase == .playing else {
            log("Ignoring completion for sentence \(finishedIndex + 1): playback phase is \(playbackPhase)", level: .warning)
            return
        }

        let nextIndex = finishedIndex + 1
        guard nextIndex < sentences.count else {
            finishPlayback()
            return
        }

        currentSentenceIndex = nextIndex
        pruneBufferedAudio(around: nextIndex)
        if let cachedBuffer = bufferedSentenceAudio[nextIndex] {
            scheduleBufferAndPlay(cachedBuffer, at: nextIndex)
        } else {
            playbackPhase = .generating
            requestBufferIfNeeded(at: nextIndex, token: playbackToken, autoplayWhenReady: true)
        }
        prefetchLookAhead(from: nextIndex, token: playbackToken)
    }

    private func generateBuffer(for sentence: String, voiceID: String) -> Result<AVAudioPCMBuffer, Error> {
        guard let synthesiser else {
            return .failure(NSError(domain: "KokoroONNX", code: 101, userInfo: [NSLocalizedDescriptionKey: "Synthesiser unavailable"]))
        }

        do {
            let buffer = try synthesiser.synthesiseBlocking(text: sentence, voiceID: voiceID, speed: Float(speed))
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
        playerNode.stop()
        currentSentenceIndex = nil
        playbackPhase = .done
        clearNowPlayingInfo()
        log("Playback completed")
    }

    private func invalidateSynthesisTokens() {
        playbackToken = UUID()
        bufferedSentenceAudio = [:]
        bufferedVoiceID = nil
        generatingIndices.removeAll()
        autoplayPendingIndices.removeAll()
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
        shouldConserveMemory = true
        trimBuffersToCurrentWindow(reason: "app entered background", forceAggressive: true)
        invalidateSynthesisTokens()
        synthesiser?.clearCache()
        log("App moved to inactive/background state")
    }

    @objc
    private func handleDidBecomeActive() {
        isAppActive = true
        shouldConserveMemory = false
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
            startStreamingPlayback(at: pendingResumeIndex)
        }
        log("App is active")
    }

    @objc
    private func handleDidReceiveMemoryWarning() {
        shouldConserveMemory = true
        synthesiser?.clearCache()
        trimBuffersToCurrentWindow(reason: "iOS memory warning", forceAggressive: true)
    }
    #endif

    private func startStreamingPlayback(at index: Int) {
        guard sentences.indices.contains(index) else { return }
        guard synthesiser != nil else {
            log("Synthesiser unavailable", level: .error)
            return
        }

        playbackToken = UUID()
        let token = playbackToken
        pendingResumeIndex = nil
        currentSentenceIndex = index
        bufferedVoiceID = selectedVoiceID
        pruneBufferedAudio(around: index)
        playerNode.stop()

        if let cachedBuffer = bufferedSentenceAudio[index] {
            scheduleBufferAndPlay(cachedBuffer, at: index)
        } else {
            playbackPhase = .generating
            requestBufferIfNeeded(at: index, token: token, autoplayWhenReady: true)
        }

        prefetchLookAhead(from: index, token: token)
    }

    private func prefetchLookAhead(from startIndex: Int, token: UUID) {
        guard startIndex < sentences.count else { return }
        let lookAheadCount = currentLookAheadSentenceCount
        guard lookAheadCount > 0 else { return }
        let endIndex = min(sentences.count, startIndex + lookAheadCount)
        for index in startIndex..<endIndex {
            requestBufferIfNeeded(at: index, token: token, autoplayWhenReady: false)
        }
    }

    private func requestBufferIfNeeded(at index: Int, token: UUID, autoplayWhenReady: Bool) {
        guard sentences.indices.contains(index) else { return }
        guard bufferedSentenceAudio[index] == nil else { return }
        if generatingIndices.contains(index) {
            if autoplayWhenReady {
                autoplayPendingIndices.insert(index)
                log("Marked sentence \(index + 1) for autoplay when synthesis completes")
            }
            return
        }
        let sentence = sentences[index]
        let voiceID = selectedVoiceID
        generatingIndices.insert(index)
        log("Synthesis start sentence \(index + 1): \(sentence.prefix(48))...")

        synthesisQueue.async { [weak self] in
            guard let self else { return }
            let result = self.generateBuffer(for: sentence, voiceID: voiceID)
            DispatchQueue.main.async {
                self.generatingIndices.remove(index)
                let shouldAutoplayAfterSynthesis = autoplayWhenReady || self.autoplayPendingIndices.remove(index) != nil

                guard self.playbackToken == token else {
                    self.log("Dropping synthesis completion for sentence \(index + 1): token mismatch", level: .warning)
                    return
                }

                guard self.sentences.indices.contains(index) else {
                    self.log("Dropping synthesis completion for sentence \(index + 1): sentence index no longer valid", level: .warning)
                    return
                }

                guard self.selectedVoiceID == voiceID else {
                    self.log("Dropping synthesis completion for sentence \(index + 1): voice changed", level: .warning)
                    return
                }

                switch result {
                case .success(let buffer):
                    self.bufferedSentenceAudio[index] = buffer
                    self.enforceCachePolicy()
                    self.bufferedVoiceID = voiceID
                    self.pruneBufferedAudio(around: self.currentSentenceIndex ?? index)
                    self.log("Synthesis done sentence \(index + 1): frames=\(buffer.frameLength)")
                    let progress = Double(self.bufferedSentenceAudio.count) / Double(max(self.sentences.count, 1))
                    self.generationProgress = min(1, max(0, progress.isFinite ? progress : 0))

                    guard shouldAutoplayAfterSynthesis else { return }
                    guard self.playbackToken == token else { return }
                    guard self.sentences.indices.contains(index) else { return }
                    guard self.selectedVoiceID == voiceID else { return }
                    guard self.currentSentenceIndex == index else { return }
                    guard self.playbackPhase == .generating || self.playbackPhase == .playing else {
                        self.log("Skipping autoplay for sentence \(index + 1): playback phase is \(self.playbackPhase)", level: .warning)
                        return
                    }

                    self.scheduleBufferAndPlay(buffer, at: index)
                case .failure(let error):
                    guard self.playbackToken == token else { return }
                    guard self.selectedVoiceID == voiceID else { return }
                    guard self.sentences.indices.contains(index) else { return }
                    guard self.currentSentenceIndex == index || self.playbackPhase == .generating else {
                        self.log("Ignoring synthesis failure for sentence \(index + 1): playback moved on", level: .warning)
                        return
                    }
                    self.playbackPhase = .idle
                    self.log("Synthesis failed sentence \(index + 1): \(error.localizedDescription)", level: .error)
                }
            }
        }
    }

    private func pruneBufferedAudio(around centerIndex: Int) {
        let lowerBound = max(0, centerIndex - retainedSentenceWindow)
        let upperBound = min(sentences.count - 1, centerIndex + currentLookAheadSentenceCount)
        bufferedSentenceAudio = bufferedSentenceAudio.filter { index, _ in
            (lowerBound...upperBound).contains(index)
        }
        generatingIndices = generatingIndices.filter { (lowerBound...upperBound).contains($0) }
        autoplayPendingIndices = autoplayPendingIndices.filter { (lowerBound...upperBound).contains($0) }
    }

    private func log(_ message: String, level: DebugLogEntry.Level = .info) {
        let entry = DebugLogEntry(timestamp: Date(), level: level, message: message)
        print("[KokoroTestApp] \(message)")

        if Thread.isMainThread {
            debugLogs.append(entry)
        } else {
            DispatchQueue.main.async {
                self.debugLogs.append(entry)
            }
        }
    }

    var memoryPolicyDescription: String {
        "lookAhead=\(currentLookAheadSentenceCount), cacheLimit=\(currentCacheSentenceLimit), lowMemoryMode=\(lowMemoryModeEnabled ? "on" : "off"), conserveMemory=\(shouldConserveMemory ? "yes" : "no"), appActive=\(isAppActive ? "yes" : "no")"
    }

    func setLowMemoryModeEnabled(_ enabled: Bool) {
        guard lowMemoryModeEnabled != enabled else { return }
        lowMemoryModeEnabled = enabled
        defaults.set(enabled, forKey: lowMemoryModeDefaultsKey)
        trimBuffersToCurrentWindow(reason: "low memory mode \(enabled ? "enabled" : "disabled")", forceAggressive: enabled)
    }

    private var currentLookAheadSentenceCount: Int {
        if !isAppActive {
            return backgroundLookAheadSentenceCount
        }
        if shouldConserveMemory {
            return 1
        }
        return lowMemoryModeEnabled ? lowMemoryLookAheadSentenceCount : defaultLookAheadSentenceCount
    }

    private var currentCacheSentenceLimit: Int {
        lowMemoryModeEnabled ? lowMemoryCacheSentenceLimit : defaultCacheSentenceLimit
    }

    private func enforceCachePolicy() {
        guard bufferedSentenceAudio.count > currentCacheSentenceLimit else { return }
        let sortedIndices = bufferedSentenceAudio.keys.sorted()
        let keepIndices = Set(idealCacheIndices())
        for idx in sortedIndices where !keepIndices.contains(idx) {
            bufferedSentenceAudio.removeValue(forKey: idx)
            if bufferedSentenceAudio.count <= currentCacheSentenceLimit {
                break
            }
        }
    }

    private func idealCacheIndices() -> [Int] {
        guard let currentSentenceIndex else {
            return Array(bufferedSentenceAudio.keys.sorted().prefix(currentCacheSentenceLimit))
        }

        let upperBound = min(sentences.count - 1, currentSentenceIndex + currentLookAheadSentenceCount)
        if upperBound < currentSentenceIndex { return [currentSentenceIndex] }
        return Array(currentSentenceIndex...upperBound)
    }

    private func trimBuffersToCurrentWindow(reason: String, forceAggressive: Bool) {
        guard !bufferedSentenceAudio.isEmpty else { return }

        if forceAggressive {
            bufferedSentenceAudio.removeAll()
            generatingIndices.removeAll()
        autoplayPendingIndices.removeAll()
            generationProgress = 0
            log("Aggressively cleared synthesis + sentence buffers (\(reason))", level: .warning)
            return
        }

        let keepIndices = Set(idealCacheIndices())
        bufferedSentenceAudio = bufferedSentenceAudio.filter { keepIndices.contains($0.key) }
        enforceCachePolicy()
        log("Trimmed sentence cache (\(reason)); policy: \(memoryPolicyDescription)")
    }

    private func logStartupConfiguration() {
        log("Startup config: model=\(modelVariant.displayName), policy={\(memoryPolicyDescription)}")
    }
}
