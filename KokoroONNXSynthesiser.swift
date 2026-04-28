import AVFoundation
import Foundation
import OnnxRuntimeBindings
import libespeak_ng

final class KokoroONNXSynthesiser {
    enum ModelVariant: String, CaseIterable, Identifiable {
        case full
        case q4

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .full:
                return "Standard"
            case .q4:
                return "Quantised Q4"
            }
        }

        var fileName: String {
            switch self {
            case .full:
                return "model.onnx"
            case .q4:
                return "model_q4.onnx"
            }
        }
    }

    struct Diagnostics {
        let isCoreMLAvailable: Bool
        let isUsingCoreML: Bool
    }

    private let env: ORTEnv
    private var session: ORTSession
    private var diagnostics: Diagnostics
    private let sampleRate = 24_000.0
    private let cacheLock = NSLock()
    private var bufferCache: [String: AVAudioPCMBuffer] = [:]
    private var cacheOrder: [String] = []
    private let maxCachedSentences = 128

    private let voiceIDs: Set<String>
    private let voiceDataLock = NSLock()
    private var voiceDataCache: [String: [Float]] = [:]
    private let tokenizerVocab: [Character: Int64]
    private let padTokenID: Int64
    private static let espeakQueue = DispatchQueue(label: "kokoro.espeak.phonemizer")
    private static var espeakReady = false
    private static var espeakInitAttempted = false

    init(voiceIDs: [String], variant: ModelVariant) throws {
        self.voiceIDs = Set(voiceIDs)
        guard !self.voiceIDs.isEmpty else {
            throw NSError(domain: "KokoroONNX", code: 3, userInfo: [NSLocalizedDescriptionKey: "Voice list is empty"])
        }
        let tokenizerData = try Self.loadTokenizerVocab()
        self.tokenizerVocab = tokenizerData.vocab
        self.padTokenID = tokenizerData.padTokenID

        // Validate that at least one voice embedding file exists in either expected layout.
        let probeVoice = voiceIDs.first ?? "af_heart"
        guard Self.findVoiceURLInBundle(for: probeVoice) != nil else {
            throw NSError(domain: "KokoroONNX", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing voice embedding files in bundle (expected voices/<voice>.bin or voices.bin)"])
        }

        self.env = try ORTEnv(loggingLevel: .warning)
        let (session, diagnostics) = try KokoroONNXSynthesiser.makeSession(env: env, variant: variant)
        self.session = session
        self.diagnostics = diagnostics
    }

    func diagnosticsInfo() -> Diagnostics {
        diagnostics
    }

    func reload(variant: ModelVariant) throws {
        let (newSession, newDiagnostics) = try KokoroONNXSynthesiser.makeSession(env: env, variant: variant)
        session = newSession
        diagnostics = newDiagnostics
        clearCache()
        voiceDataLock.lock()
        voiceDataCache.removeAll()
        voiceDataLock.unlock()
    }

    func synthesiseBlocking(text: String, voiceID: String, speed: Float) throws -> AVAudioPCMBuffer {
        let cacheKey = makeCacheKey(text: text, voiceID: voiceID, speed: speed)
        if let cached = cachedBuffer(for: cacheKey) {
            return cached
        }

        let tokenIDs = tokenize(text, voiceID: voiceID)
        let styleVector = try styleEmbedding(for: voiceID, tokenCount: tokenIDs.count)

        var tokenData = tokenIDs
        let tokenMutable = NSMutableData(bytes: &tokenData, length: tokenData.count * MemoryLayout<Int64>.stride)
        let inputIDs = try ORTValue(tensorData: tokenMutable, elementType: .int64, shape: [1, NSNumber(value: tokenIDs.count)])

        var styleData = styleVector
        let styleMutable = NSMutableData(bytes: &styleData, length: styleData.count * MemoryLayout<Float>.stride)
        let style = try ORTValue(tensorData: styleMutable, elementType: .float, shape: [1, 256])

        var speedValue = [speed]
        let speedMutable = NSMutableData(bytes: &speedValue, length: MemoryLayout<Float>.stride)
        let speedTensor = try ORTValue(tensorData: speedMutable, elementType: .float, shape: [1])

        let outputs = try runModel(inputIDs: inputIDs, style: style, speed: speedTensor)

        let outputValue = outputs["waveform"] ?? outputs["wav"] ?? outputs["audio"] ?? outputs.values.first
        guard let outputValue else {
            throw NSError(domain: "KokoroONNX", code: 4, userInfo: [NSLocalizedDescriptionKey: "ONNX output is missing"])
        }

        let outputData = try outputValue.tensorData()
        guard outputData.length % MemoryLayout<Float>.stride == 0 else {
            throw NSError(domain: "KokoroONNX", code: 5, userInfo: [NSLocalizedDescriptionKey: "Unexpected output tensor format"])
        }

        let sampleCount = outputData.length / MemoryLayout<Float>.stride
        var samples = [Float](repeating: 0, count: sampleCount)
        outputData.getBytes(&samples, length: outputData.length)

        if samples.count > 1, samples[0].isFinite, !samples[1].isFinite {
            samples.removeFirst()
        }

        let finiteSamples = samples.map { $0.isFinite ? $0 : 0 }

        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(finiteSamples.count)) else {
            throw NSError(domain: "KokoroONNX", code: 6, userInfo: [NSLocalizedDescriptionKey: "Could not allocate audio buffer"])
        }
        buffer.frameLength = buffer.frameCapacity
        if let channel = buffer.floatChannelData?[0] {
            finiteSamples.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                UnsafeMutableRawPointer(channel).copyMemory(from: base, byteCount: finiteSamples.count * MemoryLayout<Float>.stride)
            }
        }

        cacheBuffer(buffer, for: cacheKey)
        return buffer
    }

    func clearCache() {
        cacheLock.lock()
        bufferCache.removeAll()
        cacheOrder.removeAll()
        cacheLock.unlock()
    }

    private func tokenize(_ text: String, voiceID: String) -> [Int64] {
        let normalized = normalizeTextForKokoro(text)
        let language = voiceID.hasPrefix("b") ? "en-gb" : "en-us"
        var phonemes = Self.phonemizeWithEspeak(normalized, language: language) ?? phonemizeEnglishHeuristic(normalized.lowercased())
        phonemes = postProcessPhonemes(phonemes, language: language)
        let payload = Array(phonemes.prefix(510))
        var tokens = [Int64]()
        tokens.reserveCapacity(payload.count + 2)
        tokens.append(padTokenID)
        for character in payload {
            if let id = tokenizerVocab[character] {
                tokens.append(id)
            } else if character.isWhitespace, let spaceID = tokenizerVocab[" "] {
                tokens.append(spaceID)
            }
        }
        tokens.append(padTokenID)
        return tokens
    }

    private func normalizeTextForKokoro(_ raw: String) -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return text }

        let ns = text as NSString
        if let regex = try? NSRegularExpression(pattern: "\\b[A-Z]{2,6}\\b") {
            let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).reversed()
            for match in matches {
                let word = ns.substring(with: match.range)
                let expanded = word.map { String($0) }.joined(separator: " ")
                text = (text as NSString).replacingCharacters(in: match.range, with: expanded)
            }
        }

        text = text
            .replacingOccurrences(of: "“", with: "\"")
            .replacingOccurrences(of: "”", with: "\"")
            .replacingOccurrences(of: "’", with: "'")
            .replacingOccurrences(of: "—", with: " — ")
            .replacingOccurrences(of: "…", with: " ... ")
        return text
    }

    private func postProcessPhonemes(_ phonemes: String, language: String) -> String {
        var ps = phonemes
            .replacingOccurrences(of: "kəkˈoːɹoʊ", with: "kˈoʊkəɹoʊ")
            .replacingOccurrences(of: "kəkˈɔːɹəʊ", with: "kˈəʊkəɹəʊ")
            .replacingOccurrences(of: "ʲ", with: "j")
            .replacingOccurrences(of: "r", with: "ɹ")
            .replacingOccurrences(of: "x", with: "k")
            .replacingOccurrences(of: "ɬ", with: "l")

        if let r1 = try? NSRegularExpression(pattern: "(?<=[a-zɹː])(?=hˈʌndɹɪd)") {
            ps = r1.stringByReplacingMatches(in: ps, range: NSRange(ps.startIndex..., in: ps), withTemplate: " ")
        }
        if let r2 = try? NSRegularExpression(pattern: " z(?=[;:,.!?¡¿—…\"«»“” ]|$)") {
            ps = r2.stringByReplacingMatches(in: ps, range: NSRange(ps.startIndex..., in: ps), withTemplate: "z")
        }
        if language == "en-us", let r3 = try? NSRegularExpression(pattern: "(?<=nˈaɪn)ti(?!ː)") {
            ps = r3.stringByReplacingMatches(in: ps, range: NSRange(ps.startIndex..., in: ps), withTemplate: "di")
        }

        ps = String(ps.filter { tokenizerVocab[$0] != nil || $0.isWhitespace })
        return ps.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func phonemizeEnglishHeuristic(_ text: String) -> String {
        var result = String()
        var currentWord = String()

        func flushWord() {
            guard !currentWord.isEmpty else { return }
            result.append(phonemizeWord(currentWord))
            currentWord.removeAll(keepingCapacity: true)
        }

        for ch in text {
            if ch.isLetter || ch == "'" {
                currentWord.append(ch)
            } else {
                flushWord()
                result.append(ch)
            }
        }
        flushWord()
        return result
    }

    private func phonemizeWord(_ word: String) -> String {
        let w = word.lowercased()
        let explicit: [String: String] = [
            "the": "ðə",
            "this": "ðɪs",
            "that": "ðæt",
            "these": "ðiːz",
            "those": "ðoʊz",
            "there": "ðɛr",
            "their": "ðɛr",
            "they": "ðeɪ",
            "them": "ðɛm",
            "then": "ðɛn",
            "than": "ðæn",
            "you": "ju",
            "your": "jɔr",
            "are": "ɑr",
            "was": "wʌz",
            "were": "wɚ",
            "of": "ʌv",
            "to": "tu",
            "and": "ænd",
            "for": "fɔr",
            "with": "wɪθ",
            "from": "frʌm",
        ]
        if let mapped = explicit[w] {
            return mapped
        }

        let voicedThWords: Set<String> = ["the", "this", "that", "these", "those", "their", "there", "they", "them", "then", "than", "though", "thus"]
        let chars = Array(w)
        var out = String()
        var i = 0

        while i < chars.count {
            let remaining = String(chars[i...])
            let next = (i + 1 < chars.count) ? chars[i + 1] : "\0"
            let next2 = (i + 2 < chars.count) ? chars[i + 2] : "\0"
            let next3 = (i + 3 < chars.count) ? chars[i + 3] : "\0"

            if remaining.hasPrefix("tion") { out += "ʃən"; i += 4; continue }
            if remaining.hasPrefix("sion") { out += "ʒən"; i += 4; continue }
            if remaining.hasPrefix("ture") { out += "tʃɚ"; i += 4; continue }
            if remaining.hasPrefix("ough") { out += "oʊ"; i += 4; continue }
            if remaining.hasPrefix("eigh") { out += "eɪ"; i += 4; continue }
            if remaining.hasPrefix("igh") { out += "aɪ"; i += 3; continue }
            if remaining.hasPrefix("ph") { out += "f"; i += 2; continue }
            if remaining.hasPrefix("sh") { out += "ʃ"; i += 2; continue }
            if remaining.hasPrefix("ch") { out += "tʃ"; i += 2; continue }
            if remaining.hasPrefix("zh") { out += "ʒ"; i += 2; continue }
            if remaining.hasPrefix("ng") { out += "ŋ"; i += 2; continue }
            if remaining.hasPrefix("wh") { out += "w"; i += 2; continue }
            if remaining.hasPrefix("ck") { out += "k"; i += 2; continue }
            if remaining.hasPrefix("qu") { out += "kw"; i += 2; continue }
            if remaining.hasPrefix("ee") || remaining.hasPrefix("ea") { out += "i"; i += 2; continue }
            if remaining.hasPrefix("oo") { out += "u"; i += 2; continue }
            if remaining.hasPrefix("ai") || remaining.hasPrefix("ay") { out += "eɪ"; i += 2; continue }
            if remaining.hasPrefix("oa") || remaining.hasPrefix("ow") { out += "oʊ"; i += 2; continue }
            if remaining.hasPrefix("ou") { out += "aʊ"; i += 2; continue }
            if remaining.hasPrefix("oi") || remaining.hasPrefix("oy") { out += "ɔɪ"; i += 2; continue }
            if remaining.hasPrefix("au") || remaining.hasPrefix("aw") { out += "ɔ"; i += 2; continue }
            if remaining.hasPrefix("er") || remaining.hasPrefix("ir") || remaining.hasPrefix("ur") { out += "ɚ"; i += 2; continue }
            if remaining.hasPrefix("ar") { out += "ɑr"; i += 2; continue }
            if remaining.hasPrefix("or") { out += "ɔr"; i += 2; continue }
            if remaining.hasPrefix("th") {
                out += voicedThWords.contains(w) ? "ð" : "θ"
                i += 2
                continue
            }

            let c = chars[i]
            switch c {
            case "a": out += "æ"
            case "b": out += "b"
            case "c":
                if ["e", "i", "y"].contains(next) { out += "s" } else { out += "k" }
            case "d": out += "d"
            case "e":
                if i == chars.count - 1, chars.count > 2 { break }
                out += "ɛ"
            case "f": out += "f"
            case "g":
                if ["e", "i", "y"].contains(next) { out += "dʒ" } else { out += "ɡ" }
            case "h": out += "h"
            case "i": out += "ɪ"
            case "j": out += "dʒ"
            case "k": out += "k"
            case "l": out += "l"
            case "m": out += "m"
            case "n": out += "n"
            case "o": out += "o"
            case "p": out += "p"
            case "q": out += "k"
            case "r": out += "r"
            case "s": out += "s"
            case "t": out += "t"
            case "u": out += "ʌ"
            case "v": out += "v"
            case "w": out += "w"
            case "x": out += "ks"
            case "y":
                if i == chars.count - 1 { out += "i" } else { out += "j" }
            case "z": out += "z"
            default: out.append(c)
            }
            i += 1

            _ = next2
            _ = next3
        }

        return out
    }

    private func styleEmbedding(for voiceID: String, tokenCount: Int) throws -> [Float] {
        guard voiceIDs.contains(voiceID) else {
            throw NSError(domain: "KokoroONNX", code: 7, userInfo: [NSLocalizedDescriptionKey: "Unknown voice: \(voiceID)"])
        }
        let voiceData = try loadVoiceData(for: voiceID)
        let styleCount = voiceData.count / 256
        guard styleCount > 0 else {
            throw NSError(domain: "KokoroONNX", code: 11, userInfo: [NSLocalizedDescriptionKey: "Voice embedding data is empty for \(voiceID)"])
        }
        let clampedTokenIndex = max(0, min(tokenCount - 1, styleCount - 1))
        let start = clampedTokenIndex * 256
        let end = start + 256
        return Array(voiceData[start..<end])
    }

    private static func makeSession(env: ORTEnv, variant: ModelVariant) throws -> (ORTSession, Diagnostics) {
        let modelURL = Bundle.main.url(forResource: variant.fileName, withExtension: nil)
            ?? Bundle.main.url(forResource: variant.fileName, withExtension: nil, subdirectory: "onnx")
        guard let modelURL else {
            throw NSError(domain: "KokoroONNX", code: 9, userInfo: [NSLocalizedDescriptionKey: "Missing \(variant.fileName) in bundle"])
        }

        let options = try ORTSessionOptions()
        var usingCoreML = false
        let coreMLAvailable = ORTIsCoreMLExecutionProviderAvailable()

        if coreMLAvailable {
            do {
                let coreMLOptions = ORTCoreMLExecutionProviderOptions()
                coreMLOptions.enableOnSubgraphs = true
                coreMLOptions.useCPUOnly = false
                try options.appendCoreMLExecutionProvider(with: coreMLOptions)
                usingCoreML = true
            } catch {
                usingCoreML = false
            }
        }

        if !usingCoreML {
            _ = try? options.appendExecutionProvider("xnnpack", providerOptions: [:])
        }

        let session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
        return (session, Diagnostics(isCoreMLAvailable: coreMLAvailable, isUsingCoreML: usingCoreML))
    }

    private func makeCacheKey(text: String, voiceID: String, speed: Float) -> String {
        let normalizedText = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let speedKey = String(format: "%.3f", speed)
        return "\(voiceID)|\(speedKey)|\(normalizedText)"
    }

    private func cachedBuffer(for key: String) -> AVAudioPCMBuffer? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return bufferCache[key]
    }

    private func cacheBuffer(_ buffer: AVAudioPCMBuffer, for key: String) {
        cacheLock.lock()
        defer { cacheLock.unlock() }

        if bufferCache[key] == nil {
            cacheOrder.append(key)
        }
        bufferCache[key] = buffer

        while cacheOrder.count > maxCachedSentences {
            let oldest = cacheOrder.removeFirst()
            bufferCache.removeValue(forKey: oldest)
        }
    }

    private func runModel(inputIDs: ORTValue, style: ORTValue, speed: ORTValue) throws -> [String: ORTValue] {
        let inputs: [String: ORTValue] = [
            "input_ids": inputIDs,
            "style": style,
            "speed": speed,
        ]

        let candidateOutputs = ["waveform", "wav", "audio"]
        var lastError: Error?

        for outputName in candidateOutputs {
            do {
                let outputs = try session.run(withInputs: inputs, outputNames: Set([outputName]), runOptions: nil)
                if !outputs.isEmpty {
                    return outputs
                }
            } catch {
                lastError = error
                continue
            }
        }

        // Final attempt: ask for all outputs by using an empty output set.
        do {
            let outputs = try session.run(withInputs: inputs, outputNames: Set<String>(), runOptions: nil)
            if !outputs.isEmpty {
                return outputs
            }
        } catch {
            lastError = error
        }

        throw lastError ?? NSError(
            domain: "KokoroONNX",
            code: 13,
            userInfo: [NSLocalizedDescriptionKey: "Could not resolve ONNX output tensor name"]
        )
    }

    private func findVoiceURL(for voiceID: String) -> URL? {
        Self.findVoiceURLInBundle(for: voiceID)
    }

    private static func findVoiceURLInBundle(for voiceID: String) -> URL? {
        Bundle.main.url(forResource: voiceID, withExtension: "bin", subdirectory: "voices")
            ?? Bundle.main.url(forResource: voiceID, withExtension: "bin")
            ?? Bundle.main.url(forResource: "voices", withExtension: "bin")
    }

    private static func loadTokenizerVocab() throws -> (vocab: [Character: Int64], padTokenID: Int64) {
        guard let tokenizerURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
            throw NSError(domain: "KokoroONNX", code: 14, userInfo: [NSLocalizedDescriptionKey: "Missing tokenizer.json in bundle"])
        }

        let data = try Data(contentsOf: tokenizerURL)
        let root = try JSONDecoder().decode(TokenizerRoot.self, from: data)
        var vocab: [Character: Int64] = [:]
        vocab.reserveCapacity(root.model.vocab.count)
        for (key, value) in root.model.vocab {
            guard key.count == 1, let char = key.first else { continue }
            vocab[char] = value
        }
        guard !vocab.isEmpty else {
            throw NSError(domain: "KokoroONNX", code: 15, userInfo: [NSLocalizedDescriptionKey: "Tokenizer vocab is empty"])
        }

        let padTokenID = root.model.vocab["$"] ?? 0
        return (vocab, padTokenID)
    }

    private static func ensureEspeakInitialized() -> Bool {
        withEspeakQueue {
            if espeakReady {
                return true
            }
            if espeakInitAttempted {
                return false
            }
            espeakInitAttempted = true

            do {
                let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                try EspeakLib.ensureBundleInstalled(inRoot: docs)
                espeak_ng_InitializePath(docs.path)
                guard espeak_ng_Initialize(nil) == ENS_OK else { return false }
                guard espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, 0, nil) == ENS_OK else { return false }
                guard espeak_ng_SetVoiceByName("en-us") == ENS_OK else { return false }
                espeakReady = true
                return true
            } catch {
                return false
            }
        }
    }

    private static func phonemizeWithEspeak(_ text: String, language: String) -> String? {
        guard !text.isEmpty else { return text }
        guard ensureEspeakInitialized() else { return nil }

        return withEspeakQueue {
            let mode = Int32(espeakPHONEMES_IPA | espeakPHONEMES_TIE)
            let charsMode = Int32(espeakCHARS_UTF8)
            var result = ""

            if espeak_ng_SetVoiceByName(language) != ENS_OK,
               espeak_ng_SetVoiceByName("en-us") != ENS_OK
            {
                return nil
            }

            var word = ""
            func flushWord() {
                guard !word.isEmpty else { return }
                var input = Array(word.utf8CString)
                input.withUnsafeMutableBufferPointer { buffer in
                    guard var ptr = buffer.baseAddress else { return }
                    var textPtr: UnsafeRawPointer? = UnsafeRawPointer(ptr)
                    while let phonemePtr = espeak_TextToPhonemes(&textPtr, charsMode, mode) {
                        let chunk = String(cString: phonemePtr)
                        if chunk.isEmpty { break }
                        result += chunk
                    }
                }
                word.removeAll(keepingCapacity: true)
            }

            for ch in text {
                if ch.isLetter || ch == "'" {
                    word.append(ch)
                } else {
                    flushWord()
                    result.append(ch)
                }
            }
            flushWord()

            let cleaned = result
                .replacingOccurrences(of: "_", with: " ")
                .replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "  ", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return cleaned.isEmpty ? nil : cleaned
        }
    }

    private static func withEspeakQueue<T>(_ block: @escaping () -> T) -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var value: T!
        espeakQueue.async {
            value = block()
            semaphore.signal()
        }
        semaphore.wait()
        return value
    }

    private struct TokenizerRoot: Decodable {
        let model: TokenizerModel
    }

    private struct TokenizerModel: Decodable {
        let vocab: [String: Int64]
    }

    private func loadVoiceData(for voiceID: String) throws -> [Float] {
        voiceDataLock.lock()
        if let cached = voiceDataCache[voiceID] {
            voiceDataLock.unlock()
            return cached
        }
        voiceDataLock.unlock()

        guard let url = findVoiceURL(for: voiceID) else {
            throw NSError(domain: "KokoroONNX", code: 12, userInfo: [NSLocalizedDescriptionKey: "Missing voice file for \(voiceID)"])
        }

        let data = try Data(contentsOf: url)
        guard !data.isEmpty else {
            throw NSError(domain: "KokoroONNX", code: 10, userInfo: [NSLocalizedDescriptionKey: "Voice file is empty for \(voiceID)"])
        }
        guard data.count % MemoryLayout<Float>.stride == 0 else {
            throw NSError(domain: "KokoroONNX", code: 2, userInfo: [NSLocalizedDescriptionKey: "Voice file size is invalid for \(voiceID)"])
        }

        let floatCount = data.count / MemoryLayout<Float>.stride
        guard floatCount % 256 == 0 else {
            throw NSError(domain: "KokoroONNX", code: 8, userInfo: [NSLocalizedDescriptionKey: "Voice embedding length is invalid for \(voiceID)"])
        }

        var floats = [Float](repeating: 0, count: floatCount)
        _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }

        voiceDataLock.lock()
        voiceDataCache[voiceID] = floats
        voiceDataLock.unlock()
        return floats
    }
}
