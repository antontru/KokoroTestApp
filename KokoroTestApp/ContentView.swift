import PDFKit
import SwiftUI
import UniformTypeIdentifiers

#if os(iOS)
struct ContentView: View {
    @ObservedObject var viewModel: TestAppModel

    @State private var inputText: String = ""
    @State private var isReadView: Bool = false
    @FocusState private var isEditorFocused: Bool

    @State private var isShowingImporter: Bool = false
    @State private var isShowingSettings: Bool = false
    @State private var isShowingDebugOverlay: Bool = false

    @State private var importErrorMessage: String?
    @State private var titleTapTimestamps: [Date] = []

    var body: some View {
        #if targetEnvironment(simulator)
        Text("Not supported on Simulator")
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(
                LinearGradient(
                    colors: [
                        Color(red: 0.09, green: 0.09, blue: 0.1),
                        Color.black,
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .foregroundStyle(.white)
        #else
        NavigationStack {
            VStack(spacing: 0) {
                if let importErrorMessage {
                    Text(importErrorMessage)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                        .transition(.opacity)
                }

                Group {
                    if isReadView {
                        ReadView(viewModel: viewModel)
                    } else {
                        EditView(
                            inputText: $inputText,
                            isEditorFocused: $isEditorFocused,
                            onClear: clearInputText,
                            wordCount: wordCount
                        )
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background(Color.black)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if isReadView {
                        Button("Edit") {
                            isReadView = false
                        }
                        .foregroundStyle(.white)
                    } else {
                        Button(action: registerTitleTap) {
                            Text("Kokoro Reader")
                                .font(.headline)
                                .foregroundStyle(.white)
                        }
                    }
                }

                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        isShowingImporter = true
                    } label: {
                        Image(systemName: "doc.badge.plus")
                            .foregroundStyle(.white)
                    }

                    Button {
                        isShowingSettings = true
                    } label: {
                        Image(systemName: "line.3.horizontal")
                            .foregroundStyle(.white)
                    }
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                PlayerBar(
                    viewModel: viewModel,
                    canPlay: !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                    onPrimaryAction: handlePrimaryActionTapped
                )
                .frame(maxWidth: .infinity)
            }
            .fileImporter(
                isPresented: $isShowingImporter,
                allowedContentTypes: [.plainText, .pdf],
                allowsMultipleSelection: false
            ) { result in
                handleImportResult(result)
            }
            .sheet(isPresented: $isShowingSettings) {
                SettingsSheet(viewModel: viewModel)
                    .presentationDetents([.large])
                    .presentationDragIndicator(.visible)
            }
            .fullScreenCover(isPresented: $isShowingDebugOverlay) {
                DebugLogOverlay(
                    logs: viewModel.debugLogs,
                    onClear: viewModel.clearLogs,
                    onClose: { isShowingDebugOverlay = false }
                )
            }
            .onChange(of: inputText) { _, newValue in
                if newValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    if viewModel.hasLoadedText {
                        viewModel.clearText()
                    }
                    if isReadView {
                        isReadView = false
                    }
                }
            }
            .preferredColorScheme(.dark)
        }
        #endif
    }

    private var wordCount: Int {
        inputText.split { $0.isWhitespace || $0.isNewline }.count
    }

    private func clearInputText() {
        isEditorFocused = false
        inputText = ""
        isReadView = false
        viewModel.clearText()
    }

    private func handlePrimaryActionTapped() {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isEditorFocused = false
        isReadView = true
        viewModel.togglePlay(using: inputText)
    }

    private func handleImportResult(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }

            do {
                let imported = try readImportedFile(from: url)
                inputText = imported
                isReadView = false
                viewModel.importText(imported)
                clearImportError()
            } catch {
                showImportError("Could not import file")
            }
        case .failure:
            showImportError("Could not import file")
        }
    }

    private func readImportedFile(from url: URL) throws -> String {
        let didAccessSecurityScopedResource = url.startAccessingSecurityScopedResource()
        defer {
            if didAccessSecurityScopedResource {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let typeIdentifier = try url.resourceValues(forKeys: [.contentTypeKey]).contentType

        if typeIdentifier?.conforms(to: .pdf) == true {
            return try extractPDFText(from: url)
        }

        return try String(contentsOf: url, encoding: .utf8)
    }

    private func extractPDFText(from url: URL) throws -> String {
        guard let document = PDFDocument(url: url) else {
            throw NSError(domain: "KokoroTestApp", code: 50, userInfo: [NSLocalizedDescriptionKey: "Invalid PDF"])
        }

        var collected: [String] = []
        for pageIndex in 0..<document.pageCount {
            if let pageText = document.page(at: pageIndex)?.string {
                collected.append(pageText)
            }
        }
        return collected.joined(separator: "\n\n")
    }

    private func showImportError(_ message: String) {
        importErrorMessage = message
        Task {
            try? await Task.sleep(for: .seconds(3))
            await MainActor.run {
                if importErrorMessage == message {
                    importErrorMessage = nil
                }
            }
        }
    }

    private func clearImportError() {
        importErrorMessage = nil
    }

    private func registerTitleTap() {
        let now = Date()
        titleTapTimestamps.append(now)
        titleTapTimestamps = titleTapTimestamps.filter { now.timeIntervalSince($0) <= 2.0 }

        if titleTapTimestamps.count >= 5 {
            titleTapTimestamps.removeAll()
            isShowingDebugOverlay = true
        }
    }
}

private struct EditView: View {
    @Binding var inputText: String
    @FocusState.Binding var isEditorFocused: Bool
    let onClear: () -> Void
    let wordCount: Int

    var body: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .topLeading) {
                TextEditor(text: $inputText)
                    .focused($isEditorFocused)
                    .padding(12)
                    .scrollContentBackground(.hidden)
                    .background(Color.clear)

                if inputText.isEmpty {
                    Text("Paste your text here, or import a file…")
                        .foregroundStyle(.gray)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 20)
                }
            }

            if !inputText.isEmpty {
                Divider()
                    .overlay(Color.white.opacity(0.1))

                HStack {
                    Text("\(wordCount) words")
                        .font(.subheadline)
                        .foregroundStyle(.gray)

                    Spacer()

                    Button("Clear", role: .destructive, action: onClear)
                        .font(.subheadline)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color.white.opacity(0.04))
            }
        }
        .background(Color.clear)
    }
}

private struct ReadView: View {
    @ObservedObject var viewModel: TestAppModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(viewModel.sentences.enumerated()), id: \.offset) { index, sentence in
                        Button {
                            viewModel.jumpToSentence(index)
                        } label: {
                            Text(sentence)
                                .foregroundStyle(textColor(for: index))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(backgroundColor(for: index))
                                .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                        .id(index)
                    }
                }
                .padding(.horizontal, 18)
                .padding(.top, 10)
                .padding(.bottom, 20)
            }
            .onChange(of: viewModel.currentSentenceIndex) { _, newValue in
                guard let newValue else { return }
                withAnimation(.easeInOut(duration: 0.2)) {
                    proxy.scrollTo(newValue, anchor: .center)
                }
            }
        }
    }

    private func backgroundColor(for index: Int) -> Color {
        guard let current = viewModel.currentSentenceIndex else { return .clear }

        if index == current {
            return Color.green.opacity(0.28)
        }
        return .clear
    }

    private func textColor(for index: Int) -> Color {
        guard let current = viewModel.currentSentenceIndex else {
            return .white
        }

        if index == current {
            return .white
        }

        return index < current ? Color.white.opacity(0.62) : .white
    }
}

private struct PlayerBar: View {
    @ObservedObject var viewModel: TestAppModel
    let canPlay: Bool
    let onPrimaryAction: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            VStack(spacing: 8) {
                HStack(spacing: 10) {
                    Image(systemName: "waveform.circle.fill")
                        .font(.title3)
                        .foregroundStyle(Color.green.opacity(0.95))

                    VStack(alignment: .leading, spacing: 2) {
                        Text(statusTitle)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.white)
                        Text(statusSubtitle)
                            .font(.caption)
                            .foregroundStyle(.gray)
                    }

                    Spacer()
                    if viewModel.playbackPhase == .generating {
                        ProgressView()
                            .tint(.green)
                    }
                }

                ProgressView(value: viewModel.generationProgress, total: 1)
                    .tint(.green)
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)

            HStack(spacing: 18) {
                playerButton(systemImage: "backward.end.fill", disabled: !viewModel.hasLoadedText) {
                    viewModel.skipPrevious()
                }

                playPauseButton

                playerButton(systemImage: "stop.fill", disabled: !viewModel.hasLoadedText) {
                    viewModel.stopPlayback()
                }

                playerButton(systemImage: "forward.end.fill", disabled: !viewModel.hasLoadedText) {
                    viewModel.skipNext()
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 18)
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.11, green: 0.11, blue: 0.13),
                    Color(red: 0.06, green: 0.06, blue: 0.08),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        )
        .overlay(alignment: .top) {
            Divider().overlay(Color.white.opacity(0.1))
        }
    }

    private var currentDisplayIndex: Int {
        guard let current = viewModel.currentSentenceIndex else {
            return 0
        }
        return current + 1
    }

    private var playButtonDisabled: Bool {
        !canPlay
    }

    private var statusTitle: String {
        switch viewModel.playbackPhase {
        case .playing:
            return "Now Playing"
        case .paused:
            return "Paused"
        case .generating:
            return "Buffering Speech"
        case .done:
            return "Playback Complete"
        case .idle:
            return "Ready"
        }
    }

    private var statusSubtitle: String {
        if viewModel.sentences.isEmpty {
            return "Paste text or import a file to begin"
        }
        return "\(currentDisplayIndex) / \(viewModel.sentences.count)"
    }

    @ViewBuilder
    private var playPauseButton: some View {
        Button(action: onPrimaryAction) {
            ZStack {
                Circle()
                    .fill(Color.green.opacity(playButtonDisabled ? 0.4 : 0.95))
                    .frame(width: 56, height: 56)

                if viewModel.playbackPhase == .generating {
                    ProgressView()
                        .tint(.white)
                } else if viewModel.playbackPhase == .playing {
                    Image(systemName: "pause.fill")
                        .foregroundStyle(.white)
                } else {
                    Image(systemName: "play.fill")
                        .foregroundStyle(.white)
                }
            }
        }
        .disabled(playButtonDisabled)
        .shadow(color: Color.green.opacity(0.35), radius: 12, x: 0, y: 6)
    }

    private func playerButton(systemImage: String, disabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .foregroundStyle(disabled ? Color.gray : Color.white)
                .frame(width: 42, height: 42)
                .background(Color.white.opacity(0.06))
                .clipShape(Circle())
        }
        .disabled(disabled)
    }
}

private struct SettingsSheet: View {
    @ObservedObject var viewModel: TestAppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Voice") {
                    if !viewModel.usVoices.isEmpty {
                        Text("American English")
                            .font(.caption)
                            .foregroundStyle(.gray)

                        ForEach(viewModel.usVoices) { voice in
                            VoiceRow(
                                voice: voice,
                                isSelected: viewModel.selectedVoiceID == voice.id
                            ) {
                                viewModel.selectVoice(voice.id)
                            }
                        }
                    }

                    if !viewModel.gbVoices.isEmpty {
                        Text("British English")
                            .font(.caption)
                            .foregroundStyle(.gray)
                            .padding(.top, 6)

                        ForEach(viewModel.gbVoices) { voice in
                            VoiceRow(
                                voice: voice,
                                isSelected: viewModel.selectedVoiceID == voice.id
                            ) {
                                viewModel.selectVoice(voice.id)
                            }
                        }
                    }
                }

                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Speed")
                                .font(.headline)
                            Spacer()
                            Text("\(String(format: "%.1f", viewModel.speed))x")
                                .foregroundStyle(.gray)
                        }

                        Slider(
                            value: Binding(
                                get: { viewModel.speed },
                                set: { viewModel.updateSpeed($0) }
                            ),
                            in: 0.5...2.0,
                            step: 0.1
                        )

                        HStack {
                            Text("0.5x")
                            Spacer()
                            Text("2.0x")
                        }
                        .font(.caption)
                        .foregroundStyle(.gray)
                    }
                    .padding(.vertical, 4)
                }

                Section("Model") {
                    Picker("Variant", selection: Binding(
                        get: { viewModel.modelVariant },
                        set: { viewModel.updateModelVariant($0) }
                    )) {
                        ForEach(KokoroONNXSynthesiser.ModelVariant.allCases) { variant in
                            Text(variant.displayName).tag(variant)
                        }
                    }
                    .pickerStyle(.segmented)

                    if let warning = viewModel.coreMLWarningMessage {
                        Text(warning)
                            .font(.caption)
                            .foregroundStyle(.yellow)
                    }
                }

                Section("Memory") {
                    Text(viewModel.memoryPolicyDescription)
                        .font(.caption)
                        .foregroundStyle(.gray)
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

private struct VoiceRow: View {
    let voice: VoiceOption
    let isSelected: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 10) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? Color.green : Color.gray)

                VStack(alignment: .leading, spacing: 3) {
                    Text(voice.name)
                        .foregroundStyle(.white)
                    Text(voice.descriptor)
                        .font(.caption)
                        .foregroundStyle(.gray)
                }
            }
            .padding(.vertical, 6)
        }
        .buttonStyle(.plain)
        .listRowBackground(isSelected ? Color.green.opacity(0.12) : Color.clear)
    }
}

private struct DebugLogOverlay: View {
    let logs: [DebugLogEntry]
    let onClear: () -> Void
    let onClose: () -> Void

    private let formatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(logs) { entry in
                            Text("[\(formatter.string(from: entry.timestamp))] \(entry.message)")
                                .font(.system(size: 12, weight: .regular, design: .monospaced))
                                .foregroundStyle(color(for: entry.level))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(entry.id)
                        }
                    }
                    .padding(14)
                }
                .onChange(of: logs.count) { _, _ in
                    guard let last = logs.last else { return }
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
                .onAppear {
                    guard let last = logs.last else { return }
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
            .navigationTitle("Debug Log")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Clear") {
                        onClear()
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark")
                    }
                }
            }
            .background(Color.black)
        }
        .preferredColorScheme(.dark)
    }

    private func color(for level: DebugLogEntry.Level) -> Color {
        switch level {
        case .info:
            return .white
        case .warning:
            return .yellow
        case .error:
            return .red
        }
    }
}

#Preview {
    ContentView(viewModel: TestAppModel())
}
#else
struct ContentView: View {
    let viewModel: TestAppModel

    var body: some View {
        Text("iPhone only")
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
#endif
