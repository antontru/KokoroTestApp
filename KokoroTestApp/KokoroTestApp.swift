import SwiftUI

@main
struct KokoroTestApp: App {
    let model = TestAppModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: model)
        }
    }
}
