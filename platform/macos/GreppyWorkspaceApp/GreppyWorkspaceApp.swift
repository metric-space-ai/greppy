import AppKit
import FSKit
import ObjectiveC

@main
enum GreppyWorkspaceApp {
    static func main() {
        guard openFileSystemExtensionsSettings() else {
            fatalError("macOS refused to open File System Extensions settings")
        }
    }

    private static func openFileSystemExtensionsSettings() -> Bool {
        // Apple added the direct FSKit settings API after the oldest SDK Greppy
        // supports. Resolve it dynamically so the same host remains deployable
        // on macOS 15.4 while newer systems open the exact category pane.
        let client = FSClient.shared
        let selector = NSSelectorFromString("openFileSystemExtensionsSettings")
        if let method = class_getInstanceMethod(FSClient.self, selector) {
            typealias OpenSettings = @convention(c) (AnyObject, Selector) -> Bool
            let implementation = unsafeBitCast(
                method_getImplementation(method),
                to: OpenSettings.self
            )
            return implementation(client, selector)
        }

        // Older systems have no public category-specific API. This opens the
        // Extensions pane itself rather than the unrelated Login Items root.
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.ExtensionsPreferences"
        ) else {
            return false
        }
        return NSWorkspace.shared.open(url)
    }
}
