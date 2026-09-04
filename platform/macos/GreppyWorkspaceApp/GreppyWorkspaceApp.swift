import AppKit
import FSKit
import ObjectiveC

@main
enum GreppyWorkspaceApp {
    static func main() {
        if CommandLine.arguments.dropFirst().contains("--fskit-status") {
            reportFileSystemExtensionStatus()
            dispatchMain()
        }
        guard openFileSystemExtensionsSettings() else {
            fatalError("macOS refused to open File System Extensions settings")
        }
    }

    private static func reportFileSystemExtensionStatus() {
        DispatchQueue.global().asyncAfter(deadline: .now() + 10) {
            fputs("FSClient installedExtensions query timed out\n", stderr)
            exit(75)
        }
        Task {
            do {
                let modules = try await FSClient.shared.installedExtensions
                let matches = modules.filter {
                    $0.bundleIdentifier == "ai.metricspace.greppy.workspacefs.extension"
                }.map {
                    ["bundle_id": $0.bundleIdentifier, "path": $0.url.path,
                     "enabled": $0.isEnabled] as [String: Any]
                }
                let data = try JSONSerialization.data(
                    withJSONObject: ["schema": "greppy.fskit-status.v1", "modules": matches],
                    options: [.sortedKeys]
                )
                print(String(decoding: data, as: UTF8.self))
                exit(0)
            } catch {
                fputs("FSClient installedExtensions query failed: \(error)\n", stderr)
                exit(1)
            }
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
