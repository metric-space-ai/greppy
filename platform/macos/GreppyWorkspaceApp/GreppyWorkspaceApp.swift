import AppKit

@main
final class GreppyWorkspaceApp: NSObject, NSApplicationDelegate {
    static func main() {
        let application = NSApplication.shared
        let delegate = GreppyWorkspaceApp()
        application.delegate = delegate
        application.setActivationPolicy(.accessory)
        application.run()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let settings = URL(
            string: "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"
        )!
        NSWorkspace.shared.open(settings)
        NSApplication.shared.terminate(nil)
    }
}
