import FSKit
import Foundation

// Entry point: the extension is linked with `-e _NSExtensionMain` (like Apple's
// msdos/exfat FSKit modules) and `EXExtensionPrincipalClass = GreppyFileSystem`.
// Foundation's NSExtensionMain instantiates the principal class and serves the
// ExtensionKit session; a Swift `@main` conforming to `UnaryFileSystemExtension`
// returned immediately on macOS 26.2 and the process exited before fskitd could
// obtain its endpoint.
