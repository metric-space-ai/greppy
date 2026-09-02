import Foundation
import FSKit

@objc(GreppyFileSystem) final class GreppyFileSystem: FSUnaryFileSystem, FSUnaryFileSystemOperations {
    private static let containerID = UUID(uuidString: "F5BC37B2-0AB6-4D4C-AE1D-E3489732E590")!
    private var loadedResource: FSBlockDeviceResource?

    func probeResource(
        resource: FSResource,
        replyHandler: @escaping (FSProbeResult?, (any Error)?) -> Void
    ) {
        guard resource is FSBlockDeviceResource else {
            replyHandler(nil, POSIXError(.EINVAL))
            return
        }
        replyHandler(
            .usable(
                name: "Greppy Workspaces",
                containerID: FSContainerIdentifier(uuid: Self.containerID)
            ),
            nil
        )
    }

    func loadResource(
        resource: FSResource,
        options: FSTaskOptions,
        replyHandler: @escaping (FSVolume?, (any Error)?) -> Void
    ) {
        guard let blockResource = resource as? FSBlockDeviceResource else {
            replyHandler(nil, POSIXError(.EINVAL))
            return
        }
        do {
            let dataRoot = try Self.dataRoot()
            let core = try RustWorkspaceCore(dataRoot: dataRoot)
            loadedResource = blockResource
            containerStatus = .ready
            replyHandler(
                try GreppyFSVolume(resource: blockResource, core: core, dataRoot: dataRoot),
                nil
            )
        } catch {
            containerStatus = .blocked(status: error)
            replyHandler(nil, error)
        }
    }

    func unloadResource(
        resource: FSResource,
        options: FSTaskOptions,
        replyHandler: @escaping ((any Error)?) -> Void
    ) {
        guard let blockResource = resource as? FSBlockDeviceResource,
              blockResource == loadedResource else {
            replyHandler(POSIXError(.EINVAL))
            return
        }
        loadedResource = nil
        replyHandler(nil)
    }

    func didFinishLoading() {}

    private static func dataRoot() throws -> URL {
        guard let group = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "group.ai.metricspace.greppy"
        ) else {
            throw RustWorkspaceError.operation(
                "Greppy app-group store is unavailable; reinstall the signed macOS package"
            )
        }
        return group.appendingPathComponent("workspace", isDirectory: true)
    }
}
