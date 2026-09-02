import Foundation

private struct ProviderCapabilities: Codable {
    let hardLinks = true
    let symbolicLinks = true
    let byteRangeLocks = true
    let memoryMaps = true
    let atomicRename = true
    let casePreserving = true

    enum CodingKeys: String, CodingKey {
        case hardLinks = "hard_links"
        case symbolicLinks = "symbolic_links"
        case byteRangeLocks = "byte_range_locks"
        case memoryMaps = "memory_maps"
        case atomicRename = "atomic_rename"
        case casePreserving = "case_preserving"
    }
}

private struct ProviderManifest: Codable {
    let protocolVersion = 1
    let adapterVersion: String
    let adapterKind = "fs-kit"
    let state = "ready"
    let instanceID: String
    let dataRoot: String
    let mountRoot: String
    let heartbeatUnixMilliseconds: UInt64
    let capabilities = ProviderCapabilities()

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case adapterVersion = "adapter_version"
        case adapterKind = "adapter_kind"
        case state
        case instanceID = "instance_id"
        case dataRoot = "data_root"
        case mountRoot = "mount_root"
        case heartbeatUnixMilliseconds = "heartbeat_unix_ms"
        case capabilities
    }
}

final class ProviderHeartbeat {
    let dataRoot: URL
    let mountRoot: URL
    let doctorRoot: URL
    private let instanceID = UUID().uuidString.lowercased()
    private let adapterVersion: String
    private let lock = NSLock()
    private var published = Data()
    private let timer: DispatchSourceTimer

    init(dataRoot: URL) throws {
        self.dataRoot = dataRoot.standardizedFileURL
        mountRoot = dataRoot.deletingLastPathComponent()
            .appendingPathComponent("workspace-mount", isDirectory: true)
            .standardizedFileURL
        doctorRoot = dataRoot.appendingPathComponent("provider-doctor", isDirectory: true)
        adapterVersion = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "0.4.0"
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "greppy.fskit.heartbeat"))
        try FileManager.default.createDirectory(
            at: doctorRoot,
            withIntermediateDirectories: true
        )
        try publish()
        timer.schedule(deadline: .now() + 2, repeating: 2)
        timer.setEventHandler { [weak self] in try? self?.publish() }
        timer.resume()
    }

    deinit { timer.cancel() }

    func manifestData() -> Data {
        lock.lock()
        defer { lock.unlock() }
        return published
    }

    private func publish() throws {
        let manifest = ProviderManifest(
            adapterVersion: adapterVersion,
            instanceID: instanceID,
            dataRoot: dataRoot.path,
            mountRoot: mountRoot.path,
            heartbeatUnixMilliseconds: UInt64(Date().timeIntervalSince1970 * 1_000)
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: dataRoot.appendingPathComponent("provider.json"), options: .atomic)
        lock.lock()
        published = data
        lock.unlock()
    }
}
