import Darwin
import Foundation
import FSKit

final class GreppyFSVolume: FSVolume {
    private struct DirectoryChild {
        let item: GreppyFSItem
        let metadata: RustWorkspaceMetadata?
    }

    private static let volumeID = UUID(uuidString: "70B5BBC5-38F6-4F2A-8F8C-1A3FC950A392")!
    private let resource: FSResource
    private let core: RustWorkspaceCore
    private let heartbeat: ProviderHeartbeat
    private let mountTime: timespec = {
        var value = timespec()
        clock_gettime(CLOCK_REALTIME, &value)
        return value
    }()
    private let cache = NSLock()
    private var items: [GreppyFSItem.Location: GreppyFSItem] = [:]
    private var nextIdentifier: UInt64 = FSItem.Identifier.rootDirectory.rawValue + 16

    private lazy var root = item(
        location: .root,
        name: FSFileName(string: "/"),
        fixedIdentifier: .rootDirectory
    )

    init(resource: FSResource, core: RustWorkspaceCore, dataRoot: URL) throws {
        self.resource = resource
        self.core = core
        heartbeat = try ProviderHeartbeat(dataRoot: dataRoot)
        super.init(
            volumeID: FSVolume.Identifier(uuid: Self.volumeID),
            volumeName: FSFileName(string: "Greppy Workspaces")
        )
    }

    private func item(
        location: GreppyFSItem.Location,
        name: FSFileName,
        fixedIdentifier: FSItem.Identifier? = nil
    ) -> GreppyFSItem {
        cache.lock()
        defer { cache.unlock() }
        if let existing = items[location] { return existing }
        let identifier: FSItem.Identifier
        if let fixedIdentifier {
            identifier = fixedIdentifier
        } else {
            identifier = FSItem.Identifier(rawValue: nextIdentifier) ?? .invalid
            nextIdentifier += 1
        }
        let value = GreppyFSItem(location: location, name: name, identifier: identifier)
        items[location] = value
        return value
    }

    private func relativePath(parent: GreppyFSItem, name: FSFileName) throws -> String {
        guard let component = name.string,
              !component.isEmpty,
              component != ".",
              component != "..",
              !component.contains("/"),
              !component.contains("\\") else {
            throw posix(EINVAL)
        }
        switch parent.location {
        case .workspace:
            return component
        case .path(_, let relative):
            return relative.isEmpty ? component : "\(relative)/\(component)"
        case .doctor:
            return component
        case .doctorPath(let relative):
            return relative.isEmpty ? component : "\(relative)/\(component)"
        case .root, .workspaces, .marker:
            throw posix(EPERM)
        }
    }

    private func workspace(_ item: FSItem) throws -> GreppyFSItem {
        guard let item = item as? GreppyFSItem else { throw posix(EIO) }
        return item
    }

    private func privateInode(for item: GreppyFSItem) throws -> (String, UInt64) {
        guard let (workspaceID, relative) = item.workspaceAndPath, !relative.isEmpty else {
            throw posix(EINVAL)
        }
        if let inode = item.boundPrivateInode() {
            try core.promoteFileInode(workspace: workspaceID, inode: inode)
            return (workspaceID, inode)
        }
        let inode = try core.openFileInode(workspace: workspaceID, path: relative)
        return (workspaceID, item.bindPrivateInode(inode))
    }

    private func readableInode(for item: GreppyFSItem) throws -> (String, UInt64) {
        guard let (workspaceID, relative) = item.workspaceAndPath, !relative.isEmpty else {
            throw posix(EINVAL)
        }
        if let inode = item.boundPrivateInode() { return (workspaceID, inode) }
        let inode = try core.openFileReadOnlyInode(workspace: workspaceID, path: relative)
        return (workspaceID, item.bindPrivateInode(inode))
    }

    private func posix(_ code: Int32) -> Error {
        fs_errorForPOSIXError(code)
    }

    private func evict(_ location: GreppyFSItem.Location) {
        cache.lock()
        items[location] = nil
        cache.unlock()
    }
}

extension GreppyFSVolume: FSVolume.PathConfOperations {
    var maximumLinkCount: Int { Int(Int32.max) }
    var maximumNameLength: Int { 255 }
    var restrictsOwnershipChanges: Bool { true }
    var truncatesLongNames: Bool { false }
    var maximumXattrSize: Int { 0 }
    var maximumFileSize: UInt64 { UInt64.max }
}

extension GreppyFSVolume: FSVolume.Operations {
    var supportedVolumeCapabilities: FSVolume.SupportedCapabilities {
        let capabilities = FSVolume.SupportedCapabilities()
        capabilities.supportsHardLinks = true
        capabilities.supportsSymbolicLinks = true
        capabilities.supportsPersistentObjectIDs = false
        capabilities.doesNotSupportVolumeSizes = true
        capabilities.supportsHiddenFiles = true
        capabilities.supports64BitObjectIDs = true
        capabilities.caseFormat = .sensitive
        return capabilities
    }

    var volumeStatistics: FSStatFSResult {
        let value = FSStatFSResult(fileSystemTypeName: "greppy-cow")
        value.blockSize = 1_048_576
        value.ioSize = 1_048_576
        return value
    }

    func activate(options: FSTaskOptions) async throws -> FSItem { root }
    func deactivate(options: FSDeactivateOptions = []) async throws {}
    func mount(options: FSTaskOptions) async throws {}
    func unmount() async {}
    func synchronize(flags: FSSyncFlags) async throws {
        let descriptor = open(heartbeat.doctorRoot.path, O_RDONLY)
        guard descriptor >= 0 else { throw posix(errno) }
        defer { close(descriptor) }
        guard fsync(descriptor) == 0 else { throw posix(errno) }
    }

    func attributes(
        _ desiredAttributes: FSItem.GetAttributesRequest,
        of rawItem: FSItem
    ) async throws -> FSItem.Attributes {
        let item = try workspace(rawItem)
        switch item.location {
        case .root:
            return syntheticAttributes(item: item, parent: .parentOfRoot)
        case .workspaces:
            return syntheticAttributes(item: item, parent: .rootDirectory)
        case .workspace:
            return syntheticAttributes(item: item, parent: workspacesItem().identifier)
        case .doctor:
            return syntheticAttributes(item: item, parent: .rootDirectory)
        case .marker:
            return markerAttributes(item: item)
        case .doctorPath(let relative):
            return try doctorAttributes(relative: relative, item: item)
        case .path(let workspace, let relative):
            let metadata = if let inode = item.boundPrivateInode() {
                try core.metadata(workspace: workspace, inode: inode)
            } else {
                try core.metadata(workspace: workspace, path: relative)
            }
            return attributes(
                metadata: metadata,
                item: item,
                parent: parentIdentifier(of: item)
            )
        }
    }

    func setAttributes(
        _ request: FSItem.SetAttributesRequest,
        on rawItem: FSItem
    ) async throws -> FSItem.Attributes {
        let item = try workspace(rawItem)
        if case .doctorPath(let relative) = item.location {
            let url = doctorURL(relative)
            var consumed: FSItem.Attribute = []
            if request.isValid(.mode), chmod(url.path, mode_t(request.mode & 0o7777)) != 0 {
                throw posix(errno)
            }
            if request.isValid(.mode) { consumed.insert(.mode) }
            if request.isValid(.size), truncate(url.path, off_t(request.size)) != 0 {
                throw posix(errno)
            }
            if request.isValid(.size) { consumed.insert(.size) }
            request.consumedAttributes = consumed
            return try doctorAttributes(relative: relative, item: item)
        }
        guard let (workspaceID, relative) = item.workspaceAndPath, !relative.isEmpty else {
            throw posix(EPERM)
        }
        var valid: UInt32 = 0
        var mode: UInt32 = 0
        var atime: Int64 = 0
        var mtime: Int64 = 0
        var consumed: FSItem.Attribute = []
        if request.isValid(.mode) {
            valid |= 1
            mode = request.mode
            consumed.insert(.mode)
        }
        if request.isValid(.accessTime) {
            valid |= 2
            atime = nanoseconds(request.accessTime)
            consumed.insert(.accessTime)
        }
        if request.isValid(.modifyTime) {
            valid |= 4
            mtime = nanoseconds(request.modifyTime)
            consumed.insert(.modifyTime)
        }
        if request.isValid(.size) {
            let (_, inode) = try privateInode(for: item)
            try core.truncate(workspace: workspaceID, inode: inode, size: request.size)
            consumed.insert(.size)
        }
        if valid != 0 {
            let (_, inode) = try privateInode(for: item)
            try core.setMetadata(
                workspace: workspaceID,
                inode: inode,
                valid: valid,
                mode: mode,
                accessedNanoseconds: atime,
                modifiedNanoseconds: mtime
            )
        }
        request.consumedAttributes = consumed
        return try await attributes(FSItem.GetAttributesRequest(), of: item)
    }

    func lookupItem(
        named name: FSFileName,
        inDirectory rawDirectory: FSItem
    ) async throws -> (FSItem, FSFileName) {
        let directory = try workspace(rawDirectory)
        guard let component = name.string else { throw posix(EINVAL) }
        switch directory.location {
        case .root where component == "workspaces":
            return (workspacesItem(), name)
        case .root where component == "doctor":
            return (doctorItem(), name)
        case .root where component == ".greppy-provider.json":
            return (markerItem(), name)
        case .workspaces:
            guard try core.workspaces().contains(component) else { throw posix(ENOENT) }
            return (item(location: .workspace(component), name: name), name)
        case .workspace(let workspaceID), .path(let workspaceID, _):
            let relative = try relativePath(parent: directory, name: name)
            _ = try core.metadata(workspace: workspaceID, path: relative)
            return (item(location: .path(workspace: workspaceID, relative: relative), name: name), name)
        case .doctor, .doctorPath:
            let relative = try relativePath(parent: directory, name: name)
            guard FileManager.default.fileExists(atPath: doctorURL(relative).path) else {
                throw posix(ENOENT)
            }
            return (item(location: .doctorPath(relative), name: name), name)
        default:
            throw posix(ENOENT)
        }
    }

    func reclaimItem(_ item: FSItem) async throws {}

    func readSymbolicLink(_ rawItem: FSItem) async throws -> FSFileName {
        let item = try workspace(rawItem)
        if case .doctorPath(let relative) = item.location {
            var bytes = [UInt8](repeating: 0, count: Int(PATH_MAX))
            let count = bytes.withUnsafeMutableBufferPointer {
                Darwin.readlink(doctorURL(relative).path, $0.baseAddress, $0.count)
            }
            guard count >= 0 else { throw posix(errno) }
            guard let target = String(bytes: bytes.prefix(count), encoding: .utf8) else {
                throw posix(EILSEQ)
            }
            return FSFileName(string: target)
        }
        guard let (workspaceID, relative) = item.workspaceAndPath, !relative.isEmpty else {
            throw posix(EINVAL)
        }
        let data = try core.readSymbolicLink(workspace: workspaceID, path: relative)
        guard let target = String(data: data, encoding: .utf8) else { throw posix(EILSEQ) }
        return FSFileName(string: target)
    }

    func createItem(
        named name: FSFileName,
        type: FSItem.ItemType,
        inDirectory rawDirectory: FSItem,
        attributes request: FSItem.SetAttributesRequest
    ) async throws -> (FSItem, FSFileName) {
        let directory = try workspace(rawDirectory)
        if case .doctor = directory.location {
            return try createDoctorItem(
                named: name,
                type: type,
                directory: directory,
                mode: request.isValid(.mode) ? request.mode : 0o700
            )
        }
        if case .doctorPath = directory.location {
            return try createDoctorItem(
                named: name,
                type: type,
                directory: directory,
                mode: request.isValid(.mode) ? request.mode : 0o700
            )
        }
        guard let (workspaceID, _) = directory.workspaceAndPath else { throw posix(EPERM) }
        let relative = try relativePath(parent: directory, name: name)
        let mode = request.isValid(.mode) ? request.mode : (type == .directory ? 0o755 : 0o644)
        switch type {
        case .file:
            try core.createFile(workspace: workspaceID, path: relative, mode: mode)
        case .directory:
            try core.createDirectory(workspace: workspaceID, path: relative, mode: mode)
        default:
            throw posix(EINVAL)
        }
        var consumed: FSItem.Attribute = []
        var valid: UInt32 = 0
        var accessedNanoseconds: Int64 = 0
        var modifiedNanoseconds: Int64 = 0
        if request.isValid(.mode) {
            valid |= 1
            consumed.insert(.mode)
        }
        if request.isValid(.accessTime) {
            valid |= 2
            accessedNanoseconds = nanoseconds(request.accessTime)
            consumed.insert(.accessTime)
        }
        if request.isValid(.modifyTime) {
            valid |= 4
            modifiedNanoseconds = nanoseconds(request.modifyTime)
            consumed.insert(.modifyTime)
        }
        if request.isValid(.size), type == .file {
            try core.truncate(workspace: workspaceID, path: relative, size: request.size)
            consumed.insert(.size)
        }
        if valid != 0 {
            try core.setMetadata(
                workspace: workspaceID,
                path: relative,
                valid: valid,
                mode: mode,
                accessedNanoseconds: accessedNanoseconds,
                modifiedNanoseconds: modifiedNanoseconds
            )
        }
        request.consumedAttributes = consumed
        return (item(location: .path(workspace: workspaceID, relative: relative), name: name), name)
    }

    func createSymbolicLink(
        named name: FSFileName,
        inDirectory rawDirectory: FSItem,
        attributes newAttributes: FSItem.SetAttributesRequest,
        linkContents contents: FSFileName
    ) async throws -> (FSItem, FSFileName) {
        let directory = try workspace(rawDirectory)
        if case .doctor = directory.location {
            return try createDoctorSymbolicLink(named: name, directory: directory, contents: contents)
        }
        if case .doctorPath = directory.location {
            return try createDoctorSymbolicLink(named: name, directory: directory, contents: contents)
        }
        guard let (workspaceID, _) = directory.workspaceAndPath,
              let target = contents.string?.data(using: .utf8) else { throw posix(EINVAL) }
        let relative = try relativePath(parent: directory, name: name)
        try core.symbolicLink(workspace: workspaceID, path: relative, target: target)
        return (item(location: .path(workspace: workspaceID, relative: relative), name: name), name)
    }

    func createLink(
        to rawItem: FSItem,
        named name: FSFileName,
        inDirectory rawDirectory: FSItem
    ) async throws -> FSFileName {
        let source = try workspace(rawItem)
        let directory = try workspace(rawDirectory)
        if case .doctorPath(let sourcePath) = source.location {
            switch directory.location {
            case .doctor, .doctorPath:
                return try createDoctorHardLink(
                    source: sourcePath,
                    named: name,
                    directory: directory
                )
            default:
                throw posix(EXDEV)
            }
        }
        guard let (sourceWorkspace, sourcePath) = source.workspaceAndPath,
              let (destinationWorkspace, _) = directory.workspaceAndPath,
              sourceWorkspace == destinationWorkspace,
              !sourcePath.isEmpty else { throw posix(EXDEV) }
        let destination = try relativePath(parent: directory, name: name)
        try core.hardLink(workspace: sourceWorkspace, source: sourcePath, destination: destination)
        return name
    }

    func removeItem(
        _ rawItem: FSItem,
        named name: FSFileName,
        fromDirectory directory: FSItem
    ) async throws {
        let item = try workspace(rawItem)
        if case .doctorPath(let relative) = item.location {
            do { try FileManager.default.removeItem(at: doctorURL(relative)) }
            catch { throw posix(EIO) }
            evict(item.location)
            return
        }
        guard let (workspaceID, relative) = item.workspaceAndPath, !relative.isEmpty else {
            throw posix(EPERM)
        }
        if try core.metadata(workspace: workspaceID, path: relative).kind == .file {
            _ = try privateInode(for: item)
        }
        try core.unlink(workspace: workspaceID, path: relative)
        evict(item.location)
    }

    func renameItem(
        _ rawItem: FSItem,
        inDirectory sourceDirectory: FSItem,
        named sourceName: FSFileName,
        to destinationName: FSFileName,
        inDirectory rawDestinationDirectory: FSItem,
        overItem: FSItem?
    ) async throws -> FSFileName {
        let source = try workspace(rawItem)
        let destinationDirectory = try workspace(rawDestinationDirectory)
        if case .doctorPath(let sourcePath) = source.location {
            switch destinationDirectory.location {
            case .doctor, .doctorPath:
                return try renameDoctor(
                    source: sourcePath,
                    directory: destinationDirectory,
                    destinationName: destinationName
                )
            default:
                throw posix(EXDEV)
            }
        }
        guard let (sourceWorkspace, sourcePath) = source.workspaceAndPath,
              let (destinationWorkspace, _) = destinationDirectory.workspaceAndPath,
              sourceWorkspace == destinationWorkspace,
              !sourcePath.isEmpty else { throw posix(EXDEV) }
        let destination = try relativePath(parent: destinationDirectory, name: destinationName)
        if try core.metadata(workspace: sourceWorkspace, path: sourcePath).kind == .file {
            _ = try privateInode(for: source)
        }
        try core.rename(workspace: sourceWorkspace, source: sourcePath, destination: destination)
        evict(source.location)
        return destinationName
    }

    func enumerateDirectory(
        _ rawDirectory: FSItem,
        startingAt cookie: FSDirectoryCookie,
        verifier: FSDirectoryVerifier,
        attributes requested: FSItem.GetAttributesRequest?,
        packer: FSDirectoryEntryPacker
    ) async throws -> FSDirectoryVerifier {
        let directory = try workspace(rawDirectory)
        let children = try directoryChildren(directory)
        let start = Int(cookie.rawValue)
        guard start <= children.count else { throw posix(EINVAL) }
        for index in start..<children.count {
            let child = children[index]
            let complete = if let metadata = child.metadata {
                attributes(metadata: metadata, item: child.item, parent: directory.identifier)
            } else {
                try await attributes(FSItem.GetAttributesRequest(), of: child.item)
            }
            let attrs = requested == nil ? nil : complete
            let type = complete.type
            let packed = packer.packEntry(
                name: child.item.name,
                itemType: type,
                itemID: child.item.identifier,
                nextCookie: FSDirectoryCookie(UInt64(index + 1)),
                attributes: attrs
            )
            if !packed { break }
        }
        return FSDirectoryVerifier(directoryGeneration(children.map(\.item)))
    }

    private func directoryChildren(_ directory: GreppyFSItem) throws -> [DirectoryChild] {
        switch directory.location {
        case .root:
            return [markerItem(), doctorItem(), workspacesItem()].map {
                DirectoryChild(item: $0, metadata: nil)
            }
        case .workspaces:
            return try core.workspaces().sorted().map {
                DirectoryChild(
                    item: item(location: .workspace($0), name: FSFileName(string: $0)),
                    metadata: nil
                )
            }
        case .workspace(let workspaceID):
            return try core.directory(workspace: workspaceID, path: "").map {
                DirectoryChild(
                    item: item(
                        location: .path(workspace: workspaceID, relative: $0.name),
                        name: FSFileName(string: $0.name)
                    ),
                    metadata: $0.metadata.workspaceMetadata
                )
            }
        case .path(let workspaceID, let relative):
            return try core.directory(workspace: workspaceID, path: relative).map {
                let child = relative.isEmpty ? $0.name : "\(relative)/\($0.name)"
                return DirectoryChild(
                    item: item(
                        location: .path(workspace: workspaceID, relative: child),
                        name: FSFileName(string: $0.name)
                    ),
                    metadata: $0.metadata.workspaceMetadata
                )
            }
        case .doctor:
            return try doctorChildren(relative: "").map {
                DirectoryChild(item: $0, metadata: nil)
            }
        case .doctorPath(let relative):
            return try doctorChildren(relative: relative).map {
                DirectoryChild(item: $0, metadata: nil)
            }
        case .marker:
            throw posix(ENOTDIR)
        }
    }

    private func directoryGeneration(_ items: [GreppyFSItem]) -> UInt64 {
        items.reduce(14_695_981_039_346_656_037) { value, item in
            item.name.description.utf8.reduce(value) { ($0 ^ UInt64($1)) &* 1_099_511_628_211 }
        }
    }

    private func workspacesItem() -> GreppyFSItem {
        item(location: .workspaces, name: FSFileName(string: "workspaces"))
    }

    private func doctorItem() -> GreppyFSItem {
        item(location: .doctor, name: FSFileName(string: "doctor"))
    }

    private func markerItem() -> GreppyFSItem {
        item(location: .marker, name: FSFileName(string: ".greppy-provider.json"))
    }

    private func parentIdentifier(of item: GreppyFSItem) -> FSItem.Identifier {
        switch item.location {
        case .root: return .parentOfRoot
        case .workspaces: return .rootDirectory
        case .doctor: return .rootDirectory
        case .marker: return .rootDirectory
        case .doctorPath(let relative):
            guard let slash = relative.lastIndex(of: "/") else { return doctorItem().identifier }
            let parent = String(relative[..<slash])
            let name = parent.split(separator: "/").last.map(String.init) ?? "doctor"
            return self.item(
                location: .doctorPath(parent),
                name: FSFileName(string: name)
            ).identifier
        case .workspace: return workspacesItem().identifier
        case .path(let workspaceID, let relative):
            guard let slash = relative.lastIndex(of: "/") else {
                return self.item(
                    location: .workspace(workspaceID),
                    name: FSFileName(string: workspaceID)
                ).identifier
            }
            let parent = String(relative[..<slash])
            let name = parent.split(separator: "/").last.map(String.init) ?? workspaceID
            return self.item(
                location: .path(workspace: workspaceID, relative: parent),
                name: FSFileName(string: name)
            ).identifier
        }
    }

    private func syntheticAttributes(
        item: GreppyFSItem,
        parent: FSItem.Identifier
    ) -> FSItem.Attributes {
        let value = FSItem.Attributes()
        value.fileID = item.identifier
        value.parentID = parent
        value.type = .directory
        value.mode = UInt32(S_IFDIR | 0o700)
        value.linkCount = 2
        value.size = 0
        value.allocSize = 0
        completeStandardAttributes(value, timestamp: mountTime)
        return value
    }

    private func markerAttributes(item: GreppyFSItem) -> FSItem.Attributes {
        let snapshot = heartbeat.manifestSnapshot()
        let value = FSItem.Attributes()
        value.fileID = item.identifier
        value.parentID = .rootDirectory
        value.type = .file
        value.mode = UInt32(S_IFREG | 0o400)
        value.linkCount = 1
        value.size = UInt64(snapshot.data.count)
        value.allocSize = value.size
        value.inhibitKernelOffloadedIO = true
        completeStandardAttributes(value, timestamp: dateTime(snapshot.modifiedAt))
        return value
    }

    private func doctorURL(_ relative: String) -> URL {
        relative.isEmpty
            ? heartbeat.doctorRoot
            : heartbeat.doctorRoot.appendingPathComponent(relative)
    }

    private func doctorAttributes(
        relative: String,
        item: GreppyFSItem
    ) throws -> FSItem.Attributes {
        let raw = try FileManager.default.attributesOfItem(atPath: doctorURL(relative).path)
        let kind = raw[.type] as? FileAttributeType
        let value = FSItem.Attributes()
        value.fileID = item.identifier
        value.parentID = parentIdentifier(of: item)
        value.type = kind == .typeDirectory ? .directory : (kind == .typeSymbolicLink ? .symlink : .file)
        let permissions = (raw[.posixPermissions] as? NSNumber)?.uint32Value ?? 0o600
        let typeMode: UInt32 = value.type == .directory
            ? UInt32(S_IFDIR)
            : (value.type == .symlink ? UInt32(S_IFLNK) : UInt32(S_IFREG))
        value.mode = typeMode | permissions
        value.linkCount = (raw[.referenceCount] as? NSNumber)?.uint32Value ?? 1
        value.size = (raw[.size] as? NSNumber)?.uint64Value ?? 0
        value.allocSize = value.size
        let modified = (raw[.modificationDate] as? Date).map(dateTime) ?? mountTime
        let created = (raw[.creationDate] as? Date).map(dateTime) ?? modified
        completeStandardAttributes(value, timestamp: modified, birthTime: created)
        return value
    }

    private func doctorChildren(relative: String) throws -> [GreppyFSItem] {
        try FileManager.default.contentsOfDirectory(atPath: doctorURL(relative).path)
            .sorted()
            .map { name in
                let child = relative.isEmpty ? name : "\(relative)/\(name)"
                return item(location: .doctorPath(child), name: FSFileName(string: name))
            }
    }

    private func createDoctorItem(
        named name: FSFileName,
        type: FSItem.ItemType,
        directory: GreppyFSItem,
        mode: UInt32
    ) throws -> (FSItem, FSFileName) {
        let relative = try relativePath(parent: directory, name: name)
        let url = doctorURL(relative)
        switch type {
        case .directory:
            guard mkdir(url.path, mode_t(mode & 0o7777)) == 0 else { throw posix(errno) }
        case .file:
            let descriptor = open(url.path, O_CREAT | O_EXCL | O_RDWR, mode_t(mode & 0o7777))
            guard descriptor >= 0 else { throw posix(errno) }
            close(descriptor)
        default:
            throw posix(EINVAL)
        }
        return (item(location: .doctorPath(relative), name: name), name)
    }

    private func createDoctorSymbolicLink(
        named name: FSFileName,
        directory: GreppyFSItem,
        contents: FSFileName
    ) throws -> (FSItem, FSFileName) {
        guard let target = contents.string else { throw posix(EINVAL) }
        let relative = try relativePath(parent: directory, name: name)
        guard Darwin.symlink(target, doctorURL(relative).path) == 0 else { throw posix(errno) }
        return (item(location: .doctorPath(relative), name: name), name)
    }

    private func createDoctorHardLink(
        source: String,
        named name: FSFileName,
        directory: GreppyFSItem
    ) throws -> FSFileName {
        let destination = try relativePath(parent: directory, name: name)
        guard Darwin.link(doctorURL(source).path, doctorURL(destination).path) == 0 else {
            throw posix(errno)
        }
        return name
    }

    private func renameDoctor(
        source: String,
        directory: GreppyFSItem,
        destinationName: FSFileName
    ) throws -> FSFileName {
        let destination = try relativePath(parent: directory, name: destinationName)
        guard Darwin.rename(doctorURL(source).path, doctorURL(destination).path) == 0 else {
            throw posix(errno)
        }
        evict(.doctorPath(source))
        return destinationName
    }

    private func attributes(
        metadata: RustWorkspaceMetadata,
        item: GreppyFSItem,
        parent: FSItem.Identifier
    ) -> FSItem.Attributes {
        let value = FSItem.Attributes()
        value.fileID = item.identifier
        value.parentID = parent
        value.type = itemType(metadata.kind)
        value.mode = metadata.mode
        value.linkCount = metadata.linkCount
        value.size = metadata.size
        value.allocSize = metadata.size
        value.accessTime = time(metadata.accessedNanoseconds)
        value.modifyTime = time(metadata.modifiedNanoseconds)
        value.changeTime = time(metadata.changedNanoseconds)
        value.birthTime = value.changeTime
        value.uid = getuid()
        value.gid = getgid()
        value.flags = 0
        return value
    }

    private func completeStandardAttributes(
        _ value: FSItem.Attributes,
        timestamp: timespec,
        birthTime: timespec? = nil
    ) {
        value.uid = getuid()
        value.gid = getgid()
        value.flags = 0
        value.birthTime = birthTime ?? timestamp
        value.accessTime = timestamp
        value.modifyTime = timestamp
        value.changeTime = timestamp
    }

    private func itemType(_ kind: RustWorkspaceMetadata.Kind) -> FSItem.ItemType {
        switch kind {
        case .file: return .file
        case .directory: return .directory
        case .symbolicLink: return .symlink
        }
    }

    private func time(_ nanoseconds: Int64) -> timespec {
        timespec(
            tv_sec: Int(nanoseconds / 1_000_000_000),
            tv_nsec: Int(nanoseconds % 1_000_000_000)
        )
    }

    private func dateTime(_ date: Date) -> timespec {
        time(Int64(date.timeIntervalSince1970 * 1_000_000_000))
    }

    private func nanoseconds(_ value: timespec) -> Int64 {
        Int64(value.tv_sec) * 1_000_000_000 + Int64(value.tv_nsec)
    }
}

extension GreppyFSVolume: FSVolume.OpenCloseOperations {
    func openItem(_ item: FSItem, modes: FSVolume.OpenModes) async throws {}
    func closeItem(_ item: FSItem, modes: FSVolume.OpenModes) async throws {}
}

extension GreppyFSVolume: FSVolume.ReadWriteOperations {
    func read(
        from rawItem: FSItem,
        at offset: off_t,
        length: Int,
        into buffer: FSMutableFileDataBuffer
    ) async throws -> Int {
        let item = try workspace(rawItem)
        guard offset >= 0 else { throw posix(EINVAL) }
        if case .marker = item.location {
            let data = heartbeat.manifestData()
            let start = min(Int(offset), data.count)
            let end = min(start + min(length, buffer.length), data.count)
            let slice = data[start..<end]
            return slice.withUnsafeBytes { source in
                buffer.withUnsafeMutableBytes { $0.copyMemory(from: source) }
                return slice.count
            }
        }
        if case .doctorPath(let relative) = item.location {
            var data = Data(count: min(length, buffer.length))
            let descriptor = open(doctorURL(relative).path, O_RDONLY)
            guard descriptor >= 0 else { throw posix(errno) }
            defer { close(descriptor) }
            let count = data.withUnsafeMutableBytes {
                pread(descriptor, $0.baseAddress, $0.count, offset)
            }
            guard count >= 0 else { throw posix(errno) }
            data.removeSubrange(count..<data.count)
            return data.withUnsafeBytes { source in
                buffer.withUnsafeMutableBytes { $0.copyMemory(from: source) }
                return data.count
            }
        }
        let (workspaceID, inode) = try readableInode(for: item)
        let data = try core.read(
            workspace: workspaceID,
            inode: inode,
            offset: UInt64(offset),
            length: min(length, buffer.length)
        )
        return data.withUnsafeBytes { source in
            buffer.withUnsafeMutableBytes { destination in
                destination.copyMemory(from: source)
            }
            return data.count
        }
    }

    func write(contents: Data, to rawItem: FSItem, at offset: off_t) async throws -> Int {
        let item = try workspace(rawItem)
        guard offset >= 0 else { throw posix(EINVAL) }
        if case .doctorPath(let relative) = item.location {
            let descriptor = open(doctorURL(relative).path, O_WRONLY)
            guard descriptor >= 0 else { throw posix(errno) }
            defer { close(descriptor) }
            let count = contents.withUnsafeBytes {
                pwrite(descriptor, $0.baseAddress, $0.count, offset)
            }
            guard count >= 0 else { throw posix(errno) }
            return count
        }
        let (workspaceID, inode) = try privateInode(for: item)
        return try core.write(
            workspace: workspaceID,
            inode: inode,
            offset: UInt64(offset),
            contents: contents
        )
    }
}
