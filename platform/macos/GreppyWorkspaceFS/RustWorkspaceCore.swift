import Foundation
import GreppyWorkspaceCore

enum RustWorkspaceError: Error, CustomStringConvertible {
    case operation(String)

    var description: String {
        switch self {
        case .operation(let message): message
        }
    }
}

struct RustWorkspaceMetadata {
    enum Kind: UInt8 {
        case file = 1
        case directory = 2
        case symbolicLink = 3
    }

    let kind: Kind
    let mode: UInt32
    let size: UInt64
    let inode: UInt64
    let linkCount: UInt32
    let accessedNanoseconds: Int64
    let modifiedNanoseconds: Int64
    let changedNanoseconds: Int64
}

struct RustWorkspaceDirectoryEntry: Decodable {
    let name: String
    let metadata: RustWorkspaceDirectoryMetadata
}

private struct RustWorkspaceStatus: Decodable {
    let id: String
}

struct RustWorkspaceDirectoryMetadata: Decodable {
    enum Kind: String, Decodable {
        case file = "File"
        case directory = "Directory"
        case symbolicLink = "Symlink"
    }

    let kind: Kind
    let mode: UInt32
    let size: UInt64
    let inode: UInt64
    let nlink: UInt32
    let accessedUnixNanoseconds: Int64
    let modifiedUnixNanoseconds: Int64
    let changedUnixNanoseconds: Int64

    enum CodingKeys: String, CodingKey {
        case kind, mode, size, inode, nlink
        case accessedUnixNanoseconds = "accessed_unix_ns"
        case modifiedUnixNanoseconds = "modified_unix_ns"
        case changedUnixNanoseconds = "changed_unix_ns"
    }
}

final class RustWorkspaceCore {
    private let raw: OpaquePointer

    init(dataRoot: URL) throws {
        guard dataRoot.path.hasPrefix("/") else {
            throw RustWorkspaceError.operation("workspace data root must be absolute")
        }
        let opened = dataRoot.path.withCString { greppy_workspace_core_open($0) }
        guard let opened else { throw Self.lastError() }
        raw = opened
    }

    deinit {
        greppy_workspace_core_close(raw)
    }

    func metadata(workspace: String, path: String) throws -> RustWorkspaceMetadata {
        var value = GreppyWorkspaceMetadata()
        let result = workspace.withCString { workspacePointer in
            path.withCString { pathPointer in
                greppy_workspace_metadata(raw, workspacePointer, pathPointer, &value)
            }
        }
        guard result == 0, let kind = RustWorkspaceMetadata.Kind(rawValue: value.kind) else {
            throw Self.lastError()
        }
        return RustWorkspaceMetadata(
            kind: kind,
            mode: value.mode,
            size: value.size,
            inode: value.inode,
            linkCount: value.nlink,
            accessedNanoseconds: value.accessed_unix_ns,
            modifiedNanoseconds: value.modified_unix_ns,
            changedNanoseconds: value.changed_unix_ns
        )
    }

    func read(workspace: String, path: String, offset: UInt64, length: Int) throws -> Data {
        var bytes = [UInt8](repeating: 0, count: length)
        let count = workspace.withCString { workspacePointer in
            path.withCString { pathPointer in
                bytes.withUnsafeMutableBufferPointer { buffer in
                    greppy_workspace_read(
                        raw, workspacePointer, pathPointer, offset, buffer.baseAddress, buffer.count
                    )
                }
            }
        }
        guard count >= 0 else { throw Self.lastError() }
        return Data(bytes.prefix(Int(count)))
    }

    @discardableResult
    func write(workspace: String, path: String, offset: UInt64, contents: Data) throws -> Int {
        let count = workspace.withCString { workspacePointer in
            path.withCString { pathPointer in
                contents.withUnsafeBytes { buffer in
                    greppy_workspace_write(
                        raw,
                        workspacePointer,
                        pathPointer,
                        offset,
                        buffer.bindMemory(to: UInt8.self).baseAddress,
                        buffer.count
                    )
                }
            }
        }
        guard count >= 0 else { throw Self.lastError() }
        return Int(count)
    }

    func directoryJSON(workspace: String, path: String) throws -> Data {
        let value = workspace.withCString { workspacePointer in
            path.withCString { pathPointer in
                greppy_workspace_list_json(raw, workspacePointer, pathPointer)
            }
        }
        guard let value else { throw Self.lastError() }
        defer { greppy_workspace_string_free(value) }
        return Data(String(cString: value).utf8)
    }

    func directory(workspace: String, path: String) throws -> [RustWorkspaceDirectoryEntry] {
        try JSONDecoder().decode(
            [RustWorkspaceDirectoryEntry].self,
            from: directoryJSON(workspace: workspace, path: path)
        )
    }

    func readSymbolicLink(workspace: String, path: String) throws -> Data {
        var capacity = 256
        while capacity <= 1_048_576 {
            var bytes = [UInt8](repeating: 0, count: capacity)
            let count = workspace.withCString { workspacePointer in
                path.withCString { pathPointer in
                    bytes.withUnsafeMutableBufferPointer { buffer in
                        greppy_workspace_read_symlink(
                            raw, workspacePointer, pathPointer, buffer.baseAddress, buffer.count
                        )
                    }
                }
            }
            guard count >= 0 else { throw Self.lastError() }
            if count <= capacity { return Data(bytes.prefix(Int(count))) }
            capacity = Int(count)
        }
        throw RustWorkspaceError.operation("symbolic-link target exceeds 1 MiB")
    }

    func truncate(workspace: String, path: String, size: UInt64) throws {
        try check(workspace: workspace, path: path) { workspacePointer, pathPointer in
            greppy_workspace_truncate(raw, workspacePointer, pathPointer, size)
        }
    }

    func setMetadata(
        workspace: String,
        path: String,
        valid: UInt32,
        mode: UInt32,
        accessedNanoseconds: Int64,
        modifiedNanoseconds: Int64
    ) throws {
        try check(workspace: workspace, path: path) { workspacePointer, pathPointer in
            greppy_workspace_set_metadata(
                raw,
                workspacePointer,
                pathPointer,
                valid,
                mode,
                accessedNanoseconds,
                modifiedNanoseconds
            )
        }
    }

    func createFile(workspace: String, path: String, mode: UInt32) throws {
        try check(workspace: workspace, path: path) { workspacePointer, pathPointer in
            greppy_workspace_create_file(raw, workspacePointer, pathPointer, mode)
        }
    }

    func createDirectory(workspace: String, path: String, mode: UInt32) throws {
        try check(workspace: workspace, path: path) { workspacePointer, pathPointer in
            greppy_workspace_mkdir(raw, workspacePointer, pathPointer, mode)
        }
    }

    func unlink(workspace: String, path: String) throws {
        try check(workspace: workspace, path: path) { workspacePointer, pathPointer in
            greppy_workspace_unlink(raw, workspacePointer, pathPointer)
        }
    }

    func rename(workspace: String, source: String, destination: String) throws {
        let result = workspace.withCString { workspacePointer in
            source.withCString { sourcePointer in
                destination.withCString { destinationPointer in
                    greppy_workspace_rename(raw, workspacePointer, sourcePointer, destinationPointer)
                }
            }
        }
        guard result == 0 else { throw Self.lastError() }
    }

    func hardLink(workspace: String, source: String, destination: String) throws {
        let result = workspace.withCString { workspacePointer in
            source.withCString { sourcePointer in
                destination.withCString { destinationPointer in
                    greppy_workspace_hard_link(raw, workspacePointer, sourcePointer, destinationPointer)
                }
            }
        }
        guard result == 0 else { throw Self.lastError() }
    }

    func symbolicLink(workspace: String, path: String, target: Data) throws {
        let result = workspace.withCString { workspacePointer in
            path.withCString { pathPointer in
                target.withUnsafeBytes { bytes in
                    greppy_workspace_symlink(
                        raw,
                        workspacePointer,
                        pathPointer,
                        bytes.bindMemory(to: UInt8.self).baseAddress,
                        bytes.count
                    )
                }
            }
        }
        guard result == 0 else { throw Self.lastError() }
    }

    func workspacesJSON() throws -> Data {
        guard let value = greppy_workspace_list_workspaces_json(raw) else {
            throw Self.lastError()
        }
        defer { greppy_workspace_string_free(value) }
        return Data(String(cString: value).utf8)
    }

    func workspaces() throws -> [String] {
        try JSONDecoder().decode([RustWorkspaceStatus].self, from: workspacesJSON()).map(\.id)
    }

    private func check(
        workspace: String,
        path: String,
        operation: (UnsafePointer<CChar>, UnsafePointer<CChar>) -> Int32
    ) throws {
        let result = workspace.withCString { workspacePointer in
            path.withCString { pathPointer in operation(workspacePointer, pathPointer) }
        }
        guard result == 0 else { throw Self.lastError() }
    }

    private static func lastError() -> RustWorkspaceError {
        guard let value = greppy_workspace_last_error() else {
            return .operation("portable workspace core failed without diagnostic")
        }
        defer { greppy_workspace_string_free(value) }
        return .operation(String(cString: value))
    }
}
