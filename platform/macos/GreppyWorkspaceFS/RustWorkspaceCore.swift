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

    func workspacesJSON() throws -> Data {
        guard let value = greppy_workspace_list_workspaces_json(raw) else {
            throw Self.lastError()
        }
        defer { greppy_workspace_string_free(value) }
        return Data(String(cString: value).utf8)
    }

    private static func lastError() -> RustWorkspaceError {
        guard let value = greppy_workspace_last_error() else {
            return .operation("portable workspace core failed without diagnostic")
        }
        defer { greppy_workspace_string_free(value) }
        return .operation(String(cString: value))
    }
}
