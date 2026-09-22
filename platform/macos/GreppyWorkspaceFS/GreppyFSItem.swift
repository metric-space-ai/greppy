import Foundation
import FSKit

final class GreppyFSItem: FSItem {
    enum Location: Hashable {
        case root
        case workspaces
        case doctor
        case doctorPath(String)
        case marker
        case workspace(String)
        case path(workspace: String, relative: String)
    }

    let identifier: FSItem.Identifier
    private let stateLock = NSLock()
    private var currentLocation: Location
    private var currentName: FSFileName
    private var privateInode: UInt64?

    init(location: Location, name: FSFileName, identifier: FSItem.Identifier) {
        currentLocation = location
        currentName = name
        self.identifier = identifier
        super.init()
    }

    /// The kernel keeps using the same item object after a rename, so the
    /// path the item answers for has to follow the rename (see
    /// `GreppyFSVolume.relocate`).
    var location: Location {
        stateLock.withLock { currentLocation }
    }

    var name: FSFileName {
        stateLock.withLock { currentName }
    }

    func relocate(to location: Location, name: FSFileName) {
        stateLock.withLock {
            currentLocation = location
            currentName = name
        }
    }

    var workspaceAndPath: (String, String)? {
        switch location {
        case .workspace(let workspace):
            return (workspace, "")
        case .path(let workspace, let relative):
            return (workspace, relative)
        case .root, .workspaces, .doctor, .doctorPath, .marker:
            return nil
        }
    }

    func boundPrivateInode() -> UInt64? {
        stateLock.withLock { privateInode }
    }

    @discardableResult
    func bindPrivateInode(_ inode: UInt64) -> UInt64 {
        stateLock.withLock {
            if let privateInode { return privateInode }
            privateInode = inode
            return inode
        }
    }
}
