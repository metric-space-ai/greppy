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

    let location: Location
    let name: FSFileName
    let identifier: FSItem.Identifier

    init(location: Location, name: FSFileName, identifier: FSItem.Identifier) {
        self.location = location
        self.name = name
        self.identifier = identifier
        super.init()
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
}
