import FSKit

@main
struct GreppyWorkspaceFSExtension: UnaryFileSystemExtension {
    var fileSystem: FSUnaryFileSystem & FSUnaryFileSystemOperations {
        GreppyFileSystem()
    }
}
