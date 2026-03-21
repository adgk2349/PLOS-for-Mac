import Foundation

enum SidecarLaunchService {
    static func makeLogFile(prefix: String) throws -> (URL, FileHandle) {
        let fm = FileManager.default
        let url = fm.temporaryDirectory.appendingPathComponent("\(prefix)-\(UUID().uuidString).log")
        fm.createFile(atPath: url.path, contents: nil)
        return (url, try FileHandle(forWritingTo: url))
    }


}
