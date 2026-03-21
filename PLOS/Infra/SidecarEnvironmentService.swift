import Foundation

enum SidecarEnvironmentService {
    static func normalizedRuntimePath(existing: String?) -> String {
        var segments = (existing ?? "")
            .split(separator: ":")
            .map(String.init)
            .filter { segment in
                !segment.contains("/LocalAICore/SidecarRuntime/.venv/bin")
            }

        for required in ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"] {
            if !segments.contains(required) {
                segments.append(required)
            }
        }
        return segments.joined(separator: ":")
    }

    static func detectPopplerDirectory() -> String? {
        let fm = FileManager.default
        let candidates = ["/opt/homebrew/bin", "/usr/local/bin"]
        for dir in candidates {
            let pdftoppm = "\(dir)/pdftoppm"
            let pdfinfo = "\(dir)/pdfinfo"
            if fm.isExecutableFile(atPath: pdftoppm), fm.isExecutableFile(atPath: pdfinfo) {
                return dir
            }
            if fm.fileExists(atPath: pdftoppm), fm.fileExists(atPath: pdfinfo) {
                return dir
            }
        }
        for dir in candidates where fm.fileExists(atPath: dir) {
            return dir
        }
        return nil
    }

    static func detectTesseractExecutable() -> String? {
        let fm = FileManager.default
        let candidates = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
        for path in candidates where fm.isExecutableFile(atPath: path) {
            return path
        }
        for path in candidates where fm.fileExists(atPath: path) {
            return path
        }
        return nil
    }

    static func detectTessdataPrefix() -> String? {
        let fm = FileManager.default
        let candidates = [
            "/opt/homebrew/share",
            "/usr/local/share",
            "/opt/homebrew/share/tessdata",
            "/usr/local/share/tessdata",
        ]
        for path in candidates where fm.fileExists(atPath: path) {
            if path.hasSuffix("/tessdata") {
                return String(path.dropLast("/tessdata".count))
            }
            return path
        }
        return nil
    }


}
