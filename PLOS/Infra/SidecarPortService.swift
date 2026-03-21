import Darwin
import Foundation

enum SidecarPortService {
    static func portCandidates(preferred: Int) -> [Int] {
        var ports = [preferred]
        let fallback = [8777, 8787, 8797, 8807, 8817, 8827]
        for port in fallback where !ports.contains(port) {
            ports.append(port)
        }
        for _ in 0 ..< 12 {
            if let dynamic = findAvailablePort(), !ports.contains(dynamic) {
                ports.append(dynamic)
            }
        }
        for port in stride(from: 18080, through: 18140, by: 2) where !ports.contains(port) {
            ports.append(port)
        }
        return ports
    }

    static func findAvailablePort() -> Int? {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            return nil
        }
        defer { _ = close(socketFD) }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(0).bigEndian
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            return nil
        }

        var assigned = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &assigned) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socketFD, $0, &len)
            }
        }
        guard nameResult == 0 else {
            return nil
        }
        return Int(UInt16(bigEndian: assigned.sin_port))
    }

    static func isPortListening(_ port: Int) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["lsof", "-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                return false
            }
            let raw = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            return !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        } catch {
            return false
        }
    }

    static func terminateStaleRuntimeSidecars(runtimeDirectory: URL, ports: [Int]) {
        var visited = Set<Int32>()
        let runtimePath = runtimeDirectory.standardizedFileURL.path

        for port in ports {
            let pids = listeningPIDs(on: port)
            for pid in pids where visited.insert(pid).inserted {
                let command = processCommandLine(pid: pid).lowercased()
                guard command.contains("uvicorn local_ai_core.main:app") else {
                    continue
                }
                guard let cwd = processCurrentDirectory(pid: pid) else {
                    continue
                }
                if URL(fileURLWithPath: cwd).standardizedFileURL.path != runtimePath {
                    continue
                }
                _ = kill(pid, SIGTERM)
            }
        }

        usleep(200_000)
    }

    static func listeningPIDs(on port: Int) -> [Int32] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["lsof", "-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return []
        }
        guard process.terminationStatus == 0 else {
            return []
        }
        let raw = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return raw
            .split(separator: "\n")
            .compactMap { Int32(String($0).trimmingCharacters(in: .whitespacesAndNewlines)) }
    }

    static func processCommandLine(pid: Int32) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-p", String(pid), "-o", "command="]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return ""
        }
        guard process.terminationStatus == 0 else {
            return ""
        }
        return (String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func processCurrentDirectory(pid: Int32) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["lsof", "-a", "-p", String(pid), "-d", "cwd", "-Fn"]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }
        guard process.terminationStatus == 0 else {
            return nil
        }
        let raw = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        for line in raw.split(separator: "\n") {
            if line.hasPrefix("n") {
                return String(line.dropFirst())
            }
        }
        return nil
    }


}
