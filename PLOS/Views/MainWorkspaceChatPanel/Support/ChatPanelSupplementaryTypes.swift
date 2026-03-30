import Foundation
import SwiftUI

struct TypingDotsView: View {
    var body: some View {
        TimelineView(.animation(minimumInterval: 0.16, paused: false)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            HStack(spacing: 5) {
                ForEach(0 ..< 3, id: \.self) { index in
                    Circle()
                        .fill(Color.secondary.opacity(0.9))
                        .frame(width: 6, height: 6)
                        .opacity(dotOpacity(time: t, index: index))
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .plosGlassChip()
        }
    }

    private func dotOpacity(time: TimeInterval, index: Int) -> Double {
        let phase = (time * 3.1) + (Double(index) * 0.45)
        let wave = (sin(phase) + 1) * 0.5
        return 0.25 + (wave * 0.75)
    }
}

struct ThinkingTraceEvent: Identifiable {
    let id: String
    let status: String
    let message: String
    let source: String
    let url: String?
    let at: String?
}

struct MarkdownRenderSegment: Identifiable {
    enum Kind {
        case text
        case code
    }

    let id: String
    let kind: Kind
    let content: String
    let language: String?
}
