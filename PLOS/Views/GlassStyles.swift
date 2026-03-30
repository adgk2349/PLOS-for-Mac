import SwiftUI

struct PLOSGlassTheme {
    static let strokeStrong = Color.primary.opacity(0.24)
    static let strokeSoft = Color.primary.opacity(0.12)

    static func chromeTint(for scheme: ColorScheme) -> Color {
        if scheme == .dark {
            return Color.black.opacity(0.24)
        }
        return Color.black.opacity(0.06)
    }

    static func userBubbleTint(for scheme: ColorScheme) -> Color {
        if scheme == .dark {
            return Color(red: 0.19, green: 0.21, blue: 0.26).opacity(0.78)
        }
        return Color(red: 0.80, green: 0.82, blue: 0.86).opacity(0.84)
    }

    static func userBubbleStroke(for scheme: ColorScheme) -> Color {
        if scheme == .dark {
            return Color.white.opacity(0.18)
        }
        return Color.black.opacity(0.20)
    }

    static func titlebarBase(for scheme: ColorScheme) -> Color {
#if os(macOS)
        return Color(nsColor: .windowBackgroundColor)
#else
        if scheme == .dark {
            return Color.black.opacity(0.88)
        }
        return Color.white.opacity(0.94)
#endif
    }
}

struct PLOSGlassBackground: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        let tint = colorScheme == .dark
            ? Color.black.opacity(0.22)
            : Color.white.opacity(0.24)

        Rectangle()
            .fill(.clear)
            .glassEffect(.regular.tint(tint), in: Rectangle())
        .ignoresSafeArea()
    }
}

private struct PLOSRoundedGlassSurface: ViewModifier {
    var radius: CGFloat = 14
    var stroke: Color = PLOSGlassTheme.strokeSoft
    var lineWidth: CGFloat = 1
    var tint: Color? = nil

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        return content
            .background {
                shape
                    .fill(.clear)
                    .glassEffect(
                        tint.map { .regular.tint($0) } ?? .regular,
                        in: shape
                    )
            }
            .overlay {
                shape
                    .stroke(stroke, lineWidth: lineWidth)
            }
            .clipShape(shape)
    }
}

private struct PLOSCapsuleGlassSurface: ViewModifier {
    var stroke: Color = PLOSGlassTheme.strokeSoft
    var lineWidth: CGFloat = 1
    var tint: Color? = nil

    func body(content: Content) -> some View {
        let shape = Capsule(style: .continuous)
        return content
            .background {
                shape
                    .fill(.clear)
                    .glassEffect(
                        tint.map { .regular.tint($0) } ?? .regular,
                        in: shape
                    )
            }
            .overlay {
                shape
                    .stroke(stroke, lineWidth: lineWidth)
            }
            .clipShape(shape)
    }
}

private struct PLOSCircleGlassSurface: ViewModifier {
    var stroke: Color = PLOSGlassTheme.strokeSoft
    var lineWidth: CGFloat = 1
    var tint: Color? = nil

    func body(content: Content) -> some View {
        let shape = Circle()
        return content
            .background {
                shape
                    .fill(.clear)
                    .glassEffect(
                        tint.map { .regular.tint($0) } ?? .regular,
                        in: shape
                    )
            }
            .overlay {
                shape
                    .stroke(stroke, lineWidth: lineWidth)
            }
            .clipShape(shape)
    }
}

extension View {
    func plosGlassPanel(radius: CGFloat = 16) -> some View {
        modifier(PLOSRoundedGlassSurface(radius: radius, stroke: PLOSGlassTheme.strokeSoft, lineWidth: 1, tint: nil))
    }

    func plosGlassControl(radius: CGFloat = 12) -> some View {
        modifier(PLOSRoundedGlassSurface(radius: radius, stroke: PLOSGlassTheme.strokeStrong, lineWidth: 1, tint: Color.primary.opacity(0.06)))
    }

    func plosGlassInputFrame(radius: CGFloat = 12) -> some View {
        modifier(PLOSRoundedGlassSurface(radius: radius, stroke: PLOSGlassTheme.strokeSoft, lineWidth: 1, tint: Color.primary.opacity(0.05)))
    }

    func plosGlassChip(radius: CGFloat = 12) -> some View {
        modifier(PLOSRoundedGlassSurface(radius: radius, stroke: PLOSGlassTheme.strokeSoft, lineWidth: 1, tint: Color.primary.opacity(0.07)))
    }

    func plosGlassCapsule(tint: Color? = Color.primary.opacity(0.06)) -> some View {
        modifier(PLOSCapsuleGlassSurface(stroke: PLOSGlassTheme.strokeStrong, lineWidth: 1, tint: tint))
    }

    func plosGlassHeaderCapsule() -> some View {
        modifier(PLOSCapsuleGlassSurface(stroke: Color.primary.opacity(0.28), lineWidth: 1, tint: Color.primary.opacity(0.12)))
            .shadow(color: .black.opacity(0.20), radius: 8, y: 3)
    }

    func plosGlassCircle(tint: Color? = Color.primary.opacity(0.06)) -> some View {
        modifier(PLOSCircleGlassSurface(stroke: PLOSGlassTheme.strokeStrong, lineWidth: 1, tint: tint))
            .contentShape(Circle())
    }
}
