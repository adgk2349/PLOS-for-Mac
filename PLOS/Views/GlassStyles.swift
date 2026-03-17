import SwiftUI

// MARK: - Views

struct GlassTheme {
    static let bgStart = Color(red: 0.06, green: 0.07, blue: 0.10)
    static let bgMid = Color(red: 0.08, green: 0.10, blue: 0.14)
    static let bgEnd = Color(red: 0.10, green: 0.13, blue: 0.18)
    static let userTint = Color.blue.opacity(0.24)
    static let localTint = Color.white.opacity(0.18)
    static let externalTint = Color.orange.opacity(0.24)
}

struct GlassCardModifier: ViewModifier {
    var cornerRadius: CGFloat = 16

    func body(content: Content) -> some View {
        content
            .glassEffect(
                .regular,
                in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            )
    }
}

extension View {
    func glassCard(cornerRadius: CGFloat = 16) -> some View {
        modifier(GlassCardModifier(cornerRadius: cornerRadius))
    }

    func glassTint(_ color: Color, cornerRadius: CGFloat = 12) -> some View {
        glassEffect(
            .regular.tint(color),
            in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        )
    }
}

struct AnimatedGlassBackground: View {
    @State private var animate = false

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [GlassTheme.bgStart, GlassTheme.bgMid, GlassTheme.bgEnd],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(Color.white.opacity(0.10))
                .blur(radius: 90)
                .frame(width: animate ? 420 : 300, height: animate ? 420 : 300)
                .offset(x: animate ? 180 : -120, y: animate ? -140 : 160)

            Circle()
                .fill(Color.blue.opacity(0.10))
                .blur(radius: 110)
                .frame(width: animate ? 520 : 360, height: animate ? 520 : 360)
                .offset(x: animate ? -220 : 160, y: animate ? 180 : -140)
        }
        .ignoresSafeArea()
        .onAppear {
            withAnimation(.easeInOut(duration: 8).repeatForever(autoreverses: true)) {
                animate = true
            }
        }
    }
}
