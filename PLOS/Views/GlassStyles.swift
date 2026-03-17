import SwiftUI

// MARK: - Views

struct GlassTheme {
    static let bgStart = Color(red: 0.05, green: 0.10, blue: 0.16)
    static let bgMid = Color(red: 0.07, green: 0.20, blue: 0.22)
    static let bgEnd = Color(red: 0.12, green: 0.24, blue: 0.28)
    static let cardStroke = Color.white.opacity(0.26)
    static let cardShadow = Color.black.opacity(0.24)
    static let userTint = Color.blue.opacity(0.20)
    static let localTint = Color.green.opacity(0.18)
    static let externalTint = Color.orange.opacity(0.20)
}

struct GlassCardModifier: ViewModifier {
    var cornerRadius: CGFloat = 16

    func body(content: Content) -> some View {
        content
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(GlassTheme.cardStroke, lineWidth: 1)
            )
            .shadow(color: GlassTheme.cardShadow, radius: 18, x: 0, y: 10)
    }
}

extension View {
    func glassCard(cornerRadius: CGFloat = 16) -> some View {
        modifier(GlassCardModifier(cornerRadius: cornerRadius))
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
                .fill(Color.white.opacity(0.16))
                .blur(radius: 90)
                .frame(width: animate ? 420 : 300, height: animate ? 420 : 300)
                .offset(x: animate ? 180 : -120, y: animate ? -140 : 160)

            Circle()
                .fill(Color.cyan.opacity(0.15))
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

