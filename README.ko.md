# PLOS for Mac (한국어)

PLOS는 macOS에서 동작하는 로컬 우선 AI 워크스페이스 코어입니다.  
기본 응답/인덱싱은 로컬에서 처리하고, 필요할 때만 외부 AI를 호출합니다.

## 핵심 기능
- 로컬 파일 인덱싱(txt/md/pdf + OCR fallback)
- 로컬 RAG 질의응답 + 출처 표시
- 작업 모드(일반/요약/연구/개발/글쓰기/기획/엄격검색)
- 대화형 응답 레이어(v2)
- 메모리 계층(Session/Workspace/Preference/Episodic/Pinned)
- 외부 AI 수동/정책 기반 호출(Privacy gate)

## UI/UX 방향
- Apple glassEffect 기반 UI
- 캡슐/라운드 컴포넌트 통일
- 라이트/다크에서 가독성 있는 중성 톤
- 채팅 중심 레이아웃 + 액션 칩

## 프로젝트 구조
- `PLOS/`: macOS SwiftUI 앱
- `sidecar/local_ai_core/`: Python FastAPI sidecar
- `sidecar/tests/`: sidecar 테스트

## 실행
1. Xcode에서 `PLOS.xcodeproj` 열기
2. 앱 실행 (sidecar 자동 부트스트랩)

## 문서
- 변경점: [CHANGELOG.ko.md](CHANGELOG.ko.md)
- English: [README.en.md](README.en.md)
- 日本語: [README.ja.md](README.ja.md)
