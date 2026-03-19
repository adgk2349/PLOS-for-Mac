# 변경점 (Korean)

## v0.2.1 (2026-03-19)

### 주요 변경
- `/v2/chat/local` 중심의 대화형 응답 파이프라인 강화
  - Follow-up 해석, 후보 우선 응답, 확인 질문 과다 억제 정책 반영
  - 응답 구조를 `lead + result + actions + metadata` 형태로 일관화
- 로컬 메모리 계층(Session / Workspace / Preference / Episodic / Pinned) 통합
  - 질문/액션/파일 선택 기반 메모리 쓰기 훅 추가
  - 대화/검색/응답 단계별 관련 메모리 선택 주입
- 로컬/외부 라우팅 안정화
  - Privacy gate(`LOCAL_ONLY`, `HYBRID`, `CONFIRM`) 분기 보강
  - 일반 대화 시 로컬 LLM 경로 우선 사용 및 실패 처리 개선

### UI/UX
- macOS 메인 워크스페이스 레이아웃 정리
  - 헤더/사이드바/채팅 영역 간 경계선/겹침 이슈 완화
  - 캡슐/라운드 컴포넌트 정렬 일관성 개선
- 설정/상태/온보딩 화면의 스타일 및 플로우 정비

### 인덱싱/검색
- 문서 메타데이터(카테고리/태그/서브카테고리/중요도) 처리 흐름 강화
- 워크스페이스 경계 기반 검색 필터링 보강
- PDF/OCR 처리 안정성 및 실패 기록 경로 개선

### 안정성/개발
- sidecar 부트스트랩 및 포트 점유 충돌 대응 강화
- 세션 토큰/인증 처리 로직 보완
- 테스트 및 모듈 분리(파이프라인/메모리/검증 컴포넌트) 확장

---

## 참고
- 메인 문서: [README.ko.md](README.ko.md)
- 영문: [CHANGELOG.en.md](CHANGELOG.en.md)
- 일본어: [CHANGELOG.ja.md](CHANGELOG.ja.md)
