# Local AI Core for Mac (MVP v0.1)

SwiftUI(macOS) + Python FastAPI sidecar 기반의 로컬 우선 AI 코어 구현체입니다.

## 구현 범위
- macOS 앱 셸 + 6단계 온보딩
- 폴더 선택 + 보안 북마크 저장
- 로컬 인덱싱(txt/md/pdf + OCR fallback)
- 청크화 + 임베딩 + LanceDB 저장
- 로컬 Q&A + 출처(citation) 표시
- 수동 외부 심화 분석(OpenAI/Anthropic)
- 프라이버시 모드 게이트(LOCAL_ONLY/HYBRID/CONFIRM)
- 상태 패널/설정/실패 파일 목록
- 증분 인덱싱(파일 변경 감지 폴링 watcher)

## 프로젝트 구조
- `Sources/LocalAICoreApp`: SwiftUI macOS 앱
- `sidecar/local_ai_core`: FastAPI sidecar
- `sidecar/tests`: 단위/통합 테스트

## 실행 방법
### 1) Sidecar 테스트
```bash
cd sidecar
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

### 2) macOS 앱 빌드
```bash
swift build
swift run LocalAICoreApp
```

앱이 시작되면 내부에서 sidecar를 자동 실행합니다.
환경변수 예시는 [`.env.example`](/Users/seungminlee/Desktop/Development/PLOS/sidecar/.env.example) 참고.

## Sidecar API
- `POST /v1/workspaces`
- `POST /v1/index/jobs`
- `GET /v1/index/jobs/{job_id}`
- `GET /v1/index/failures`
- `POST /v1/chat/local`
- `POST /v1/chat/deep-analysis`
- `GET /v1/settings`
- `PUT /v1/settings`
- `GET /v1/status`

모든 `/v1/*` 요청은 `x-session-token` 헤더가 필요합니다.

## 핵심 타입
- `PrivacyMode`: `LOCAL_ONLY | HYBRID | CONFIRM_BEFORE_EXTERNAL`
- `WorkMode`: `GENERAL | SUMMARY | RESEARCH | DEVELOPMENT | WRITING | PLANNING | STRICT_SEARCH`
- `Citation`: `doc_id, chunk_id, file_path, snippet, score, modified_at`
- `ExternalCallEvent`: `provider, sent_chars, approved_by_user, timestamp`

## 비고
- `STRICT_SEARCH`는 근거 신뢰도가 낮으면 “근거 부족” 응답을 강제합니다.
- 외부 호출은 자동 라우팅 없이 사용자 수동 액션에서만 실행됩니다.
- OCR fallback은 `pdf2image + pytesseract` 의존성을 사용합니다.
