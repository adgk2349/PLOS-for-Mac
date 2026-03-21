# PLOS for Mac (한국어)

로컬 우선 AI 워크스페이스 (macOS).

[English](README.md) | 한국어 | [日本語](README.ja.md)

## 개요
PLOS는 SwiftUI 데스크톱 앱과 Python FastAPI 사이드카를 결합해, 로컬 중심 AI 대화/검색/요약 워크플로우를 제공합니다.

- 로컬 우선 대화 + RAG(출처 표시)
- 정책 기반 외부 제공자 호출(선택/허용 시)
- 메모리 계층(Session/Workspace/Preference/Pinned)
- 하드웨어 등급 기반 모델 카탈로그
- 일반 대화 Direct-First 응답 정책

## 저장소 구조
- `PLOS/`: SwiftUI macOS 앱
- `sidecar/local_ai_core/`: FastAPI 사이드카
- `sidecar/tests/`: 사이드카 테스트
- `PLOSTests/`, `PLOSUITests/`: Swift 테스트 타깃

## 요구 사항
- Apple Silicon Mac 권장 (M 시리즈)
- macOS 14+
- Xcode 15+
- Python 3.11+
- OCR 선택 의존성: `tesseract`, `poppler`

## 설치
### 1) 클론
```bash
git clone https://github.com/adgk2349/PLOS-for-Mac.git
cd PLOS-for-Mac
```

### 2) 사이드카 환경 준비
```bash
cd sidecar
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[test]'
```

### 3) OCR 도구(선택)
```bash
brew install tesseract poppler
```

### 4) 앱 실행
- Xcode에서 `PLOS.xcodeproj` 열기
- `PLOS` 타깃 실행
- 앱 시작/종료 시 사이드카 라이프사이클 자동 연동

## 사이드카 단독 실행(개발용)
```bash
cd sidecar
source .venv/bin/activate
export LOCAL_AI_SESSION_TOKEN=dev-token
export LOCAL_AI_DATA_DIR="$(pwd)/data"
uvicorn local_ai_core.main:create_app --factory --host 127.0.0.1 --port 8787
```

## 모델 권장 구간
현재 카탈로그 기준 현실적 구간:
- 16GB: 7B/8B 중심, 12B~14B 상한 시도 가능
- 64GB+: 20B/70B급
- 256GB+: GPT-OSS 120B
- 500GB+: Kimi 2.5 / Qwen 3.5 397B급

## 메모리 구조
- 채팅방별 Session memory는 분리 저장
- Workspace memory는 프로젝트 단위 맥락 저장
- Preference/Pinned memory는 사용자 명시 정보만 유지

## 테스트
### 사이드카
```bash
cd sidecar
source .venv/bin/activate
pytest -q
```

### 핵심 회귀 세트
```bash
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

### Swift 테스트
```bash
xcodebuild \
  -project PLOS.xcodeproj \
  -scheme PLOS \
  -destination 'platform=macOS' \
  test
```

## 성능 테스트
반복 가능한 측정 시나리오는 [PERFORMANCE.ko.md](PERFORMANCE.ko.md) 참고.

## 기여 방법
[CONTRIBUTING.ko.md](CONTRIBUTING.ko.md) 또는 [CONTRIBUTING.md](CONTRIBUTING.md) 참고.

## 변경 이력
- [CHANGELOG.ko.md](CHANGELOG.ko.md)
- [CHANGELOG.en.md](CHANGELOG.en.md)
- [CHANGELOG.ja.md](CHANGELOG.ja.md)

## 라이선스
MIT License ([LICENSE](LICENSE)).
