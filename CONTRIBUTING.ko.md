# PLOS for Mac 기여 가이드 (한국어)

기여해 주셔서 감사합니다.

영문 원본 가이드는 [CONTRIBUTING.md](CONTRIBUTING.md) 입니다.

## 시작 전
- [README.ko.md](README.ko.md) 또는 [README.md](README.md) 확인
- 중복 작업 방지를 위해 이슈/PR 검색
- 가능한 한 작은 단위로 변경

## 개발 환경
```bash
git clone https://github.com/adgk2349/PLOS-for-Mac.git
cd PLOS-for-Mac

cd sidecar
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[test]'
```

## 브랜치/PR
- 기능 브랜치에서 작업 후 `main` 대상으로 PR 생성
- 권장 네이밍: `codex/<scope>-<summary>`, `feat/...`, `fix/...`

## 테스트
```bash
cd sidecar
source .venv/bin/activate
pytest -q
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

Swift:
```bash
xcodebuild -project PLOS.xcodeproj -scheme PLOS -destination 'platform=macOS' test
```

## PR 체크리스트
- [ ] 변경 내용 문서화
- [ ] 동작 변경에 대한 테스트 추가/수정
- [ ] 관련 테스트 통과
- [ ] 사용자 영향이 있으면 README/문서 갱신
- [ ] 비밀키/토큰 미포함 확인

## 보안/개인정보
- API 키/시크릿 커밋 금지
- 세션 메모리 격리 보장 유지
- 민감 데이터는 최소 수집/저장 원칙 적용
