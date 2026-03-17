from local_ai_core.local_inference import LocalInferenceEngine


def test_sanitize_generated_answer_dedupes_repeated_sentences():
    raw = "찾았습니다. 찾았습니다. 찾았습니다. 가장 관련 파일은 notes.md입니다. 가장 관련 파일은 notes.md입니다."
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert cleaned.count("찾았습니다.") <= 2
    assert "notes.md" in cleaned


def test_sanitize_generated_answer_removes_prompt_echo():
    prompt = "Mode: GENERAL Question: test"
    raw = f"{prompt} Mode: GENERAL Question: test 결과입니다."
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt=prompt)
    assert cleaned
    assert cleaned != raw
