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


def test_sanitize_generated_answer_removes_answer_prefix_and_repeated_blocks():
    raw = (
        "Answer: 지금 강의는 컴퓨터 공학부 2024년 2학기 3~4학년에 대한 강의입니다. "
        "강의는 NE322-A 강의실에서 진행됩니다. "
        "지금 강의는 컴퓨터 공학부 2024년 2학기 3~4학년에 대한 강의입니다. "
        "강의는 NE322-A 강의실에서 진행됩니다."
    )
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert not cleaned.lower().startswith("answer:")
    assert cleaned.count("NE322-A") == 1


def test_sanitize_generated_answer_caps_long_unpunctuated_segment():
    raw = "token " * 900
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert len(cleaned) <= 1210
    assert cleaned


def test_sanitize_generated_answer_removes_reasoning_leak_lines():
    raw = (
        "User: 안녕하세요\n"
        "Follow-up question: 오늘 날씨는?\n"
        "Okay, let's see. The user asked about weather.\n"
        "안녕하세요. 오늘은 어느 지역 날씨를 확인할까요?"
    )
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert "Follow-up question" not in cleaned
    assert "Okay, let's see" not in cleaned
    assert "어느 지역 날씨" in cleaned


def test_sanitize_generated_answer_returns_empty_when_only_thought_leak():
    raw = "User: hi\nOkay, let's see. I should infer what the user means."
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert cleaned == ""
