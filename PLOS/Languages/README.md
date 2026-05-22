# Language Pack Guide

Contributors can add a new UI language by dropping one JSON file into this folder.

## 1) File naming

- Use either:
  - 3-letter id file name (example: `kor.json`, `eng.json`, `jpn.json`)
  - 2-letter short alias file name (example: `kr.json`, `en.json`, `jp.json`)

The loader normalizes known aliases (`kr` -> `kor`, `jp` -> `jpn`) automatically.

## 2) File shape

```json
{
  "id": "fr",
  "iso": "fr",
  "display_name": "French",
  "native_name": "Français",
  "is_default": false,
  "strings": {
    "settings.title": "Paramètres"
  }
}
```

## 3) Validation rules

- `id` must match the file name after normalization.
  - example: `kr.json` can use `"id": "kr"` or `"id": "kor"`.
- `iso` should be a valid language code used by sidecar (`fr`, `de`, `es`, etc.).
- Empty string values are ignored.

## 4) Fallback behavior

- Missing keys fall back to English, then Korean.
- `auto` language selection follows system locale, then falls back safely.

