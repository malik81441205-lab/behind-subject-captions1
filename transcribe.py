"""Speech -> words with exact timestamps (OpenAI Whisper, runs locally, free).
The result is cached next to the video as <video>.words.json, so the slow step runs only once.
You can open that JSON and fix any misheard word, then re-render - no re-transcribing needed."""
import json, os


def cache_path(video: str) -> str:
    return os.path.splitext(video)[0] + ".words.json"


def transcribe(video: str, model: str = "base", language: str = None, use_cache: bool = True):
    """-> list of {"text","start","end"} dicts (seconds)."""
    cp = cache_path(video)
    if use_cache and os.path.exists(cp):
        print(f"Using saved transcript: {cp}")
        return json.load(open(cp, encoding="utf-8"))["words"]
    try:
        import whisper
    except ImportError:
        raise SystemExit("Auto captions need Whisper:  pip install openai-whisper")
    print(f"Listening to the video (Whisper '{model}' model) - the first run downloads the model...")
    result = whisper.load_model(model).transcribe(video, word_timestamps=True, language=language, fp16=False)
    words = words_from_whisper(result)
    if not words:
        raise SystemExit("No speech found in this video. Provide an .srt with --srt instead.")
    json.dump({"language": result.get("language"), "words": words}, open(cp, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"Found {len(words)} words. Saved transcript -> {cp}")
    return words


def words_from_whisper(result: dict):
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w["word"].strip()
            if not text:
                continue
            start, end = float(w["start"]), float(w["end"])
            if end <= start:
                end = start + 0.08          # Whisper sometimes returns zero-length words
            words.append({"text": text, "start": round(start, 3), "end": round(end, 3)})
    return words
