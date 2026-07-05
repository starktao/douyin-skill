---
name: douyin-transcribe
description: Extract audio from a public Douyin video link and transcribe it to Chinese text. Use when the user provides a douyin.com, v.douyin.com, /video/, share, or user-page URL with modal_id and asks to get the speech, audio content, transcript, subtitles, or text from the video.
---

# Douyin Transcribe

Use this skill to turn a public Douyin video URL into a Chinese transcript. The bundled script normalizes common Douyin URL shapes, extracts audio with `yt-dlp`, transcribes with `faster-whisper`, and writes timestamped, raw, and lightly cleaned text files.

## Workflow

1. Run the script from the user's working directory:

```powershell
python "$env:USERPROFILE\.codex\skills\douyin-transcribe\scripts\douyin_transcribe.py" "DOUYIN_URL"
```

2. If the input is a user-page URL containing `modal_id=...`, the script converts it to `https://www.douyin.com/video/<modal_id>`.

3. If extraction fails, retry after the user opens the video in a browser and completes any Douyin verification. Then pass browser cookies through `--yt-dlp-arg`, for example:

```powershell
python "$env:USERPROFILE\.codex\skills\douyin-transcribe\scripts\douyin_transcribe.py" "DOUYIN_URL" --yt-dlp-arg "--cookies-from-browser" --yt-dlp-arg "edge"
```

4. Inspect the generated files:

- `*_metadata.json`: title, creator, duration, track, and source URL.
- `*_audio.mp3`: extracted audio only.
- `*_segments.txt`: timestamped Whisper segments.
- `*_raw.txt`: raw ASR text without timestamps.
- `*_clean.txt`: lightly normalized text.

5. Deliver `*_clean.txt` as the transcript, but check `*_segments.txt` and metadata before finalizing. Correct obvious ASR homophones only when the audio topic, title, or user-provided context supports the correction. Mention uncertain phrases instead of inventing wording.

## Options

- Use `--out-dir PATH` to choose the output directory.
- Use `--model base|small|medium|large-v3` to trade speed for accuracy; default to `small`.
- Use `--context "keyword list"` to improve recognition of names, jargon, tickers, products, or domain terms.
- Use `--keep-video` only when the user also wants the video file retained.

## Requirements

The script expects `yt-dlp`, `ffmpeg`, and Python packages `faster_whisper` and `torch` to be available. If a dependency is missing, install or use the user's existing environment before rerunning.
