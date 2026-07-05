#!/usr/bin/env python
"""Extract audio from a Douyin URL and transcribe it with faster-whisper."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse


VIDEO_ID_RE = re.compile(r"(?:modal_id=|/video/|/share/video/)(\d{16,22})")


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing command: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        joined = " ".join(cmd)
        message = exc.stderr.strip() or exc.stdout.strip()
        raise SystemExit(f"Command failed: {joined}\n{message}") from exc


def normalize_douyin_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    modal_id = query.get("modal_id", [None])[0]
    if modal_id and modal_id.isdigit():
        return f"https://www.douyin.com/video/{modal_id}"

    match = VIDEO_ID_RE.search(url)
    if match:
        return f"https://www.douyin.com/video/{match.group(1)}"

    return url


def extract_video_id(url: str) -> str | None:
    match = VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def safe_stem(value: str, fallback: str = "douyin") -> str:
    value = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", value).strip(" ._")
    value = re.sub(r"\s+", " ", value)
    return (value[:80] or fallback).strip()


def load_metadata(url: str, ytdlp_args: list[str]) -> dict:
    cmd = ["yt-dlp", "--dump-single-json", "--no-playlist", "--skip-download", *ytdlp_args, url]
    result = run(cmd)
    text = result.stdout.strip()
    if not text or text == "null":
        return {}
    return json.loads(text)


def download_audio(url: str, out_dir: Path, stem: str, ytdlp_args: list[str], keep_video: bool) -> Path:
    output_template = str(out_dir / f"{stem}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        *ytdlp_args,
        url,
    ]
    if keep_video:
        cmd.insert(1, "-k")

    run(cmd, cwd=out_dir)
    audio = out_dir / f"{stem}.mp3"
    if audio.exists():
        return audio

    candidates = sorted(out_dir.glob(f"{stem}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.suffix.lower() in {".mp3", ".m4a", ".aac", ".wav", ".opus"}:
            return candidate
    raise SystemExit(f"Audio file was not created for stem: {stem}")


def normalize_transcript(text: str) -> str:
    replacements = {
        "\u4e2d\u6b63\u7ea2\u5229": "\u4e2d\u8bc1\u7ea2\u5229",
        "\u4e2d\u6b63\u7d05\u5229": "\u4e2d\u8bc1\u7ea2\u5229",
        "\u53e0\u9ebb": "\u8dcc\u9ebb",
        "\u5b9a\u982d": "\u5b9a\u6295",
        "\u5b9a\u5934": "\u5b9a\u6295",
        "\u5206\u6a5f": "\u5206\u7ea7",
        "\u5206\u673a": "\u5206\u7ea7",
        "\u8ecd\u7dda": "\u5747\u7ebf",
        "\u519b\u7ebf": "\u5747\u7ebf",
        "\u54c0\u751f": "\u54c0\u58f0",
        "\u6328\u751f": "\u54c0\u58f0",
        "\u8d70\u843d": "\u8d70\u5f31",
        "\u80a1\u606f\u529b": "\u80a1\u606f\u7387",
        "\u52a0\u6301\u76c8": "\u52a0\u6b62\u76c8",
        "\u4e00\u5ea6\u4e00\u5f84": "\u4e00\u52a8\u4e00\u9759",
        "\u8d2a\u4fdd": "\u644a\u8584",
        "\u6307\u8d62": "\u6b62\u76c8",
        "1\u00be": "\u4e09\u5206\u4e4b\u4e00",
        "\u5360\u5f97\u592a\u9ad8": "\u6da8\u5f97\u592a\u9ad8",
        "\u5206\u7ea7\u5356": "\u5206\u7ea7\u4e70",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def paragraphize(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = normalize_transcript(text)
    parts = re.split(r"(?<=[\u3002\uff01\uff1f!?])\s*", text)
    if len(parts) <= 1:
        chunks = re.findall(r".{1,120}(?:\s|$)", text) or [text]
        return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return "\n\n".join(part.strip() for part in parts if part.strip())


def transcribe(
    audio: Path,
    out_dir: Path,
    stem: str,
    model_name: str,
    language: str,
    context: str,
) -> tuple[Path, Path, Path]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Missing Python package: faster_whisper") from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    initial_prompt = context.strip() or None
    segments, info = model.transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
        initial_prompt=initial_prompt,
    )

    segment_lines = [f"language={info.language} prob={info.language_probability:.4f} duration={info.duration:.3f}"]
    raw_lines: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        segment_lines.append(f"[{seg.start:06.2f}-{seg.end:06.2f}] {text}")
        raw_lines.append(text)

    segments_path = out_dir / f"{stem}_segments.txt"
    raw_path = out_dir / f"{stem}_raw.txt"
    clean_path = out_dir / f"{stem}_clean.txt"

    raw_text = "\n".join(raw_lines).strip() + "\n"
    clean_text = paragraphize(raw_lines).strip() + "\n"

    segments_path.write_text("\n".join(segment_lines) + "\n", encoding="utf-8-sig")
    raw_path.write_text(raw_text, encoding="utf-8-sig")
    clean_path.write_text(clean_text, encoding="utf-8-sig")
    return segments_path, raw_path, clean_path


def check_requirements() -> None:
    missing = [name for name in ("yt-dlp", "ffmpeg") if shutil.which(name) is None]
    if missing:
        raise SystemExit("Missing required command(s): " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and transcribe a Douyin video's audio.")
    parser.add_argument("url", help="Douyin video/share/user URL. User URLs with modal_id are supported.")
    parser.add_argument("--out-dir", default=".", help="Directory for audio and transcript outputs.")
    parser.add_argument("--model", default="small", help="faster-whisper model name. Default: small.")
    parser.add_argument("--language", default="zh", help="Whisper language code. Default: zh.")
    parser.add_argument("--context", default="", help="Recognition prompt with names, jargon, or topic keywords.")
    parser.add_argument("--yt-dlp-arg", action="append", default=[], help="Extra yt-dlp argument. Repeat per token.")
    parser.add_argument("--keep-video", action="store_true", help="Keep the temporary video downloaded by yt-dlp.")
    args = parser.parse_args()

    check_requirements()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    url = normalize_douyin_url(args.url)
    metadata = load_metadata(url, args.yt_dlp_arg)

    video_id = str(metadata.get("id") or extract_video_id(url) or "douyin")
    title = str(metadata.get("title") or metadata.get("description") or video_id)
    audio_stem = safe_stem(f"{video_id}_audio", "douyin_audio")

    metadata_path = out_dir / f"{video_id}_metadata.json"
    metadata_payload = {
        "input_url": args.url,
        "normalized_url": url,
        "id": metadata.get("id"),
        "title": metadata.get("title"),
        "creator": metadata.get("creator") or metadata.get("uploader"),
        "track": metadata.get("track"),
        "duration": metadata.get("duration"),
        "upload_date": metadata.get("upload_date"),
        "webpage_url": metadata.get("webpage_url"),
    }
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig")

    prompt_parts = [title]
    if metadata.get("description") and metadata.get("description") != title:
        prompt_parts.append(str(metadata["description"]))
    if args.context:
        prompt_parts.append(args.context)
    prompt = "\u3002".join(part for part in prompt_parts if part)

    audio = download_audio(url, out_dir, audio_stem, args.yt_dlp_arg, args.keep_video)
    segments_path, raw_path, clean_path = transcribe(audio, out_dir, video_id, args.model, args.language, prompt)

    print("Douyin transcript complete")
    print(f"metadata: {metadata_path}")
    print(f"audio: {audio}")
    print(f"segments: {segments_path}")
    print(f"raw: {raw_path}")
    print(f"clean: {clean_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
