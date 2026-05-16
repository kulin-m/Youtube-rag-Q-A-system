"""
app/transcript_extractor.py
Extracts transcripts from YouTube videos, preserving timestamps.

Extraction method priority:
  1. youtube-transcript-api  (most reliable, now returns timestamp data)
  2. yt-dlp subtitle file download
  3. yt-dlp caption URL fetch
  4. Video description fallback
"""

import re
import os
import tempfile
import logging
import urllib.request
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

def _extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

def _validate_youtube_url(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url))

def _clean_vtt(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    deduped, prev = [], None
    for ln in lines:
        if ln != prev:
            deduped.append(ln)
        prev = ln
    return " ".join(deduped)

def _get_video_info(url: str) -> dict:
    try:
        import yt_dlp
        opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:
        return {}

def _try_youtube_transcript_api(video_id: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """Returns both the concatenated plain text AND the raw list of timestamped dicts."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
        for lang_codes in [["en", "en-US", "en-GB"], None]:
            try:
                if lang_codes:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_codes)
                else:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript = transcript_list.find_transcript(
                        transcript_list._manually_created_transcripts or
                        transcript_list._generated_transcripts
                    ).fetch()

                text = " ".join(entry["text"] for entry in transcript)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text.split()) > 50:
                    logger.info(f"[Transcript] youtube-transcript-api succeeded for {video_id}")
                    return text, transcript
            except (NoTranscriptFound, Exception):
                continue
    except Exception as e:
        logger.warning(f"[Transcript] youtube-transcript-api failed: {e}")
    return None, None

def _try_ytdlp_subtitle_file(url: str) -> Optional[str]:
    try:
        import yt_dlp
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-GB"],
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            for fname in os.listdir(tmpdir):
                if fname.endswith(".vtt") or fname.endswith(".srt"):
                    with open(os.path.join(tmpdir, fname), "r", encoding="utf-8", errors="ignore") as f:
                        text = _clean_vtt(f.read())
                    if len(text.split()) > 50:
                        logger.info("[Transcript] yt-dlp subtitle file succeeded.")
                        return text
    except Exception as e:
        logger.warning(f"[Transcript] yt-dlp subtitle file failed: {e}")
    return None

def _try_ytdlp_caption_url(url: str) -> Optional[str]:
    try:
        import yt_dlp
        opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        for src in [info.get("subtitles", {}), info.get("automatic_captions", {})]:
            for lang in ["en", "en-US", "en-GB"]:
                cap_list = src.get(lang, [])
                for cap in cap_list:
                    if cap.get("ext") in ("vtt", "srv3", "json3"):
                        cap_url = cap.get("url", "")
                        if not cap_url:
                            continue
                        try:
                            with urllib.request.urlopen(cap_url, timeout=15) as resp:
                                raw = resp.read().decode("utf-8", errors="ignore")
                            text = _clean_vtt(raw)
                            if len(text.split()) > 50:
                                logger.info("[Transcript] yt-dlp caption URL fetch succeeded.")
                                return text
                        except Exception:
                            continue
    except Exception as e:
        logger.warning(f"[Transcript] yt-dlp caption URL fetch failed: {e}")
    return None

def _try_description_fallback(url: str) -> Optional[str]:
    try:
        info = _get_video_info(url)
        desc = info.get("description", "")
        if len(desc.split()) > 100:
            logger.warning("[Transcript] Using video description as fallback transcript.")
            return desc
    except Exception as e:
        logger.warning(f"[Transcript] Description fallback failed: {e}")
    return None

def extract_transcript(url: str) -> dict:
    url = url.strip()
    if not _validate_youtube_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")

    video_id = _extract_video_id(url)
    info = _get_video_info(url)
    title = info.get("title", "Unknown Video")

    if video_id:
        text, data = _try_youtube_transcript_api(video_id)
        if text:
            return {
                "url": url, 
                "title": title, 
                "transcript": text, 
                "transcript_data": data, # NEW: raw list of timestamp dicts
                "language": "en", 
                "method": "youtube-transcript-api"
            }

    text = _try_ytdlp_subtitle_file(url)
    if text:
        return {"url": url, "title": title, "transcript": text, "transcript_data": None, "language": "en", "method": "ytdlp-subtitle-file"}

    text = _try_ytdlp_caption_url(url)
    if text:
        return {"url": url, "title": title, "transcript": text, "transcript_data": None, "language": "en", "method": "ytdlp-caption-url"}

    text = _try_description_fallback(url)
    if text:
        return {"url": url, "title": title, "transcript": text, "transcript_data": None, "language": "en", "method": "description-fallback"}

    raise RuntimeError(
        f"Could not extract transcript for: {url}\n"
        "All methods exhausted. The video may have no captions, be age-restricted, or be private."
    )

def extract_multiple_transcripts(urls: list) -> list:
    if len(urls) > 3:
        raise ValueError("Maximum 3 YouTube URLs are supported.")
    return [extract_transcript(url) for url in urls]