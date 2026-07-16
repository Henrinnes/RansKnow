"""
RansKnow transcript fetcher
Usage:
    python3 Scripts/fetch_transcripts.py
    python3 Scripts/fetch_transcripts.py --per-channel 20 --scan 150
    python3 Scripts/fetch_transcripts.py --channels C25 C12 C27
    python3 Scripts/fetch_transcripts.py --retry-missing
    python3 Scripts/fetch_transcripts.py --retry-missing --whisper-model small
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import isodate
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
STATE_FILE  = TRANSCRIPTS / "_state.json"
REGISTRY    = ROOT / "rubrics" / "Dataset_Channel_Registry_Updated_50_fixed_urls.xlsx"
YTDLP       = shutil.which("yt-dlp") or os.path.expanduser("~/.local/bin/yt-dlp")

KEYWORDS = [
    "ransomware", "extortion", "double extortion",
    "encrypt", "encryption", "decrypt", "data leak",
    "incident response", " ir ", "lockbit", "alphv", "blackcat",
    "conti", "ryuk", "cl0p", "clop", "akira", "revil",
    "sodinokibi", "ransomware-as-a-service", "raas",
]

PREFER_LANGS = ["en", "en-GB", "en-US"]
MIN_SECONDS  = 300  # 5 minutes

# ── API key (lazy — only needed for fetch(), not retry_missing()) ──────────────

def _load_api_key() -> str:
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        key_file = ROOT / "rubrics" / "youtube_API.txt"
        if key_file.exists():
            key = key_file.read_text().strip()
    if not key or key == "YOUR_YOUTUBE_API_KEY":
        raise ValueError(
            "YouTube API key not found. "
            "Set the YOUTUBE_API_KEY environment variable, "
            "or put your key in rubrics/youtube_API.txt"
        )
    return key


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\- ]+", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:max_len]


def _load_counter() -> int:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())["global_counter"]
    return 1


def _save_counter(n: int) -> None:
    STATE_FILE.write_text(json.dumps({"global_counter": n}, indent=2))


def _existing_youtube_ids() -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for meta_file in TRANSCRIPTS.rglob("*.meta.json"):
        try:
            data = json.loads(meta_file.read_text())
            yt_id = data.get("YouTube_Video_ID")
            if yt_id:
                index[yt_id] = meta_file
        except Exception:
            pass
    return index


def _resolve_channel_uc(youtube, url: str, name: str) -> Optional[str]:
    m = re.search(r"(UC[a-zA-Z0-9_-]{20,})", url)
    if m:
        return m.group(1)

    m = re.search(r"/@([a-zA-Z0-9_.-]+)", url)
    if m:
        try:
            resp = youtube.channels().list(part="id", forHandle=m.group(1)).execute()
            items = resp.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

    try:
        resp = youtube.search().list(
            part="snippet", q=name, type="channel", maxResults=1
        ).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["snippet"]["channelId"]
    except Exception:
        pass

    return None


def _uploads_playlist(youtube, channel_uc: str) -> Optional[str]:
    resp = youtube.channels().list(part="contentDetails", id=channel_uc).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def _list_video_ids(youtube, playlist_id: str, max_n: int) -> List[str]:
    ids: List[str] = []
    page_token = None
    while len(ids) < max_n:
        try:
            resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=min(50, max_n - len(ids)),
                pageToken=page_token,
            ).execute()
        except HttpError:
            break
        for item in resp["items"]:
            ids.append(item["contentDetails"]["videoId"])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return ids


def _search_video_ids(youtube, channel_uc: str, query: str, max_n: int) -> List[str]:
    ids: List[str] = []
    page_token = None
    while len(ids) < max_n:
        try:
            resp = youtube.search().list(
                part="id",
                channelId=channel_uc,
                q=query,
                type="video",
                order="date",
                maxResults=min(50, max_n - len(ids)),
                pageToken=page_token,
            ).execute()
        except HttpError:
            break
        for item in resp.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                ids.append(vid)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return ids


def _video_metadata(youtube, video_ids: List[str]) -> List[dict]:
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet,contentDetails", id=",".join(batch)
        ).execute()
        for v in resp.get("items", []):
            s = v["snippet"]
            pub = s.get("publishedAt", "")
            dur_iso = v["contentDetails"].get("duration", "PT0S")
            rows.append({
                "video_id":         v["id"],
                "title":            s.get("title", ""),
                "description":      s.get("description", ""),
                "published_at":     pub,
                "year":             int(pub[:4]) if pub else None,
                "duration_iso":     dur_iso,
                "duration_seconds": int(isodate.parse_duration(dur_iso).total_seconds()),
            })
    return rows


def _keyword_match(title: str, description: str) -> Tuple[bool, str]:
    text = (title + " " + description).lower()
    matched = [k for k in KEYWORDS if k in text]
    return bool(matched), ";".join(matched)


# ── Transcript methods (cascade: API → yt-dlp captions → Whisper) ─────────────

def _fmt_segments(segments) -> str:
    """Format a list of transcript segments into M:SS\\ntext lines."""
    lines = []
    for seg in segments:
        start = seg.start if hasattr(seg, "start") else seg["start"]
        text  = seg.text  if hasattr(seg, "text")  else seg["text"]
        text  = (text or "").strip()
        if not text:
            continue
        m, s = divmod(int(start), 60)
        lines.append(f"{m}:{s:02d}")
        lines.append(text)
    return "\n".join(lines)


def _fetch_via_api(video_id: str) -> Optional[str]:
    """Method 1: youtube-transcript-api (fastest, works when not IP-blocked)."""
    try:
        api = YouTubeTranscriptApi()
        for lang in PREFER_LANGS:
            try:
                t = api.fetch(video_id, languages=[lang])
                return _fmt_segments(t.snippets)
            except Exception:
                pass
        t = api.fetch(video_id)
        return _fmt_segments(t.snippets)
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return None
    except Exception:
        return None


def _parse_srv1(xml: str) -> Optional[str]:
    """Parse yt-dlp's srv1 subtitle XML into timestamped text."""
    lines = []
    for start_ms, text in re.findall(r'start="(\d+)"[^>]*>([^<]*)</text>', xml):
        m, s = divmod(int(start_ms) // 1000, 60)
        text = re.sub(r"<[^>]+>", "", text).strip()
        for esc, ch in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                        ("&#39;", "'"), ("&quot;", '"')]:
            text = text.replace(esc, ch)
        if text:
            lines.append(f"{m}:{s:02d}")
            lines.append(text)
    return "\n".join(lines) if lines else None


def _fetch_via_ytdlp_captions(video_id: str) -> Optional[str]:
    """Method 2: yt-dlp auto-generated captions (works when API is IP-blocked)."""
    if not YTDLP or not Path(YTDLP).exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([
            YTDLP, "--write-auto-subs", "--skip-download",
            "--sub-langs", "en,en-orig,en-US,en-GB",
            "--sub-format", "srv1", "--quiet",
            "-o", str(Path(tmp) / "sub"),
            f"https://www.youtube.com/watch?v={video_id}",
        ], capture_output=True, timeout=60)
        for f in Path(tmp).iterdir():
            if f.suffix == ".srv1":
                return _parse_srv1(f.read_text())
    return None


_whisper_model_cache: dict = {}

def _load_whisper_model(model_name: str):
    """Lazy-load a Whisper model, caching it for reuse across videos."""
    if model_name not in _whisper_model_cache:
        try:
            import whisper
            print(f"  [Whisper] Loading '{model_name}' model (one-time download if first use)...",
                  flush=True)
            _whisper_model_cache[model_name] = whisper.load_model(model_name)
            print(f"  [Whisper] Model ready.", flush=True)
        except ImportError:
            print("  [Whisper] openai-whisper not installed. Run: pip install openai-whisper",
                  file=sys.stderr)
            _whisper_model_cache[model_name] = None
    return _whisper_model_cache[model_name]


def _fetch_via_whisper(video_id: str, model_name: str = "base") -> Optional[str]:
    """Method 3: download audio with yt-dlp and transcribe locally with Whisper."""
    if not YTDLP or not Path(YTDLP).exists():
        return None
    model = _load_whisper_model(model_name)
    if model is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.m4a"
        try:
            r = subprocess.run([
                YTDLP, "-x",
                "--audio-format", "m4a",
                "--audio-quality", "5",   # ~128 kbps — sufficient for speech
                "--quiet",
                "-o", str(audio_path),
                f"https://www.youtube.com/watch?v={video_id}",
            ], capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            return None

        if r.returncode != 0 or not audio_path.exists():
            return None

        try:
            result = model.transcribe(str(audio_path), fp16=False, verbose=False)
        except Exception:
            return None

        lines = []
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            m, s = divmod(int(seg["start"]), 60)
            lines.append(f"{m}:{s:02d}")
            lines.append(text)

        return "\n".join(lines) if lines else None


def _fetch_transcript(video_id: str, whisper_model: str = "base") -> Tuple[Optional[str], str]:
    """
    Try all three methods in order.
    Returns (transcript_text, method_name) or (None, "").
    """
    t = _fetch_via_api(video_id)
    if t:
        return t, "youtube_api"

    t = _fetch_via_ytdlp_captions(video_id)
    if t:
        return t, "ytdlp_captions"

    t = _fetch_via_whisper(video_id, whisper_model)
    if t:
        return t, f"whisper_{whisper_model}"

    return None, ""


# ── Retry missing ─────────────────────────────────────────────────────────────

def retry_missing(whisper_model: str = "base"):
    """Re-process every video that still has [NO TRANSCRIPT AVAILABLE]."""
    flagged = []
    for meta_file in sorted(TRANSCRIPTS.rglob("*.meta.json")):
        txt_path = meta_file.with_name(meta_file.name.replace(".meta.json", ".txt"))
        if not txt_path.exists():
            continue
        if not txt_path.read_text(encoding="utf-8").strip().startswith("[NO TRANSCRIPT AVAILABLE]"):
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        flagged.append((meta, txt_path, meta_file))

    if not flagged:
        print("No missing transcripts found — nothing to do.")
        return

    print(f"Retrying {len(flagged)} missing transcripts "
          f"(API → yt-dlp captions → Whisper {whisper_model})...\n")

    recovered = still_missing = 0
    method_counts: Dict[str, int] = {}

    for i, (meta, txt_path, meta_file) in enumerate(flagged, 1):
        yt_id  = meta["YouTube_Video_ID"]
        vid_id = meta["Video_ID"]

        transcript, method = _fetch_transcript(yt_id, whisper_model)

        if transcript:
            txt_path.write_text(transcript, encoding="utf-8")
            meta["Transcript_Available"] = True
            meta["Transcript_Provider"]  = method
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            recovered += 1
            method_counts[method] = method_counts.get(method, 0) + 1
            print(f"  [{i}/{len(flagged)}] {vid_id} ✓ ({method})", flush=True)
        else:
            still_missing += 1
            print(f"  [{i}/{len(flagged)}] {vid_id} ✗ (no transcript available)", flush=True)

        time.sleep(0.3)

    print(f"\n{'─'*40}")
    print(f"Recovered:     {recovered} / {len(flagged)}")
    for m, n in method_counts.items():
        print(f"  via {m}: {n}")
    print(f"Still missing: {still_missing}")


# ── Main fetch pipeline ───────────────────────────────────────────────────────

def fetch(
    per_channel: int = 10,
    scan_recent: int = 100,
    channel_filter: Optional[List[str]] = None,
    whisper_model: str = "base",
):
    api_key = _load_api_key()
    youtube  = build("youtube", "v3", developerKey=api_key)
    df       = pd.read_excel(REGISTRY, sheet_name=0)
    existing = _existing_youtube_ids()
    counter  = _load_counter()

    print(f"Already in dataset:  {len(existing)} videos")
    print(f"Next Video ID:       V{counter:04d}")
    print()

    if "Included (Yes/No)" in df.columns:
        df = df[df["Included (Yes/No)"].str.strip().str.lower() == "yes"]

    if channel_filter:
        df = df[df["Channel_ID"].isin(channel_filter)]

    total_new = 0

    for _, row in df.iterrows():
        ch_id   = str(row["Channel_ID"]).strip()
        ch_name = str(row["Channel_Name"]).strip()
        ch_url  = str(row["Channel_URL"]).strip()

        ch_folder = TRANSCRIPTS / f"{ch_id}_{_sanitize(ch_name)}"
        already   = len(list(ch_folder.glob("V*.txt"))) if ch_folder.exists() else 0
        need      = max(0, per_channel - already)

        if need == 0:
            print(f"[SKIP] {ch_id} {ch_name} — already has {already} videos")
            continue

        print(f"[{ch_id}] {ch_name} — has {already}, fetching up to {need} more ...")

        ch_uc = _resolve_channel_uc(youtube, ch_url, ch_name)
        if not ch_uc:
            print(f"  [WARN] Could not resolve channel UC id — skipping")
            continue

        playlist = _uploads_playlist(youtube, ch_uc)
        if playlist:
            video_ids = _list_video_ids(youtube, playlist, max_n=scan_recent)
        else:
            video_ids = []

        if not video_ids:
            print(f"  [INFO] Uploads playlist unavailable — using search fallback")
            video_ids = _search_video_ids(
                youtube, ch_uc,
                query="ransomware malware incident response threat",
                max_n=scan_recent,
            )

        if not video_ids:
            print(f"  [WARN] No videos found — skipping")
            continue

        meta_list = _video_metadata(youtube, video_ids)

        candidates = []
        for m in meta_list:
            if m["duration_seconds"] < MIN_SECONDS:
                continue
            if m["video_id"] in existing:
                continue
            matched, kw = _keyword_match(m["title"], m["description"])
            if not matched:
                continue
            m["matched_keywords"] = kw
            candidates.append(m)

        candidates = candidates[:need]

        if not candidates:
            print(f"  No new matching videos found")
            continue

        ch_folder.mkdir(parents=True, exist_ok=True)
        saved = 0

        for m in candidates:
            yt_vid    = m["video_id"]
            vid_id    = f"V{counter:04d}"
            txt_file  = ch_folder / f"{vid_id}.txt"
            meta_file = ch_folder / f"{vid_id}.meta.json"

            transcript, method = _fetch_transcript(yt_vid, whisper_model)
            transcript_available = transcript is not None

            txt_file.write_text(
                transcript if transcript else "[NO TRANSCRIPT AVAILABLE]\n",
                encoding="utf-8",
            )
            meta_file.write_text(
                json.dumps({
                    "Video_ID":             vid_id,
                    "Channel_ID":           ch_id,
                    "Channel_Name":         ch_name,
                    "Channel_UC":           ch_uc,
                    "YouTube_Video_ID":     yt_vid,
                    "Video_Title":          m["title"],
                    "Year":                 m["year"],
                    "PublishedAt":          m["published_at"],
                    "DurationSeconds":      m["duration_seconds"],
                    "DurationISO":          m["duration_iso"],
                    "Matched_Keywords":     m["matched_keywords"],
                    "Transcript_Available": transcript_available,
                    "Transcript_Provider":  method or None,
                }, indent=2),
                encoding="utf-8",
            )

            status = f"✓ {method}" if transcript_available else "✗ no transcript"
            print(f"  {vid_id} | {m['title'][:55]} [{status}]")

            existing[yt_vid] = meta_file
            counter  += 1
            saved    += 1
            total_new += 1

            _save_counter(counter)
            time.sleep(0.3)

        print(f"  Saved {saved} new video(s)")

    print(f"\nDone. {total_new} new video(s) added. Next counter: V{counter:04d}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ransomware transcripts from YouTube")
    parser.add_argument(
        "--per-channel", type=int, default=10,
        help="Target number of videos per channel (default: 10)",
    )
    parser.add_argument(
        "--scan", type=int, default=100,
        help="How many recent videos to scan per channel (default: 100)",
    )
    parser.add_argument(
        "--channels", nargs="+", metavar="C##",
        help="Only process specific channel IDs, e.g. --channels C25 C12 C27",
    )
    parser.add_argument(
        "--retry-missing", action="store_true",
        help="Re-process all videos that have [NO TRANSCRIPT AVAILABLE]",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for audio transcription fallback (default: base)",
    )
    args = parser.parse_args()

    if args.retry_missing:
        retry_missing(whisper_model=args.whisper_model)
    else:
        fetch(
            per_channel=args.per_channel,
            scan_recent=args.scan,
            channel_filter=args.channels,
            whisper_model=args.whisper_model,
        )
