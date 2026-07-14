"""
RansKnow transcript fetcher
Usage:
    python3 Scripts/fetch_transcripts.py
    python3 Scripts/fetch_transcripts.py --per-channel 20 --scan 150
    python3 Scripts/fetch_transcripts.py --channels C25 C12 C27
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import isodate
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
STATE_FILE  = TRANSCRIPTS / "_state.json"
REGISTRY    = ROOT / "rubrics" / "Dataset_Channel_Registry_Updated_50_fixed_urls.xlsx"
API_KEY     = (ROOT / "rubrics" / "youtube_API.txt").read_text().strip()

KEYWORDS = [
    "ransomware", "extortion", "double extortion",
    "encrypt", "encryption", "decrypt", "data leak",
    "incident response", " ir ", "lockbit", "alphv", "blackcat",
    "conti", "ryuk", "cl0p", "clop", "akira", "revil",
    "sodinokibi", "ransomware-as-a-service", "raas",
]

PREFER_LANGS = ["en", "en-GB", "en-US"]
MIN_SECONDS  = 300  # 5 minutes


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    """Map YouTube video ID → meta.json path for every video already in the dataset."""
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
    # 1) UC id in URL
    m = re.search(r"(UC[a-zA-Z0-9_-]{20,})", url)
    if m:
        return m.group(1)

    # 2) @handle
    m = re.search(r"/@([a-zA-Z0-9_.-]+)", url)
    if m:
        try:
            resp = youtube.channels().list(
                part="id", forHandle=m.group(1)
            ).execute()
            items = resp.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

    # 3) Name search fallback
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
    resp = youtube.channels().list(
        part="contentDetails", id=channel_uc
    ).execute()
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
    """Fallback: search within a channel for ransomware-relevant videos."""
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
                "video_id":        v["id"],
                "title":           s.get("title", ""),
                "description":     s.get("description", ""),
                "published_at":    pub,
                "year":            int(pub[:4]) if pub else None,
                "duration_iso":    dur_iso,
                "duration_seconds": int(
                    isodate.parse_duration(dur_iso).total_seconds()
                ),
            })
    return rows


def _keyword_match(title: str, description: str) -> Tuple[bool, str]:
    text = (title + " " + description).lower()
    matched = [k for k in KEYWORDS if k in text]
    return bool(matched), ";".join(matched)


def _fetch_transcript(video_id: str) -> Optional[str]:
    """Return transcript in timestamped format, or None."""
    try:
        for lang in PREFER_LANGS:
            try:
                segments = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[lang]
                )
                return _format_transcript(segments)
            except Exception:
                pass
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        return _format_transcript(segments)
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return None
    except Exception:
        return None


def _format_transcript(segments: list) -> str:
    """Convert youtube-transcript-api segments to the dataset's timestamped format."""
    lines = []
    for seg in segments:
        start = seg["start"]
        minutes = int(start // 60)
        seconds = int(start % 60)
        lines.append(f"{minutes}:{seconds:02d}")
        lines.append(seg["text"])
    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def fetch(
    per_channel: int = 10,
    scan_recent: int = 100,
    channel_filter: Optional[List[str]] = None,
):
    youtube  = build("youtube", "v3", developerKey=API_KEY)
    df       = pd.read_excel(REGISTRY, sheet_name=0)
    existing = _existing_youtube_ids()
    counter  = _load_counter()

    print(f"Already in dataset:  {len(existing)} videos")
    print(f"Next Video ID:       V{counter:04d}")
    print()

    # Filter to included channels only
    if "Included (Yes/No)" in df.columns:
        df = df[df["Included (Yes/No)"].str.strip().str.lower() == "yes"]

    # Optional channel filter
    if channel_filter:
        df = df[df["Channel_ID"].isin(channel_filter)]

    total_new = 0

    for _, row in df.iterrows():
        ch_id   = str(row["Channel_ID"]).strip()
        ch_name = str(row["Channel_Name"]).strip()
        ch_url  = str(row["Channel_URL"]).strip()

        # Count how many this channel already has
        ch_folder = TRANSCRIPTS / f"{ch_id}_{_sanitize(ch_name)}"
        already   = len(list(ch_folder.glob("V*.txt"))) if ch_folder.exists() else 0
        need      = max(0, per_channel - already)

        if need == 0:
            print(f"[SKIP] {ch_id} {ch_name} — already has {already} videos")
            continue

        print(f"[{ch_id}] {ch_name} — has {already}, fetching up to {need} more ...")

        # Resolve channel
        ch_uc = _resolve_channel_uc(youtube, ch_url, ch_name)
        if not ch_uc:
            print(f"  [WARN] Could not resolve channel UC id — skipping")
            continue

        playlist = _uploads_playlist(youtube, ch_uc)
        if playlist:
            video_ids = _list_video_ids(youtube, playlist, max_n=scan_recent)
        else:
            video_ids = []

        # Fallback: search API when uploads playlist is unavailable or empty
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

        # Filter: duration, keyword, not already fetched
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
            yt_vid  = m["video_id"]
            vid_id  = f"V{counter:04d}"
            txt_file  = ch_folder / f"{vid_id}.txt"
            meta_file = ch_folder / f"{vid_id}.meta.json"

            transcript = _fetch_transcript(yt_vid)
            transcript_available = transcript is not None

            if transcript is None:
                transcript = "[NO TRANSCRIPT AVAILABLE]\n"

            txt_file.write_text(transcript, encoding="utf-8")
            meta_file.write_text(
                json.dumps({
                    "Video_ID":            vid_id,
                    "Channel_ID":          ch_id,
                    "Channel_Name":        ch_name,
                    "Channel_UC":          ch_uc,
                    "YouTube_Video_ID":    yt_vid,
                    "Video_Title":         m["title"],
                    "Year":                m["year"],
                    "PublishedAt":         m["published_at"],
                    "DurationSeconds":     m["duration_seconds"],
                    "DurationISO":         m["duration_iso"],
                    "Matched_Keywords":    m["matched_keywords"],
                    "Transcript_Available": transcript_available,
                }, indent=2),
                encoding="utf-8",
            )

            status = "transcript" if transcript_available else "no transcript"
            print(f"  {vid_id} | {m['title'][:60]} [{status}]")

            existing[yt_vid] = meta_file
            counter += 1
            saved   += 1
            total_new += 1

            # Update state after every video so progress survives interruption
            _save_counter(counter)
            time.sleep(0.3)  # be gentle with the API

        print(f"  Saved {saved} new video(s)")

    print(f"\nDone. {total_new} new video(s) added. Next counter: V{counter:04d}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ransomware transcripts from YouTube")
    parser.add_argument(
        "--per-channel", type=int, default=10,
        help="Target number of videos per channel (default: 10)"
    )
    parser.add_argument(
        "--scan", type=int, default=100,
        help="How many recent videos to scan per channel (default: 100)"
    )
    parser.add_argument(
        "--channels", nargs="+", metavar="C##",
        help="Only process specific channel IDs, e.g. --channels C25 C12 C27"
    )
    args = parser.parse_args()

    fetch(
        per_channel=args.per_channel,
        scan_recent=args.scan,
        channel_filter=args.channels,
    )
