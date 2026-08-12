"""
RansKnow Phase 0.4 -- ASR jargon spot-check.

Finding B (Section 4.2) establishes that 92.7% of transcripts are ASR
output, 554 of those from whisper_base -- the smallest, least accurate
Whisper checkpoint. This script measures, on a random sample of
whisper_base-transcribed videos, how often a fixed list of
ransomware-relevant jargon terms is missed relative to a stronger
Whisper checkpoint re-transcribing the same audio.

No source audio is stored locally (the released dataset ships
transcripts, not audio), so this re-fetches audio via yt-dlp for the
sampled videos only, transcribes it with a larger Whisper model, and
compares jargon-term occurrence counts against the already-stored
whisper_base transcript. This is an automated proxy for the originally
planned manual audio-vs-transcript listening check: a stronger ASR
model's output is treated as the reference, exact-term OR known-mangled-
variant match counts as "detected" in the weaker transcript, anything
else counts as a miss.

Usage:
    python3 Scripts/phase0_4_asr_spotcheck.py --n 40 --model medium
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

FEATURES_CSV = ROOT / "outputs" / "Knowledge_Agent_Features_1034.csv"
OUT = ROOT / "outputs" / "phase0_4_asr_spotcheck.json"
YTDLP = "yt-dlp"

# Same fixed jargon list as the paper's Phase 0.4 plan, plus the known
# phonetic-mangling map already used for Phase 5.3's simulated-ASR-noise
# work (Scripts/phase5_perturbation_robustness.py) -- reused here as the
# "known mangled variant" set, not reinvented.
JARGON_TERMS = ["lockbit", "mimikatz", "psexec", "cobalt strike", "rclone", "bloodhound"]
KNOWN_MANGLES = {
    "lockbit": ["lock bit"],
    "mimikatz": ["mimi cats", "mimi katz", "mimikats"],
    "psexec": ["p s exec", "sysinternals exec"],
    "cobalt strike": ["cobalt strike", "cs beacon"],
    "rclone": ["are clone", "our clone"],
    "bloodhound": ["blood hound"],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _count_term(text: str, term: str) -> int:
    variants = [term] + KNOWN_MANGLES.get(term, [])
    total = 0
    for v in variants:
        pat = re.escape(v).replace(r"\ ", r"\s+")
        total += len(re.findall(rf"\b{pat}\b", text, flags=re.IGNORECASE))
    return total


def _download_audio(youtube_url: str, out_path: Path, cap_minutes: int) -> bool:
    # Some sampled videos run 40+ minutes; transcribing the whole thing
    # with a stronger model is not the point of a *spot*-check and made
    # a 2-video smoke test take 15+ minutes on one video alone. Capping
    # to the opening cap_minutes bounds runtime per video without biasing
    # jargon-term presence toward any particular part of the corpus (the
    # sampled-video set, not the in-video position, is what's random).
    # --download-sections segfaults this environment's ffmpeg build
    # (confirmed: ffmpeg exit code -11 on every attempt). Trimming via
    # --postprocessor-args instead -- still downloads the full audio
    # (compressed, so bandwidth cost is modest even for long videos) but
    # only feeds the opening cap_minutes to Whisper, which is what
    # actually dominated runtime in the untrimmed smoke test.
    try:
        r = subprocess.run([
            YTDLP, "-x", "--audio-format", "m4a", "--audio-quality", "5",
            "--postprocessor-args", f"ffmpeg:-t {cap_minutes * 60}",
            "--quiet", "-o", str(out_path), youtube_url,
        ], capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and out_path.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-minutes", type=int, default=6,
                     help="only download/transcribe the opening N minutes per video")
    args = ap.parse_args()

    import whisper
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{args.model}' on {device}...")
    model = whisper.load_model(args.model, device=device)

    df = pd.read_csv(FEATURES_CSV)
    pool = df[df["Transcript_Provider"] == "whisper_base"].reset_index(drop=True)
    print(f"whisper_base pool: {len(pool)} videos")
    sample = pool.sample(n=min(args.n, len(pool)), random_state=args.seed).reset_index(drop=True)
    print(f"Sampled {len(sample)} videos (seed={args.seed})")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in sample.iterrows():
            vid = row["Video_ID"]
            url = row["YouTube_URL"]
            base_text = _norm((ROOT / row["Transcript_Path"]).read_text(encoding="utf-8", errors="ignore"))

            audio_path = Path(tmp) / f"{vid}.m4a"
            t0 = time.time()
            ok = _download_audio(url, audio_path, args.cap_minutes)
            if not ok:
                print(f"  [{i+1}/{len(sample)}] {vid}: audio download failed, skipping")
                results.append({"video_id": vid, "status": "download_failed"})
                continue

            try:
                out = model.transcribe(str(audio_path), fp16=(device == "cuda"), verbose=False)
                strong_text = _norm(out["text"])
            except Exception as e:
                print(f"  [{i+1}/{len(sample)}] {vid}: transcription failed ({e}), skipping")
                results.append({"video_id": vid, "status": "transcribe_failed"})
                continue
            finally:
                audio_path.unlink(missing_ok=True)

            per_term = {}
            for term in JARGON_TERMS:
                strong_count = _count_term(strong_text, term)
                base_count = _count_term(base_text, term)
                if strong_count > 0:
                    per_term[term] = {"strong_count": strong_count, "base_count": base_count,
                                       "missed": base_count == 0}

            elapsed = time.time() - t0
            n_terms_present = len(per_term)
            n_missed = sum(1 for v in per_term.values() if v["missed"])
            print(f"  [{i+1}/{len(sample)}] {vid}: {n_terms_present} jargon term(s) in strong transcript, "
                  f"{n_missed} missed by whisper_base ({elapsed:.0f}s)")

            results.append({
                "video_id": vid, "status": "ok",
                "channel_id": row["Channel_ID"], "duration_s": int(row["DurationSeconds"]),
                "per_term": per_term,
            })

    ok_results = [r for r in results if r["status"] == "ok"]
    total_occurrences = sum(sum(t["strong_count"] for t in r["per_term"].values()) for r in ok_results)
    total_present = sum(len(r["per_term"]) for r in ok_results)
    total_missed = sum(sum(1 for t in r["per_term"].values() if t["missed"]) for r in ok_results)

    summary = {
        "n_sampled": len(sample), "n_ok": len(ok_results),
        "n_download_failed": sum(1 for r in results if r["status"] == "download_failed"),
        "n_transcribe_failed": sum(1 for r in results if r["status"] == "transcribe_failed"),
        "model": args.model, "seed": args.seed,
        "n_jargon_term_instances_in_strong_transcript": total_present,
        "n_jargon_term_total_occurrences_in_strong_transcript": total_occurrences,
        "n_jargon_terms_entirely_missed_by_whisper_base": total_missed,
        "miss_rate_by_term_instance": (total_missed / total_present) if total_present else None,
    }
    print("\n" + json.dumps(summary, indent=2))

    OUT.write_text(json.dumps({"summary": summary, "per_video": results}, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
