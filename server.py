"""
HumToScore - Basic Pitch Server
Receives WAV/MP3 audio, runs Spotify's Basic Pitch, returns detected notes as JSON.
Deployed on Railway.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

app = FastAPI(title="HumToScore")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ─── Tuning parameters ─────────────────────────────────────
# These dramatically affect accuracy for humming/singing input.
# Basic Pitch detects many ghost notes — we filter aggressively.

MIN_CONFIDENCE = 0.4    # Drop notes with velocity/confidence below this (0-1)
MIN_DURATION_SEC = 0.08 # Drop notes shorter than 80ms (pitch artifacts)
ONSET_THRESHOLD = 0.5   # Basic Pitch onset sensitivity (higher = fewer note splits)
FRAME_THRESHOLD = 0.3   # Basic Pitch frame threshold (higher = stricter pitch detection)
MIN_NOTE_LENGTH = 50    # Basic Pitch minimum note length in frames (~58ms each)


def midi_to_note_name(midi_pitch: int) -> str:
    octave = (midi_pitch // 12) - 1
    return f"{NOTE_NAMES[midi_pitch % 12]}{octave}"


def quantize_duration(duration: float, beat_duration: float) -> dict:
    beats = duration / beat_duration
    standard = [
        (0.25, "sixteenth"), (0.5, "eighth"), (0.75, "dotted_eighth"),
        (1.0, "quarter"), (1.5, "dotted_quarter"), (2.0, "half"),
        (3.0, "dotted_half"), (4.0, "whole"),
    ]
    closest = min(standard, key=lambda x: abs(x[0] - beats))
    return {"beats": float(closest[0]), "name": closest[1]}


def estimate_bpm(notes: list) -> float:
    if len(notes) < 3:
        return 120.0

    onsets = [n[0] for n in notes]
    intervals = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
    intervals = [i for i in intervals if 0.15 < i < 2.0]

    if not intervals:
        return 120.0

    intervals.sort()
    median = intervals[len(intervals) // 2]
    bpm = 60.0 / median

    while bpm < 60: bpm *= 2
    while bpm > 200: bpm /= 2

    return round(bpm)


def detect_key(midi_pitches: list) -> dict:
    if not midi_pitches:
        return {"key": "C", "mode": "major"}

    pc = [0] * 12
    for p in midi_pitches:
        pc[int(p) % 12] += 1

    major = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    minor = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

    best_key, best_mode, best_score = 0, "major", -1
    for root in range(12):
        maj_score = sum(pc[i] * major[(i - root) % 12] for i in range(12))
        min_score = sum(pc[i] * minor[(i - root) % 12] for i in range(12))
        if maj_score > best_score:
            best_score, best_key, best_mode = maj_score, root, "major"
        if min_score > best_score:
            best_score, best_key, best_mode = min_score, root, "minor"

    return {"key": NOTE_NAMES[best_key], "mode": best_mode}


def merge_overlapping_notes(note_events: list) -> list:
    """
    When humming, Basic Pitch sometimes splits one sustained note into
    multiple overlapping notes at the same pitch. Merge them.
    """
    if not note_events:
        return note_events

    # Sort by start time, then pitch
    sorted_notes = sorted(note_events, key=lambda n: (n[0], n[2]))
    merged = [list(sorted_notes[0])]

    for start, end, pitch, velocity, bends in sorted_notes[1:]:
        prev = merged[-1]
        # Same pitch, starts within 100ms of previous note's end → merge
        if int(pitch) == int(prev[2]) and start <= prev[1] + 0.1:
            prev[1] = max(prev[1], end)  # extend end time
            prev[3] = max(prev[3], velocity)  # keep higher confidence
        else:
            merged.append([start, end, pitch, velocity, bends])

    return [tuple(n) for n in merged]


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix = ".mp3" if (audio.filename and audio.filename.endswith(".mp3")) else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        # Run Basic Pitch with tuned parameters for voice/humming
        model_output, midi_data, note_events = predict(
            tmp_path,
            onset_threshold=ONSET_THRESHOLD,
            frame_threshold=FRAME_THRESHOLD,
            minimum_note_length=MIN_NOTE_LENGTH,
        )

        if not note_events:
            return {"success": False, "error": "No notes detected", "notes": []}

        # ─── POST-PROCESSING FOR ACCURACY ───────────────────

        # 1) Filter by confidence/velocity
        filtered = [n for n in note_events if n[3] >= MIN_CONFIDENCE]

        # 2) Filter by minimum duration
        filtered = [n for n in filtered if (n[1] - n[0]) >= MIN_DURATION_SEC]

        # 3) Merge overlapping same-pitch notes (common with humming)
        filtered = merge_overlapping_notes(filtered)

        if not filtered:
            return {
                "success": False,
                "error": f"Notes detected but filtered out (try humming louder). "
                         f"Raw: {len(note_events)}, after filter: 0",
                "notes": [],
                "debug": {
                    "raw_count": len(note_events),
                    "min_confidence": MIN_CONFIDENCE,
                    "min_duration": MIN_DURATION_SEC,
                }
            }

        bpm = estimate_bpm(filtered)
        beat_duration = 60.0 / bpm
        key_info = detect_key([n[2] for n in filtered])
        min_start = min(n[0] for n in filtered)

        notes = []
        for start, end, pitch, velocity, pitch_bends in filtered:
            midi_pitch = int(pitch)
            duration = float(end - start)
            q = quantize_duration(duration, beat_duration)

            notes.append({
                "start_time": round(float(start - min_start), 3),
                "end_time": round(float(end - min_start), 3),
                "duration": round(duration, 3),
                "midi_pitch": int(midi_pitch),
                "note_name": midi_to_note_name(midi_pitch),
                "octave": int((midi_pitch // 12) - 1),
                "velocity": round(float(velocity), 3),
                "quantized_duration": q["name"],
                "quantized_beats": float(q["beats"]),
                "staff_position": int(midi_pitch - 60),
            })

        notes.sort(key=lambda n: n["start_time"])

        return {
            "success": True,
            "bpm": int(bpm),
            "key": key_info["key"],
            "mode": key_info["mode"],
            "time_signature": "4/4",
            "total_duration": round(float(max(n["end_time"] for n in notes)), 3),
            "note_count": int(len(notes)),
            "notes": notes,
            "debug": {
                "raw_note_count": int(len(note_events)),
                "filtered_note_count": int(len(notes)),
                "onset_threshold": float(ONSET_THRESHOLD),
                "frame_threshold": float(FRAME_THRESHOLD),
                "min_confidence": float(MIN_CONFIDENCE),
                "min_duration": float(MIN_DURATION_SEC),
            }
        }

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "HumToScore"}
