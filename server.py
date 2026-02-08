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
    return {"beats": closest[0], "name": closest[1]}


def estimate_bpm(notes: list) -> float:
    if len(notes) < 3:
        return 120.0

    onsets = [float(n[0]) for n in notes]
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


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix = ".mp3" if (audio.filename and audio.filename.endswith(".mp3")) else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from basic_pitch.inference import predict

        model_output, midi_data, note_events = predict(tmp_path)

        if not note_events:
            return {"success": False, "error": "No notes detected", "notes": []}

        bpm = estimate_bpm(note_events)
        beat_duration = 60.0 / bpm
        key_info = detect_key([int(n[2]) for n in note_events])
        min_start = float(min(n[0] for n in note_events))

        notes = []
        for start, end, pitch, velocity, pitch_bends in note_events:
            # Convert all numpy types to native Python types
            start = float(start)
            end = float(end)
            pitch = int(pitch)
            velocity = float(velocity)
            duration = end - start
            q = quantize_duration(duration, beat_duration)

            notes.append({
                "start_time": round(start - min_start, 3),
                "end_time": round(end - min_start, 3),
                "duration": round(duration, 3),
                "midi_pitch": pitch,
                "note_name": midi_to_note_name(pitch),
                "octave": (pitch // 12) - 1,
                "velocity": round(velocity, 3),
                "quantized_duration": q["name"],
                "quantized_beats": q["beats"],
                "staff_position": pitch - 60,
            })

        notes.sort(key=lambda n: n["start_time"])

        return {
            "success": True,
            "bpm": int(bpm),
            "key": key_info["key"],
            "mode": key_info["mode"],
            "time_signature": "4/4",
            "total_duration": round(float(max(n["end_time"] for n in notes)), 3),
            "note_count": len(notes),
            "notes": notes,
        }

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "HumToScore"}
