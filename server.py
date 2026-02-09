"""
HumToScore - Basic Pitch Server
Receives WAV/MP3 audio, runs Spotify's Basic Pitch, returns detected notes as JSON.
Deployed on Railway.

Key accuracy features:
- Round pitch instead of truncate
- Snap to detected key's scale (auto-tune)
- Merge nearby same-pitch notes
- Filter ghost notes by confidence + duration
- Collapse repeated same-pitch notes (humming wobble)
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
MIN_CONFIDENCE = 0.5    # Higher = fewer ghost notes (was 0.4)
MIN_DURATION_SEC = 0.1  # Drop notes shorter than 100ms (was 0.08)
ONSET_THRESHOLD = 0.5   # How aggressively to split notes
FRAME_THRESHOLD = 0.3   # Pitch detection strictness
MIN_NOTE_LENGTH = 58    # ~67ms per frame, 58 frames ≈ minimum note

# Scale definitions (intervals from root)
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}  # natural minor
# For snapping, we also allow harmonic minor notes
MINOR_SCALE_EXTENDED = {0, 2, 3, 5, 7, 8, 10, 11}


def midi_to_note_name(midi_pitch: int) -> str:
    octave = (midi_pitch // 12) - 1
    return f"{NOTE_NAMES[midi_pitch % 12]}{octave}"


def snap_to_scale(midi_pitch_float: float, root: int, scale_intervals: set) -> int:
    """
    Auto-tune: snap a floating-point MIDI pitch to the nearest note
    in the given scale.
    
    1. Round to nearest semitone
    2. If it's in the scale, keep it
    3. If not, try ±1 semitone and pick whichever is in the scale
    4. If neither ±1 is in scale, just use the rounded value
    """
    rounded = round(midi_pitch_float)
    
    # Check if rounded pitch is in scale
    pc = (rounded - root) % 12
    if pc in scale_intervals:
        return rounded
    
    # Try one semitone up and down
    up = rounded + 1
    down = rounded - 1
    up_in = ((up - root) % 12) in scale_intervals
    down_in = ((down - root) % 12) in scale_intervals
    
    if up_in and down_in:
        # Both neighbors in scale — pick closer to original float
        if abs(midi_pitch_float - up) < abs(midi_pitch_float - down):
            return up
        return down
    elif up_in:
        return up
    elif down_in:
        return down
    
    # Neither neighbor in scale (shouldn't happen with diatonic scales)
    return rounded


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
    """Detect key using pitch class profile correlation."""
    if not midi_pitches:
        return {"key": "C", "mode": "major", "root": 0}

    pc = [0] * 12
    for p in midi_pitches:
        # Use rounded pitch for key detection
        pc[round(p) % 12] += 1

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

    return {"key": NOTE_NAMES[best_key], "mode": best_mode, "root": best_key}


def merge_overlapping_notes(note_events: list) -> list:
    """Merge overlapping/adjacent notes at the same pitch."""
    if not note_events:
        return note_events

    sorted_notes = sorted(note_events, key=lambda n: (n[0], n[2]))
    merged = [list(sorted_notes[0])]

    for start, end, pitch, velocity, bends in sorted_notes[1:]:
        prev = merged[-1]
        # Same pitch (after rounding), starts within 150ms of previous end → merge
        if round(pitch) == round(prev[2]) and start <= prev[1] + 0.15:
            prev[1] = max(prev[1], end)
            prev[3] = max(prev[3], velocity)
        else:
            merged.append([start, end, pitch, velocity, bends])

    return [tuple(n) for n in merged]


def collapse_repeated_pitches(notes: list) -> list:
    """
    When humming, pitch detection often splits one held note into
    multiple consecutive notes at the same pitch. Collapse them.
    A sequence like E4 E4 E4 becomes one longer E4.
    """
    if not notes:
        return notes
    
    collapsed = [list(notes[0])]
    
    for n in notes[1:]:
        prev = collapsed[-1]
        # Same rounded pitch AND starts right after previous ends (within 200ms gap)
        if round(n[2]) == round(prev[2]) and (n[0] - prev[1]) < 0.2:
            prev[1] = n[1]  # extend end
            prev[3] = max(prev[3], n[3])  # keep higher velocity
        else:
            collapsed.append(list(n))
    
    return [tuple(n) for n in collapsed]


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix = ".mp3" if (audio.filename and audio.filename.endswith(".mp3")) else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from basic_pitch.inference import predict

        # Run Basic Pitch
        model_output, midi_data, note_events = predict(
            tmp_path,
            onset_threshold=ONSET_THRESHOLD,
            frame_threshold=FRAME_THRESHOLD,
            minimum_note_length=MIN_NOTE_LENGTH,
        )

        if not note_events:
            return {"success": False, "error": "No notes detected", "notes": []}

        raw_count = len(note_events)

        # ─── POST-PROCESSING PIPELINE ───────────────────────

        # 1) Filter by confidence
        filtered = [n for n in note_events if n[3] >= MIN_CONFIDENCE]

        # 2) Filter by minimum duration
        filtered = [n for n in filtered if (n[1] - n[0]) >= MIN_DURATION_SEC]

        # 3) Merge overlapping same-pitch notes
        filtered = merge_overlapping_notes(filtered)

        # 4) Collapse consecutive same-pitch notes (humming splits)
        filtered = collapse_repeated_pitches(filtered)

        if not filtered:
            return {
                "success": False,
                "error": f"Notes filtered out (try singing louder/clearer). Raw: {raw_count}",
                "notes": [],
                "debug": {"raw_count": raw_count}
            }

        # 5) Detect key BEFORE snapping (using raw float pitches)
        key_info = detect_key([n[2] for n in filtered])
        root = key_info["root"]
        scale = MINOR_SCALE_EXTENDED if key_info["mode"] == "minor" else MAJOR_SCALE

        # 6) Estimate BPM
        bpm = estimate_bpm(filtered)
        beat_duration = 60.0 / bpm
        min_start = min(n[0] for n in filtered)

        # 7) Build note list with AUTO-TUNE (snap to scale)
        notes = []
        for start, end, pitch, velocity, pitch_bends in filtered:
            # KEY FIX: round() not int() — and snap to scale
            midi_pitch = snap_to_scale(float(pitch), root, scale)
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
                "raw_pitch": round(float(pitch), 2),  # debug: original pitch
            })

        notes.sort(key=lambda n: n["start_time"])

        # 8) One more pass: if consecutive notes ended up at same pitch
        #    after snapping, merge their durations
        final_notes = [notes[0]]
        for n in notes[1:]:
            prev = final_notes[-1]
            if (n["midi_pitch"] == prev["midi_pitch"] and 
                n["start_time"] - prev["end_time"] < 0.15):
                # Merge: extend previous note
                prev["end_time"] = n["end_time"]
                prev["duration"] = round(prev["end_time"] - prev["start_time"], 3)
                q = quantize_duration(prev["duration"], beat_duration)
                prev["quantized_duration"] = q["name"]
                prev["quantized_beats"] = float(q["beats"])
                prev["velocity"] = max(prev["velocity"], n["velocity"])
            else:
                final_notes.append(n)

        return {
            "success": True,
            "bpm": int(bpm),
            "key": key_info["key"],
            "mode": key_info["mode"],
            "time_signature": "4/4",
            "total_duration": round(float(max(n["end_time"] for n in final_notes)), 3),
            "note_count": int(len(final_notes)),
            "notes": final_notes,
            "debug": {
                "raw_note_count": int(raw_count),
                "after_filter": int(len(filtered)),
                "after_snap_merge": int(len(final_notes)),
                "detected_key": f"{key_info['key']} {key_info['mode']}",
                "scale_snapping": True,
            }
        }

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "HumToScore"}
