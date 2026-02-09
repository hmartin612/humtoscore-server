"""
HumToScore - CREPE Pitch Server
Uses torchcrepe for monophonic pitch detection (built for voice).
Our own note segmentation from the raw pitch contour.
Same JSON API as the Basic Pitch version — Unity code unchanged.

Pipeline:
  Audio → CREPE (pitch + confidence per 5ms frame)
       → filter silence / low confidence
       → median smooth pitch
       → segment into notes (detect pitch changes)
       → snap to scale (auto-tune)
       → quantize durations
       → JSON response
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import numpy as np
import torch
import soundfile as sf

app = FastAPI(title="HumToScore")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ─── Tuning knobs ───────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3    # Below this = silence/unvoiced
HOP_SECONDS = 0.005           # 5ms per frame (200 frames/sec)
PITCH_CHANGE_THRESHOLD = 0.8  # Semitones — if pitch jumps more than this, new note
MIN_NOTE_FRAMES = 12          # ~60ms minimum note (12 frames × 5ms)
SMOOTHING_WINDOW = 5          # Median filter window for pitch smoothing
FMIN = 65.0                   # C2 — lowest humming pitch
FMAX = 600.0                  # ~D5 — highest reasonable sing

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10, 11}  # natural + harmonic minor


def hz_to_midi(hz: float) -> float:
    """Convert frequency in Hz to MIDI note number (float)."""
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def midi_to_note_name(midi_pitch: int) -> str:
    octave = (midi_pitch // 12) - 1
    return f"{NOTE_NAMES[midi_pitch % 12]}{octave}"


def snap_to_scale(midi_float: float, root: int, scale: set) -> int:
    """Snap a MIDI pitch to the nearest note in scale."""
    rounded = round(midi_float)
    pc = (rounded - root) % 12
    if pc in scale:
        return rounded

    up, down = rounded + 1, rounded - 1
    up_in = ((up - root) % 12) in scale
    down_in = ((down - root) % 12) in scale

    if up_in and down_in:
        return up if abs(midi_float - up) < abs(midi_float - down) else down
    if up_in:
        return up
    if down_in:
        return down
    return rounded


def detect_key(midi_pitches: list) -> dict:
    """Key detection via pitch class profile."""
    if not midi_pitches:
        return {"key": "C", "mode": "major", "root": 0}

    pc = [0] * 12
    for p in midi_pitches:
        pc[round(p) % 12] += 1

    major = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    minor = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

    best_key, best_mode, best_score = 0, "major", -1
    for root in range(12):
        maj = sum(pc[i] * major[(i - root) % 12] for i in range(12))
        mn = sum(pc[i] * minor[(i - root) % 12] for i in range(12))
        if maj > best_score:
            best_score, best_key, best_mode = maj, root, "major"
        if mn > best_score:
            best_score, best_key, best_mode = mn, root, "minor"

    return {"key": NOTE_NAMES[best_key], "mode": best_mode, "root": best_key}


def quantize_duration(duration: float, beat_duration: float) -> dict:
    beats = duration / beat_duration
    standard = [
        (0.25, "sixteenth"), (0.5, "eighth"), (0.75, "dotted_eighth"),
        (1.0, "quarter"), (1.5, "dotted_quarter"), (2.0, "half"),
        (3.0, "dotted_half"), (4.0, "whole"),
    ]
    closest = min(standard, key=lambda x: abs(x[0] - beats))
    return {"beats": float(closest[0]), "name": closest[1]}


def detect_meter_and_bpm(note_onsets: list) -> dict:
    """
    Detect time signature and BPM from note onset times.
    
    Algorithm:
    1. Compute inter-onset intervals
    2. Find the median interval (≈ one beat)
    3. Build grids for 4/4 and 3/4 using that beat duration
    4. Measure how well each onset fits each grid (quantization error)
    5. Lowest total error wins
    6. Clamp BPM to human range (60-180)
    """
    if len(note_onsets) < 3:
        return {"bpm": 120, "time_signature": "4/4", "beat_duration": 0.5,
                "meter_debug": {"reason": "too few notes, defaulting"}}

    # 1) Inter-onset intervals
    intervals = []
    for i in range(len(note_onsets) - 1):
        gap = note_onsets[i + 1] - note_onsets[i]
        if 0.1 < gap < 3.0:  # filter out tiny gaps and huge pauses
            intervals.append(gap)

    if not intervals:
        return {"bpm": 120, "time_signature": "4/4", "beat_duration": 0.5,
                "meter_debug": {"reason": "no valid intervals"}}

    # 2) Find the median interval = likely beat duration
    intervals.sort()
    median_interval = intervals[len(intervals) // 2]

    # 3) Define candidate meters
    candidates = {
        "4/4": {
            "beats_per_bar": 4,
            "beat_duration": median_interval,
            # Valid note positions: quarter, eighth, sixteenth
            "subdivisions": [1.0, 0.5, 0.25],
        },
        "3/4": {
            "beats_per_bar": 3,
            "beat_duration": median_interval,
            "subdivisions": [1.0, 0.5, 0.25],
        },
    }

    # 4) For each candidate, compute quantization error
    results = {}
    max_time = note_onsets[-1] + median_interval

    for meter, config in candidates.items():
        beat = config["beat_duration"]
        smallest_sub = min(config["subdivisions"]) * beat

        # Build the grid
        grid = []
        t = 0.0
        while t <= max_time + beat:
            grid.append(t)
            t += smallest_sub

        # Measure each onset's distance to nearest grid line
        total_error = 0.0
        for onset in note_onsets:
            min_dist = float('inf')
            for gp in grid:
                d = abs(onset - gp)
                if d < min_dist:
                    min_dist = d
                # Early exit: grid is sorted, if we've passed onset, stop
                if gp > onset + beat:
                    break
            total_error += min_dist

        avg_error = total_error / len(note_onsets)

        # Compute BPM and clamp to human range
        bpm = 60.0 / beat
        while bpm > 180:
            bpm /= 2
        while bpm < 60:
            bpm *= 2

        results[meter] = {
            "avg_error": avg_error,
            "bpm": round(bpm),
            "beat_duration": beat,
        }

    # 5) Winner = lowest average error
    best_meter = min(results, key=lambda m: results[m]["avg_error"])
    winner = results[best_meter]

    return {
        "bpm": winner["bpm"],
        "time_signature": best_meter,
        "beat_duration": winner["beat_duration"],
        "meter_debug": {
            "source": "onset_detection",
            "median_interval": round(median_interval, 4),
            "candidates": {m: {"avg_error_ms": round(r["avg_error"] * 1000, 1),
                               "bpm": r["bpm"]}
                          for m, r in results.items()},
            "winner": best_meter,
        }
    }


def detect_meter_from_taps(tap_times: list, note_onsets: list = None) -> dict:
    """
    Detect BPM and time signature by aligning user taps with note onsets.
    
    The key insight: taps tell us where beats are, note onsets tell us where
    musical events are. By overlaying them we get:
    
    1. BPM: median inter-tap interval (user's felt tempo)
    2. Beat grid: the taps ARE the grid (no guessing needed)
    3. Time signature: how note onsets cluster relative to the beat grid.
       If notes consistently land on groups of 3 beats → 3/4.
       If groups of 4 → 4/4.
    4. Quantization: each note onset snaps to the nearest beat subdivision
       using the tap grid, not a guessed grid.
    """
    if len(tap_times) < 3:
        return {"bpm": 120, "time_signature": "4/4", "beat_duration": 0.5,
                "meter_debug": {"source": "taps", "reason": "too few taps"}}

    # ─── Step 1: BPM from tap intervals ───
    intervals = [tap_times[i+1] - tap_times[i] for i in range(len(tap_times)-1)]
    intervals = [i for i in intervals if 0.1 < i < 3.0]

    if not intervals:
        return {"bpm": 120, "time_signature": "4/4", "beat_duration": 0.5,
                "meter_debug": {"source": "taps", "reason": "no valid intervals"}}

    intervals.sort()
    beat_duration = intervals[len(intervals) // 2]

    bpm = 60.0 / beat_duration
    while bpm > 180: bpm /= 2
    while bpm < 60: bpm *= 2

    # ─── Step 2: Build the beat grid from taps ───
    # Each tap is a beat. The grid is literally the tap times.
    beat_grid = list(tap_times)

    # ─── Step 3: Align note onsets to beat grid ───
    # For each note onset, find which beat it's closest to and compute
    # its position as a fraction of the beat (0.0 = on the beat, 0.5 = halfway)
    note_beat_positions = []
    if note_onsets and len(note_onsets) > 0:
        for onset in note_onsets:
            # Find the closest beat before this onset
            best_beat_idx = 0
            for bi in range(len(beat_grid)):
                if beat_grid[bi] <= onset:
                    best_beat_idx = bi
                else:
                    break

            # Position within beat (0.0 to ~1.0)
            if best_beat_idx < len(beat_grid) - 1:
                beat_start = beat_grid[best_beat_idx]
                beat_end = beat_grid[best_beat_idx + 1]
                local_beat_dur = beat_end - beat_start
                if local_beat_dur > 0:
                    frac = (onset - beat_start) / local_beat_dur
                    note_beat_positions.append({
                        "onset": onset,
                        "beat_index": best_beat_idx,
                        "beat_fraction": round(frac, 3),
                    })

    # ─── Step 4: Time signature from beat grouping ───
    # Try grouping beats into bars of 3 and 4.
    # Count how many notes land on beat 0 (downbeat) of each bar.
    # More downbeat-heavy = better fit.
    # Also check consistency of bar durations.

    n_taps = len(tap_times)

    def score_grouping(beats_per_bar):
        """Score a grouping: lower = better fit.
        Combines bar duration consistency + note alignment to downbeats."""
        if n_taps < beats_per_bar + 1:
            return float('inf'), {}

        # Bar duration consistency
        bar_durations = []
        for i in range(0, n_taps - beats_per_bar, beats_per_bar):
            bar_dur = tap_times[i + beats_per_bar] - tap_times[i]
            bar_durations.append(bar_dur)

        if not bar_durations:
            return float('inf'), {}

        mean_dur = sum(bar_durations) / len(bar_durations)
        if mean_dur == 0:
            return float('inf'), {}

        variance = sum((d - mean_dur) ** 2 for d in bar_durations) / len(bar_durations)
        cv = (variance ** 0.5) / mean_dur  # coefficient of variation

        # Note onset alignment: what fraction of notes land near a downbeat?
        downbeat_score = 0
        if note_beat_positions:
            for nbp in note_beat_positions:
                bar_beat = nbp["beat_index"] % beats_per_bar
                # Beat 0 of each bar = downbeat
                if bar_beat == 0 and nbp["beat_fraction"] < 0.15:
                    downbeat_score += 1

            downbeat_ratio = downbeat_score / len(note_beat_positions)
        else:
            downbeat_ratio = 0

        # Combined score: lower cv is better, higher downbeat_ratio is better
        combined = cv - (downbeat_ratio * 0.3)

        return combined, {
            "cv": round(cv, 4),
            "downbeat_ratio": round(downbeat_ratio, 3),
            "bar_count": len(bar_durations),
            "mean_bar_duration": round(mean_dur, 4),
        }

    score_3, detail_3 = score_grouping(3)
    score_4, detail_4 = score_grouping(4)

    if score_3 < score_4 * 0.85:  # 3/4 needs to be notably better
        time_sig = "3/4"
    else:
        time_sig = "4/4"

    return {
        "bpm": round(bpm),
        "time_signature": time_sig,
        "beat_duration": beat_duration,
        "beat_grid": [round(t, 4) for t in beat_grid],
        "note_beat_positions": note_beat_positions[:20],  # cap for response size
        "meter_debug": {
            "source": "user_taps_aligned",
            "tap_count": len(tap_times),
            "note_count_aligned": len(note_beat_positions),
            "median_interval": round(beat_duration, 4),
            "grouping_3/4": detail_3,
            "grouping_4/4": detail_4,
            "score_3/4": round(score_3, 4) if score_3 != float('inf') else "inf",
            "score_4/4": round(score_4, 4) if score_4 != float('inf') else "inf",
            "winner": time_sig,
        }
    }


def segment_notes(midi_pitches: np.ndarray, confidences: np.ndarray,
                  hop_sec: float) -> list:
    """
    Segment a continuous pitch contour into discrete notes.

    Returns list of (start_sec, end_sec, avg_midi_pitch, avg_confidence).

    Logic:
    1. Mark frames as voiced (confidence > threshold) or silent
    2. Within voiced regions, detect note boundaries when pitch
       jumps more than PITCH_CHANGE_THRESHOLD semitones
    3. For each segment, compute the median pitch (robust to wobble)
    4. Drop segments shorter than MIN_NOTE_FRAMES
    """
    n_frames = len(midi_pitches)
    notes = []

    # Find voiced regions
    voiced = confidences >= CONFIDENCE_THRESHOLD

    # Walk through frames and segment
    in_note = False
    note_start = 0
    note_frames = []

    for i in range(n_frames):
        if voiced[i]:
            if not in_note:
                # Start a new note
                in_note = True
                note_start = i
                note_frames = [midi_pitches[i]]
            else:
                # Check if pitch changed significantly
                current_pitch = midi_pitches[i]
                recent_median = np.median(note_frames[-min(10, len(note_frames)):])

                if abs(current_pitch - recent_median) > PITCH_CHANGE_THRESHOLD:
                    # End current note, start new one
                    if len(note_frames) >= MIN_NOTE_FRAMES:
                        med_pitch = float(np.median(note_frames))
                        avg_conf = float(np.mean(
                            confidences[note_start:note_start + len(note_frames)]))
                        notes.append((
                            note_start * hop_sec,
                            (note_start + len(note_frames)) * hop_sec,
                            med_pitch,
                            avg_conf
                        ))
                    # Start new note
                    note_start = i
                    note_frames = [current_pitch]
                else:
                    note_frames.append(current_pitch)
        else:
            # Silence — end current note if any
            if in_note and len(note_frames) >= MIN_NOTE_FRAMES:
                med_pitch = float(np.median(note_frames))
                avg_conf = float(np.mean(
                    confidences[note_start:note_start + len(note_frames)]))
                notes.append((
                    note_start * hop_sec,
                    (note_start + len(note_frames)) * hop_sec,
                    med_pitch,
                    avg_conf
                ))
            in_note = False
            note_frames = []

    # Don't forget last note
    if in_note and len(note_frames) >= MIN_NOTE_FRAMES:
        med_pitch = float(np.median(note_frames))
        avg_conf = float(np.mean(
            confidences[note_start:note_start + len(note_frames)]))
        notes.append((
            note_start * hop_sec,
            (note_start + len(note_frames)) * hop_sec,
            med_pitch,
            avg_conf
        ))

    return notes


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    beat_taps: str = Form(default=""),
):
    # Parse beat taps from comma-separated string
    tap_times = []
    if beat_taps and beat_taps.strip():
        try:
            tap_times = [float(t.strip()) for t in beat_taps.split(',') if t.strip()]
        except ValueError:
            tap_times = []

    suffix = ".mp3" if (audio.filename and audio.filename.endswith(".mp3")) else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import torchcrepe
        import resampy

        # Load audio with soundfile (no torchcodec dependency)
        audio_np, sr = sf.read(tmp_path, dtype='float32')

        # Convert stereo to mono
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Resample to 16kHz (CREPE expects this)
        if sr != 16000:
            audio_np = resampy.resample(audio_np, sr, 16000)
            sr = 16000

        # Convert to torch tensor [1, samples]
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        # ─── CREPE PREDICTION ───────────────────────────────
        hop_length = int(sr * HOP_SECONDS)  # 80 samples at 16kHz = 5ms

        # Predict pitch + periodicity (confidence)
        # Using Viterbi decoding to prevent octave jumps
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length,
            FMIN,
            FMAX,
            model='tiny',
            batch_size=1024,
            device='cpu',
            return_periodicity=True,
            decoder=torchcrepe.decode.viterbi,
        )

        # Move to numpy
        pitch_np = pitch.squeeze().numpy()
        conf_np = periodicity.squeeze().numpy()

        # ─── POST-PROCESSING ────────────────────────────────

        # 1) Filter silence (CREPE assigns pitch to silent regions)
        # Compute RMS energy per frame to detect silence
        frame_size = hop_length
        n_frames = len(pitch_np)
        audio_np = audio_tensor.squeeze().numpy()

        rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio_np))
            if end > start:
                rms[i] = np.sqrt(np.mean(audio_np[start:end] ** 2))

        # Zero out confidence for silent frames
        silence_threshold = np.max(rms) * 0.02  # 2% of max = silence
        conf_np[rms < silence_threshold] = 0.0

        # 2) Median filter the confidence to remove noise
        from scipy.ndimage import median_filter
        conf_np = median_filter(conf_np, size=SMOOTHING_WINDOW)

        # 3) Convert pitch Hz → MIDI (float)
        midi_pitches = np.array([hz_to_midi(p) for p in pitch_np])

        # 4) Median filter the pitch to smooth wobble
        # Only filter voiced frames to prevent bleed
        voiced_mask = conf_np >= CONFIDENCE_THRESHOLD
        if voiced_mask.any():
            voiced_pitches = midi_pitches.copy()
            voiced_pitches[~voiced_mask] = np.nan

            # Manual median filter ignoring NaN
            filtered = voiced_pitches.copy()
            hw = SMOOTHING_WINDOW // 2
            for i in range(len(filtered)):
                if voiced_mask[i]:
                    window = voiced_pitches[max(0, i-hw):i+hw+1]
                    valid = window[~np.isnan(window)]
                    if len(valid) > 0:
                        filtered[i] = np.median(valid)
            midi_pitches = filtered

        # 5) Segment into notes
        raw_notes = segment_notes(midi_pitches, conf_np, HOP_SECONDS)

        if not raw_notes:
            return {
                "success": False,
                "error": "No notes detected. Try humming louder and more clearly.",
                "notes": [],
                "debug": {
                    "total_frames": int(n_frames),
                    "voiced_frames": int(voiced_mask.sum()),
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                }
            }

        # 6) Detect key from raw note pitches
        key_info = detect_key([n[2] for n in raw_notes])
        root = key_info["root"]
        scale = MINOR_SCALE if key_info["mode"] == "minor" else MAJOR_SCALE

        # 7) Detect meter AND BPM
        onsets = [n[0] for n in raw_notes]

        if tap_times and len(tap_times) >= 3:
            # USER TAPPED BEATS — align taps with note onsets
            meter_result = detect_meter_from_taps(tap_times, onsets)
        else:
            # No taps — estimate from note onsets
            meter_result = detect_meter_and_bpm(onsets)

        bpm = meter_result["bpm"]
        time_signature = meter_result["time_signature"]
        beat_duration = meter_result["beat_duration"]
        min_start = raw_notes[0][0]

        # 8) Build final notes with scale snapping
        notes = []
        for start, end, pitch_midi, confidence in raw_notes:
            snapped = snap_to_scale(pitch_midi, root, scale)
            duration = end - start
            q = quantize_duration(duration, beat_duration)

            notes.append({
                "start_time": round(float(start - min_start), 3),
                "end_time": round(float(end - min_start), 3),
                "duration": round(float(duration), 3),
                "midi_pitch": int(snapped),
                "note_name": midi_to_note_name(snapped),
                "octave": int((snapped // 12) - 1),
                "velocity": round(float(confidence), 3),
                "quantized_duration": q["name"],
                "quantized_beats": float(q["beats"]),
                "staff_position": int(snapped - 60),
                "raw_pitch_midi": round(float(pitch_midi), 2),
            })

        # 9) Merge consecutive same-pitch notes (post-snap duplicates)
        final = [notes[0]]
        for n in notes[1:]:
            prev = final[-1]
            if (n["midi_pitch"] == prev["midi_pitch"] and
                    n["start_time"] - prev["end_time"] < 0.1):
                prev["end_time"] = n["end_time"]
                prev["duration"] = round(prev["end_time"] - prev["start_time"], 3)
                q = quantize_duration(prev["duration"], beat_duration)
                prev["quantized_duration"] = q["name"]
                prev["quantized_beats"] = float(q["beats"])
            else:
                final.append(n)

        return {
            "success": True,
            "bpm": int(bpm),
            "key": key_info["key"],
            "mode": key_info["mode"],
            "time_signature": time_signature,
            "total_duration": round(float(max(n["end_time"] for n in final)), 3),
            "note_count": int(len(final)),
            "notes": final,
            "debug": {
                "engine": "CREPE (torchcrepe)",
                "total_frames": int(n_frames),
                "voiced_frames": int(voiced_mask.sum()),
                "raw_segments": int(len(raw_notes)),
                "after_merge": int(len(final)),
                "detected_key": f"{key_info['key']} {key_info['mode']}",
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "pitch_change_threshold": PITCH_CHANGE_THRESHOLD,
                "beat_taps_received": len(tap_times),
                "meter": meter_result["meter_debug"],
            }
        }

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "HumToScore", "engine": "CREPE"}


@app.get("/")
async def root():
    return {
        "service": "HumToScore Transcription Server",
        "engine": "CREPE (torchcrepe)",
        "version": "v4",
        "endpoint": "POST /transcribe (multipart form, field: 'audio')",
        "params": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "pitch_change_threshold_semitones": PITCH_CHANGE_THRESHOLD,
            "min_note_ms": MIN_NOTE_FRAMES * HOP_SECONDS * 1000,
            "frequency_range_hz": f"{FMIN}-{FMAX}",
            "smoothing_window": SMOOTHING_WINDOW,
            "decoder": "viterbi",
            "model": "tiny",
        }
    }
