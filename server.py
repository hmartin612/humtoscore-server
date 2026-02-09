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

    # Generate beat grid from BPM
    beat_dur = winner["beat_duration"]
    max_onset = max(note_onsets) if note_onsets else 10.0
    beat_grid = []
    t = 0.0
    while t <= max_onset + beat_dur:
        beat_grid.append(round(t, 4))
        t += beat_dur

    return {
        "bpm": winner["bpm"],
        "time_signature": best_meter,
        "beat_duration": winner["beat_duration"],
        "beat_grid": beat_grid,
        "meter_debug": {
            "source": "onset_detection",
            "median_interval": round(median_interval, 4),
            "candidates": {m: {"avg_error_ms": round(r["avg_error"] * 1000, 1),
                               "bpm": r["bpm"]}
                          for m, r in results.items()},
            "winner": best_meter,
        }
    }


def detect_meter_from_taps(tap_times: list, note_onsets: list = None,
                           downbeat_indices: list = None) -> dict:
    """
    Detect BPM and time signature from user beat taps.
    
    With downbeat_indices: the user explicitly told us which taps are beat ONE.
    Count taps between consecutive downbeats = beats per bar = time signature.
    BPM = 60 / median inter-tap interval.
    
    This is the source of truth — no guessing, no statistics.
    """
    if len(tap_times) < 3:
        return {"bpm": 120, "time_signature": "4/4", "beat_duration": 0.5,
                "beat_grid": tap_times,
                "meter_debug": {"source": "taps", "reason": "too few taps"}}

    # ─── BPM from ALL inter-tap intervals (each tap = one beat) ───
    intervals = [tap_times[i+1] - tap_times[i] for i in range(len(tap_times)-1)]
    intervals = [i for i in intervals if 0.08 < i < 3.0]

    if not intervals:
        return {"bpm": 120, "time_signature": "4/4", "beat_duration": 0.5,
                "beat_grid": tap_times,
                "meter_debug": {"source": "taps", "reason": "no valid intervals"}}

    intervals.sort()
    beat_duration = intervals[len(intervals) // 2]

    bpm = 60.0 / beat_duration
    while bpm > 220: bpm /= 2
    while bpm < 40: bpm *= 2

    # ─── Time signature from downbeats ───
    time_sig = "4/4"
    bar_details = []

    if downbeat_indices and len(downbeat_indices) >= 2:
        # Count taps between consecutive downbeats
        bar_beat_counts = []
        for i in range(len(downbeat_indices) - 1):
            start_idx = downbeat_indices[i]
            end_idx = downbeat_indices[i + 1]
            beats_in_bar = end_idx - start_idx  # taps from this downbeat to next
            bar_beat_counts.append(beats_in_bar)
            bar_details.append({
                "bar": i + 1,
                "beats": beats_in_bar,
                "duration": round(tap_times[end_idx] - tap_times[start_idx], 4),
            })

        if bar_beat_counts:
            # Most common beat count = time signature
            from collections import Counter
            freq = Counter(bar_beat_counts)
            most_common_beats = freq.most_common(1)[0][0]

            sig_map = {2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4",
                       6: "6/8", 7: "7/8", 8: "8/8", 9: "9/8", 12: "12/8"}
            time_sig = sig_map.get(most_common_beats, f"{most_common_beats}/4")

            agree = sum(1 for b in bar_beat_counts if b == most_common_beats)
            print(f"[Meter] Downbeat-based: {time_sig}, "
                  f"{agree}/{len(bar_beat_counts)} bars agree, "
                  f"beats per bar: {bar_beat_counts}")
    else:
        # No downbeat info — fall back to grouping heuristic
        n_taps = len(tap_times)
        def bar_cv(bpb):
            if n_taps < bpb + 1: return float('inf')
            durs = [tap_times[i+bpb] - tap_times[i]
                    for i in range(0, n_taps - bpb, bpb)]
            if not durs: return float('inf')
            m = sum(durs) / len(durs)
            if m == 0: return float('inf')
            v = sum((d-m)**2 for d in durs) / len(durs)
            return (v**0.5) / m

        cv3 = bar_cv(3)
        cv4 = bar_cv(4)
        time_sig = "3/4" if cv3 < cv4 * 0.85 else "4/4"

    return {
        "bpm": round(bpm),
        "time_signature": time_sig,
        "beat_duration": beat_duration,
        "beat_grid": [round(t, 4) for t in tap_times],
        "meter_debug": {
            "source": "user_taps" + ("_with_downbeats" if downbeat_indices else ""),
            "tap_count": len(tap_times),
            "downbeat_count": len(downbeat_indices) if downbeat_indices else 0,
            "median_interval": round(beat_duration, 4),
            "bpm_raw": round(60.0 / beat_duration, 1),
            "bars": bar_details,
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
    # Parse beat taps — supports two formats:
    # Old: "0.500,1.000,1.500" (plain timestamps)
    # New: "D0.000,B0.500,B1.000,D1.500" (D=downbeat, B=beat)
    tap_times = []
    tap_downbeats = []  # indices into tap_times that are downbeats
    if beat_taps and beat_taps.strip():
        try:
            for t in beat_taps.split(','):
                t = t.strip()
                if not t:
                    continue
                is_down = False
                if t[0] in ('D', 'd'):
                    is_down = True
                    t = t[1:]
                elif t[0] in ('B', 'b'):
                    t = t[1:]
                time_val = float(t)
                if is_down:
                    tap_downbeats.append(len(tap_times))
                tap_times.append(time_val)
            print(f"[Taps] Parsed {len(tap_times)} taps, {len(tap_downbeats)} downbeats")
        except (ValueError, IndexError) as e:
            print(f"[Taps] Parse error: {e}, raw: {beat_taps[:100]}")
            tap_times = []
            tap_downbeats = []

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
            # USER TAPPED BEATS — use taps as source of truth
            meter_result = detect_meter_from_taps(tap_times, onsets, tap_downbeats)
        else:
            # No taps — estimate from note onsets
            meter_result = detect_meter_and_bpm(onsets)

        bpm = meter_result["bpm"]
        time_signature = meter_result["time_signature"]
        beat_duration = meter_result["beat_duration"]
        beat_grid = meter_result.get("beat_grid", [])
        min_start = raw_notes[0][0]

        # 8) Build final notes — TWO PATHS depending on whether we have a beat grid

        if beat_grid and len(beat_grid) >= 2:
            # ─── TAP-GRID QUANTIZATION ───
            # The beat grid from taps is the source of truth.
            # 1. Snap each note onset to the nearest beat/sub-beat position
            # 2. Duration = gap to next note's snapped position (not sustain length)
            # 3. Quantize that gap to standard note values

            # Build subdivision grid: each beat divided into 4 (sixteenth note resolution)
            sub_grid = []
            for i in range(len(beat_grid) - 1):
                b_start = beat_grid[i]
                b_end = beat_grid[i + 1]
                local_dur = b_end - b_start
                for sub in range(4):  # 0, 0.25, 0.5, 0.75 of beat
                    sub_grid.append(b_start + local_dur * (sub / 4.0))
            sub_grid.append(beat_grid[-1])  # add final beat
            # Extend grid a bit past the end for the last note
            if len(beat_grid) >= 2:
                avg_beat = sum(beat_grid[i+1] - beat_grid[i]
                              for i in range(len(beat_grid)-1)) / (len(beat_grid)-1)
                for extra in range(1, 9):
                    sub_grid.append(beat_grid[-1] + avg_beat * (extra / 4.0))

            def snap_to_grid(time_val):
                """Snap a time value to the nearest grid point."""
                best = min(sub_grid, key=lambda g: abs(g - time_val))
                return best

            # Snap all note onsets
            snapped_onsets = []
            for start, end, pitch_midi, confidence in raw_notes:
                snapped_start = snap_to_grid(start)
                snapped_onsets.append((snapped_start, start, end, pitch_midi, confidence))

            # Sort by snapped onset
            snapped_onsets.sort(key=lambda x: x[0])

            # Build notes: duration = gap between consecutive snapped onsets
            notes = []
            for i, (snap_start, raw_start, raw_end, pitch_midi, confidence) in enumerate(snapped_onsets):
                snapped = snap_to_scale(pitch_midi, root, scale)

                # Duration = time to next note's onset (rhythmic), not sustain
                if i < len(snapped_onsets) - 1:
                    next_snap = snapped_onsets[i + 1][0]
                    rhythmic_dur = next_snap - snap_start
                else:
                    # Last note: use sustain duration as fallback
                    rhythmic_dur = raw_end - raw_start

                # Clamp to reasonable range
                rhythmic_dur = max(rhythmic_dur, beat_duration * 0.2)

                q = quantize_duration(rhythmic_dur, beat_duration)

                notes.append({
                    "start_time": round(float(snap_start - min_start), 3),
                    "end_time": round(float(snap_start - min_start + rhythmic_dur), 3),
                    "duration": round(float(rhythmic_dur), 3),
                    "midi_pitch": int(snapped),
                    "note_name": midi_to_note_name(snapped),
                    "octave": int((snapped // 12) - 1),
                    "velocity": round(float(confidence), 3),
                    "quantized_duration": q["name"],
                    "quantized_beats": float(q["beats"]),
                    "staff_position": int(snapped - 60),
                    "raw_pitch_midi": round(float(pitch_midi), 2),
                    "raw_onset": round(float(raw_start - min_start), 3),
                    "snapped_onset": round(float(snap_start - min_start), 3),
                    "snap_offset_ms": round(abs(snap_start - raw_start) * 1000, 1),
                })

            print(f"[Quantize] Grid-snapped {len(notes)} notes to {len(sub_grid)} grid points")

        else:
            # ─── FALLBACK: old sustain-based quantization (no tap grid) ───
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
            "beat_grid": meter_result.get("beat_grid", []),
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
                "downbeats_received": len(tap_downbeats),
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
